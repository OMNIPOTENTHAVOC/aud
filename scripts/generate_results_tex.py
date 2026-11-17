r"""
Reads the three JSON summaries the pipeline produces --

    outputs/predictions/train_summary.json           (train.py)
    outputs/predictions/eval_summary.json             (evaluate.py)
    outputs/predictions/interpretability_summary.json (interpretability/run_interpretability.py)

-- and writes TWO files, kept separate because main.tex \input's them from
two different places (macros need to exist before the abstract; the table
needs to physically sit in the Results section) and \newcommand raises a
"already defined" LaTeX error if the same macro is \input twice:

    paper/macros.tex       \newcommand macros for every headline number
                            used in the paper's prose. \input near the top
                            of main.tex, before \begin{document}'s content.
    paper/results_table.tex  the per-attack-type table (EER + Grad-CAM
                            entropy + SSIM consistency) plus the Pearson
                            correlation between per-attack EER and
                            per-attack mean entropy (the
                            "saliency-consistency correlation" the
                            codebase's docstrings point to). \input inside
                            the Results section.

Run any subset of the three upstream scripts first; whichever summaries
exist get used, and macros for anything missing are emitted as literal
"TODO" strings so paper/main.tex still compiles -- rerun this script after
each upstream step to progressively fill the paper in.

Run from the project root:

    python scripts/generate_results_tex.py
"""

import json
import sys
from pathlib import Path

# scripts/ is not the project root -- python only puts this file's own
# directory on sys.path by default, so "from configs import config" would
# fail (ModuleNotFoundError: No module named 'configs') without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs import config

PRED_DIR = config.PROJECT_ROOT / "outputs" / "predictions"
FIG_DIR = config.PROJECT_ROOT / "outputs" / "figures"
PAPER_DIR = config.PROJECT_ROOT / "paper"


def _load(name):
    path = PRED_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _pearsonr(xs, ys):
    """Pearson correlation, pure-numpy (no scipy dependency)."""
    import numpy as np
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if len(xs) < 2 or np.std(xs) == 0 or np.std(ys) == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def _macro(name, value, fmt="{:.4f}"):
    rendered = fmt.format(value) if isinstance(value, (int, float)) else str(value)
    return f"\\newcommand{{\\{name}}}{{{rendered}}}\n"


def _describe_correlation(r):
    """Plain-language direction/strength label using the common rule-of-thumb
    |r| bands (weak <0.3, moderate 0.3-0.5, strong >0.5) -- a convention,
    not a statistical test; stated as such in the drafted sentence."""
    direction = "positive" if r >= 0 else "negative"
    magnitude = abs(r)
    if magnitude < 0.3:
        strength = "weak"
    elif magnitude < 0.5:
        strength = "moderate"
    else:
        strength = "strong"
    return direction, strength


def _write_autodraft_discussion(train_summary, eval_summary, interp_summary):
    """Drafts the Discussion/Results observations that are directly
    computable from the numbers themselves (not the causal "why" behind
    them, which stays a human judgment call). Written to a SEPARATE file,
    not \\input by main.tex, so an autodraft never silently becomes the
    paper's Discussion without a human reading it first. Only writes
    observations whose inputs actually exist -- partial pipeline runs get
    a partial (not crashing) autodraft.
    """
    if not (eval_summary and interp_summary):
        return  # nothing computable yet

    notes = [
        "% AUTO-DRAFTED observations from your actual pipeline output.\n",
        "% These are NOT wired into main.tex on purpose -- read each one,\n",
        "% check it against the numbers, and hand-write/edit before using it\n",
        "% in the paper. Correlation and comparison claims below use plain\n",
        "% rule-of-thumb thresholds, not a statistical significance test.\n\n",
    ]

    per_attack_eer = eval_summary["per_attack"]
    attack_ids = sorted(a for a in interp_summary if not a.startswith("_"))

    # 1. Hardest / easiest attack type by EER.
    eer_by_attack = {aid: per_attack_eer[aid]["eer"] for aid in attack_ids if aid in per_attack_eer}
    if eer_by_attack:
        hardest = max(eer_by_attack, key=eer_by_attack.get)
        easiest = min(eer_by_attack, key=eer_by_attack.get)
        notes.append(
            f"% Hardest/easiest attack by EER:\n"
            f"% {hardest} is hardest (EER={100*eer_by_attack[hardest]:.2f}%); "
            f"{easiest} is easiest (EER={100*eer_by_attack[easiest]:.2f}%).\n"
            f"% Draft sentence: \"Detection difficulty varies substantially across "
            f"attack types, from {100*eer_by_attack[easiest]:.2f}\\% EER on {easiest} "
            f"to {100*eer_by_attack[hardest]:.2f}\\% on {hardest}.\"\n"
            f"% TODO: look up what {hardest} and {easiest} actually are (TTS vs VC, "
            f"vocoder family) in the ASVspoof2019 evaluation plan before claiming why.\n\n"
        )

    # 2. Generalization gap: dev EER (model selection) vs eval EER (unseen attacks).
    #    Guarded with .get() rather than direct indexing: a train_summary.json
    #    backfilled from a checkpoint (scripts/backfill_train_summary.py)
    #    always has best_dev_eer, but this stays defensive in case of an even
    #    more partial file rather than crashing the whole generator over one
    #    optional note.
    if train_summary and eval_summary and train_summary.get("best_dev_eer") is not None:
        dev_eer = train_summary["best_dev_eer"]
        eval_eer = eval_summary["overall"]["eer"]
        gap_ratio = eval_eer / dev_eer if dev_eer > 0 else float("inf")
        notes.append(
            f"% Generalization gap (dev EER vs eval EER):\n"
            f"% dev EER={100*dev_eer:.2f}%, eval EER={100*eval_eer:.2f}% "
            f"(eval is {gap_ratio:.2f}x dev).\n"
            f"% Draft sentence, if the gap is large: \"Eval EER "
            f"({100*eval_eer:.2f}\\%) is {gap_ratio:.1f}x dev EER "
            f"({100*dev_eer:.2f}\\%), consistent with the six training attack "
            f"types not fully representing the thirteen unseen eval attack types.\"\n"
            f"% TODO: judge for yourself whether {gap_ratio:.2f}x counts as \"large\" "
            f"for this task -- no universal threshold exists.\n\n"
        )

    # 3. Entropy-EER correlation, in plain language.
    eers, entropies = [], []
    for aid in attack_ids:
        if aid in per_attack_eer:
            eers.append(100 * per_attack_eer[aid]["eer"])
            entropies.append(interp_summary[aid]["mean_entropy"])
    corr = _pearsonr(eers, entropies)
    if corr is not None:
        direction, strength = _describe_correlation(corr)
        notes.append(
            f"% Entropy-EER correlation: r={corr:.3f} ({strength} {direction}).\n"
            f"% Draft sentence: \"Per-attack EER and per-attack mean Grad-CAM "
            f"entropy show a {strength} {direction} correlation "
            f"(r={corr:.3f}) across the {len(eers)} eval attack types.\"\n"
            f"% TODO: with n={len(eers)}, this is a small-sample correlation -- "
            f"say so, and avoid claiming causation.\n\n"
        )

    # 4. Per-attack SSIM consistency vs. the fragility SSIM floor (same units,
    #    so this comparison is actually valid -- see gradcam.py docstring).
    fragility = interp_summary.get("_fragility_noise_floor", {})
    ssim_floor = fragility.get("mean_ssim")
    if ssim_floor is not None:
        below_floor, at_or_above_floor = [], []
        for aid in attack_ids:
            consistency = interp_summary[aid]["consistency_ssim"]
            if consistency is None:
                continue
            (below_floor if consistency < ssim_floor else at_or_above_floor).append(aid)
        notes.append(
            f"% SSIM consistency vs. fragility floor (mean_ssim={ssim_floor:.4f}):\n"
            f"% Attack types BELOW the floor (consistency worse than noise alone "
            f"predicts -- a genuine, attack-specific interpretability signal): "
            f"{below_floor or 'none'}\n"
            f"% Attack types AT/ABOVE the floor (consistency no worse than "
            f"Grad-CAM's baseline instability -- not evidence of an "
            f"attack-specific problem): {at_or_above_floor or 'none'}\n\n"
        )

    out_path = PAPER_DIR / "auto_draft_discussion.tex"
    out_path.write_text("".join(notes))
    print(f"Wrote {out_path} (not wired into main.tex -- read, verify, and hand-edit into Discussion/Results)")


def main():
    train_summary = _load("train_summary.json")
    eval_summary = _load("eval_summary.json")
    interp_summary = _load("interpretability_summary.json")

    header = (
        "% AUTO-GENERATED by scripts/generate_results_tex.py -- do not hand-edit.\n"
        "% Re-run the generator after each pipeline step to refresh these values.\n\n"
    )
    lines = [header]
    table_lines = [header]

    # ---- training summary -------------------------------------------------
    if train_summary:
        lines.append(_macro("NumTrainClips", train_summary.get("n_train_clips", "TODO"), "{:,}"))
        lines.append(_macro("NumDevClips", train_summary.get("n_dev_clips", "TODO"), "{:,}"))
        lines.append(_macro("BatchSize", train_summary.get("batch_size", "TODO"), "{}"))
        lines.append(_macro("LearningRate", train_summary.get("learning_rate", "TODO"), "{:.0e}"))
        # num_epochs_run is absent from a backfilled summary (see
        # scripts/backfill_train_summary.py) -- no per-epoch history exists
        # to count, so this stays TODO in that case rather than crashing.
        lines.append(_macro("NumEpochsRun", train_summary.get("num_epochs_run", "TODO"), "{}"))
        lines.append(_macro("PosWeight", train_summary.get("pos_weight", "TODO"), "{:.4f}"))
        best_dev_eer = train_summary.get("best_dev_eer")
        lines.append(_macro("BestDevEER", 100 * best_dev_eer if best_dev_eer is not None else "TODO",
                             "{:.2f}" if best_dev_eer is not None else "{}"))
    else:
        for name in ["NumTrainClips", "NumDevClips", "BatchSize", "LearningRate",
                     "NumEpochsRun", "PosWeight", "BestDevEER"]:
            lines.append(_macro(name, "TODO"))

    # ---- eval summary -------------------------------------------------------
    if eval_summary:
        lines.append(_macro("NumEvalClips", eval_summary["n_eval_clips"], "{:,}"))
        overall = eval_summary["overall"]
        lines.append(_macro("OverallEER", 100 * overall["eer"], "{:.2f}"))
        lines.append(_macro("OverallAUC", overall["auc"], "{:.4f}"))
        lines.append(_macro("OverallFOneEER", overall["f1_at_eer_threshold"], "{:.4f}"))
    else:
        for name in ["NumEvalClips", "OverallEER", "OverallAUC", "OverallFOneEER"]:
            lines.append(_macro(name, "TODO"))

    # ---- fragility control ---------------------------------------------------
    if interp_summary and "_fragility_noise_floor" in interp_summary:
        floor = interp_summary["_fragility_noise_floor"]
        lines.append(_macro("FragilityFloor", floor["mean_shift"], "{:.4f}"))
        # mean_ssim is the apples-to-apples floor for the per-attack SSIM
        # consistency column below (same units); mean_shift is a secondary,
        # differently-scaled reference kept for anyone who wants the raw
        # pixel-difference magnitude too. Older interpretability_summary.json
        # files (from before this field existed) won't have it -- fall back
        # to TODO rather than crash, so a stale JSON doesn't break the build.
        lines.append(_macro("FragilitySSIMFloor", floor.get("mean_ssim", "TODO"),
                             "{:.4f}" if "mean_ssim" in floor else "{}"))
        lines.append(_macro("FragilityNClips", floor["n_clips"], "{}"))
    else:
        lines.append(_macro("FragilityFloor", "TODO"))
        lines.append(_macro("FragilitySSIMFloor", "TODO"))
        lines.append(_macro("FragilityNClips", "TODO"))

    # ---- per-attack table + correlation --------------------------------------
    if eval_summary and interp_summary:
        per_attack_eer = eval_summary["per_attack"]
        attack_ids = sorted(a for a in interp_summary if not a.startswith("_"))

        rows = []
        eers, entropies = [], []
        for aid in attack_ids:
            interp = interp_summary[aid]
            eer_entry = per_attack_eer.get(aid)
            eer_pct = 100 * eer_entry["eer"] if eer_entry else None
            entropy = interp["mean_entropy"]
            consistency = interp["consistency_ssim"]

            if eer_pct is not None:
                eers.append(eer_pct)
                entropies.append(entropy)

            eer_str = f"{eer_pct:.2f}" if eer_pct is not None else "--"
            cons_str = f"{consistency:.3f}" if consistency is not None else "--"
            rows.append(f"{aid} & {eer_str} & {entropy:.3f} & {cons_str} \\\\")

        corr = _pearsonr(eers, entropies)
        lines.append(_macro("EntropyEERCorrelation", corr if corr is not None else "TODO",
                             "{:.3f}" if corr is not None else "{}"))

        table = [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Per-attack-type EER (eval set, unseen attacks), mean normalized "
            "Grad-CAM entropy, and mean pairwise SSIM consistency.}",
            "\\label{tab:per-attack}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Attack & EER (\\%) & Mean entropy & Consistency (SSIM) \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}\n",
        ]
        table_lines.append("\n".join(table))
    else:
        lines.append(_macro("EntropyEERCorrelation", "TODO"))
        table_lines.append(
            "% Per-attack table omitted -- run evaluate.py and "
            "interpretability/run_interpretability.py first, then re-run this script.\n"
        )

    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    macros_path = PAPER_DIR / "macros.tex"
    macros_path.write_text("".join(lines))
    print(f"Wrote {macros_path}")

    table_path = PAPER_DIR / "results_table.tex"
    table_path.write_text("".join(table_lines))
    print(f"Wrote {table_path}")

    # Copy the training curve into paper/figures/ if it exists, so main.tex's
    # \includegraphics call resolves without the author moving files by hand.
    src_curve = FIG_DIR / "training_curve.png"
    if src_curve.exists():
        dst = PAPER_DIR / "figures" / "training_curve.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src_curve.read_bytes())
        print(f"Copied {src_curve} -> {dst}")

    _write_autodraft_discussion(train_summary, eval_summary, interp_summary)


if __name__ == "__main__":
    main()
