aud/
│
├── .venv/                           # Python virtual environment
│
├── datasets/                        # Dataset only
│   └── archive/
│       └── LA/
│           └── LA/
│               ├── ASVspoof2019_LA_train/
│               ├── ASVspoof2019_LA_dev/
│               ├── ASVspoof2019_LA_eval/
│               ├── ASVspoof2019_LA_cm_protocols/
│               ├── ASVspoof2019_LA_asv_protocols/
│               ├── ASVspoof2019_LA_asv_scores/
│               └── README.LA.txt
│
├── configs/
│   └── config.py                    # Global configuration
│
├── datasets_src/                    # Dataset loading & preprocessing code
│   ├── dataset.py
│   ├── preprocessing.py
│   └── transforms.py
│
├── models/
│   ├── cnn.py
│   ├── lstm.py
│   ├── attention.py
│   ├── classifier.py
│   └── model.py
│
├── training/
│   ├── train.py
│   ├── validate.py
│   ├── loss.py
│   └── optimizer.py
│
├── evaluation/
│   ├── metrics.py
│   ├── eer.py
│   ├── inference.py
│   └── plots.py
│
├── interpretability/
│   ├── gradcam.py
│   ├── entropy.py
│   └── visualize.py
│
├── utils/
│   ├── logger.py
│   ├── seed.py
│   ├── checkpoint.py
│   └── helpers.py
│
├── notebooks/                       # Experimental notebooks
│
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── figures/
│   └── predictions/
│
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_preprocessing.py
│
├── requirements.txt
├── README.md
├── .gitignore
├── train.py                         # Main training entry point
├── evaluate.py                      # Evaluation script
└── infer.py                         # Run inference on new audio
