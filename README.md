# InceptionTime Reproduction & Improvement

Reproduction and improvement of **InceptionTime: Finding AlexNet for Time Series Classification** (Fawaz et al., 2020) as part of a university reproducibility project.

Original paper repo: https://github.com/hfawaz/InceptionTime

---

## What this is

InceptionTime applies Google's Inception architecture to 1D time series classification. This repo reproduces the core model and trains it on three UCR Archive datasets. We also test several improvements aimed at reducing overfitting on small datasets.

---

## Setup

Open Anaconda Prompt and run:

```bash
conda create -n inception_env python=3.7
conda activate inception_env
```

Navigate to the project folder:

```bash
cd C:\path\to\InceptionTimeProject
pip install numpy==1.18.5
pip install -r requirements.txt
```

Install numpy before the other packages or the build will fail on Windows.

The original codebase targets TensorFlow 1.x. TensorFlow 2.x will not work.

---

## Reproducing the baseline

Switch to the main branch and run:

```bash
git checkout main
cd inception
python main.py InceptionTime
```

Results are saved to a new folder containing:

- `df_metrics.csv` - final test accuracy, precision, recall, training duration
- `history.csv` - loss and accuracy per epoch
- `best_model.hdf5` - best checkpoint by val loss (not tracked by git)

---

## Running the improvement experiments

Each experiment is on its own branch. Switch to the relevant branch and run the same command:

| Branch                       | Experiment                           |
| ---------------------------- | ------------------------------------ |
| `improvement/augmentation`   | Jitter data augmentation             |
| `improvement/dropout`        | Dropout regularisation               |
| `improvement/early-stopping` | Early stopping                       |
| `improvement/kernel-forda`   | Larger kernel size for FordA         |
| `improvement/combined`       | Augmentation + Dropout (final model) |

```bash
git checkout improvement/combined
cd inception
python main.py InceptionTime
```

Each branch writes to its own results folder so nothing gets overwritten.

---

## Results

### Baseline vs paper

| Dataset  | Our Baseline | Epochs | Paper (ensemble of 5) |
| -------- | ------------ | ------ | --------------------- |
| GunPoint | 0.9933       | 1500   | 0.990                 |
| ECG200   | 0.9200       | 1500   | 0.880                 |
| FordA    | 0.9561       | 50     | 0.960                 |

### Final combined model vs paper

| Dataset  | Combined Model (1500 epochs) | Paper (ensemble of 5) | Difference |
| -------- | ---------------------------- | --------------------- | ---------- |
| GunPoint | 0.9933                       | 0.990                 | +0.003     |
| ECG200   | 0.9100                       | 0.880                 | +0.030     |

The combined model matches or exceeds the paper's 5-model ensemble on both small datasets using a single model.

---

## Training times (CPU only, single model)

| Dataset  | Baseline               | Combined (1500 epochs) |
| -------- | ---------------------- | ---------------------- |
| GunPoint | ~16 min (1500 epochs)  | ~32 min                |
| ECG200   | ~20 min (1500 epochs)  | ~41 min                |
| FordA    | ~70 min (50 epochs)    | not run                |

Combined takes roughly twice as long because augmentation doubles the training set size each epoch.

---

## Deviations from the paper

| Original              | Our setup            | Why                              |
| --------------------- | -------------------- | -------------------------------- |
| 5-model ensemble      | Single model         | CPU only, training time          |
| TensorFlow 1.12 + GPU | TensorFlow 1.15, CPU | CUDA version mismatch on Windows |
| 85 UCR datasets       | 3 datasets           | Computational constraints        |
| Linux                 | Windows 10           | Local hardware                   |

See `logs/methodology.md` for full details.

---

## Project structure

```
InceptionTimeProject/
├── inception/
│   ├── main.py                   - training entry point
│   ├── classifiers/
│   │   └── inception.py          - InceptionTime model
│   └── utils/
├── data/
│   ├── GunPoint/
│   ├── ECG200/
│   └── FordA/
├── results_baseline(1500ep)/     - baseline reproduction (1500 epoch run)
├── results_baseline(500ep)/      - baseline at 500 epochs (screening reference)
├── results_augmentation/         - jitter augmentation experiment
├── results_dropout/              - dropout experiment
├── results_early_stopping/       - early stopping experiment
├── results_kernel_forda/         - larger kernel size experiment (FordA only)
├── results_combined/             - final combined model (augmentation + dropout)
├── logs/
│   └── methodology.md            - deviations and setup notes
└── requirements.txt
```

---

## Reference

Fawaz et al. (2020). InceptionTime: Finding AlexNet for time series classification. Data Mining and Knowledge Discovery, 34(6), 1936-1962.
