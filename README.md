# InceptionTime Reproduction & Improvement

Reproduction and improvement of **InceptionTime: Finding AlexNet for Time Series Classification** (Fawaz et al., 2020) as part of a university reproducibility project.

Original paper repo: https://github.com/hfawaz/InceptionTime

---

## What we did and why

InceptionTime applies the Inception architecture from computer vision to time series classification. The original paper trained an ensemble of 5 models on a GPU cluster across 85 UCR datasets and reported state-of-the-art results. We wanted to check whether those results hold up under realistic constraints, and whether targeted improvements could push accuracy further.

Our work had two phases:

**Phase 1 - Reproduction.** We reproduced the InceptionTime architecture on a standard Windows laptop with no GPU. The original code targeted Linux and TensorFlow 1.x, so several compatibility changes were needed. We trained on three UCR datasets (GunPoint, ECG200, FordA) and compared our single-model results against the paper's 5-model ensemble.

**Phase 2 - Improvement.** GunPoint and ECG200 have very small training sets (50 and 100 samples), making them vulnerable to overfitting. We tested four improvements targeting this problem: jitter data augmentation, dropout regularisation, early stopping, and larger kernel sizes for FordA. The two strongest improvements (augmentation and dropout) were combined into a final model.

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

Results are saved to a folder containing:

- `df_metrics.csv` - final test accuracy, precision, recall, training duration
- `history.csv` - loss and accuracy per epoch
- `best_model.hdf5` - best checkpoint by validation loss (not tracked by git)

---

## Running the improvement experiments

Each experiment is on its own branch. Switch to the relevant branch and run the same command:

| Branch | Experiment |
|--------|-----------|
| `improvement/augmentation` | Jitter data augmentation |
| `improvement/dropout` | Dropout regularisation |
| `improvement/early-stopping` | Early stopping |
| `improvement/kernel-forda` | Larger kernel size for FordA |
| `improvement/combined` | Augmentation + Dropout (final model) |

```bash
git checkout improvement/combined
cd inception
python main.py InceptionTime
```

Each branch writes to its own results folder so nothing gets overwritten.

---

## Results

We trained a single model per dataset and compared against the paper's reported ensemble of 5 models. We tested four improvements across the datasets: jitter augmentation, dropout, early stopping, and larger kernel sizes for FordA.

### Paper reported results (ensemble of 5 models, GPU, 1500 epochs)

| Dataset | Accuracy | Training samples |
|---------|----------|-----------------|
| GunPoint | 0.990 | 50 |
| ECG200 | 0.880 | 100 |
| FordA | 0.960 | 3,601 |

### Our screening experiments - GunPoint and ECG200 (500 epochs)

We screened each improvement at 500 epochs before committing to a full run. Results are our test accuracy:

| Experiment | GunPoint | ECG200 |
|-----------|----------|--------|
| Augmentation | 0.980 | 0.920 |
| Dropout | 0.947 | 0.890 |
| Early Stopping | 0.887 | 0.710 |

Augmentation was the strongest individual result. Early stopping was too aggressive and hurt both datasets significantly.

### Our FordA experiment - kernel size (50 epochs)

FordA has much longer sequences (length 500) than the other datasets. We tested whether a larger convolutional kernel would better capture long-range temporal patterns:

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| Baseline | 0.9561 | 0.9559 | 0.9563 |
| Larger kernel | 0.9568 | 0.9569 | 0.9574 |

The larger kernel produced only a negligible improvement (+0.0007 accuracy) while increasing training time from ~2 hours to ~2.7 hours for just 50 epochs.

### Final comparison - our best results vs paper

GunPoint and ECG200 used the combined model (augmentation + dropout, 1500 epochs). FordA used the larger kernel model (50 epochs). FordA falls short of the paper because we could only run 50 epochs on CPU. The kernel experiment alone took ~2.7 hours, making a full 1500-epoch run impractical (estimated 80+ hours on CPU).

| Dataset | Paper (ensemble of 5) | Ours (single model) | Difference |
|---------|-----------------------|---------------------|------------|
| GunPoint | 0.990 | 0.9933 | +0.003 |
| ECG200 | 0.880 | 0.9100 | +0.030 |
| FordA | 0.960 | 0.9568 | -0.004 |

---

## Training times (CPU only, single model)

| Dataset | Run | Epochs | Approximate time |
|---------|-----|--------|-----------------|
| GunPoint | Baseline | 1500 | ~16 min |
| GunPoint | Combined (aug + dropout) | 1500 | ~32 min |
| ECG200 | Baseline | 1500 | ~20 min |
| ECG200 | Combined (aug + dropout) | 1500 | ~41 min |
| FordA | Baseline | 50 | ~2 hours |
| FordA | Larger kernel | 50 | ~2.7 hours |

Combined runs take roughly twice as long because augmentation doubles the training set size each epoch. FordA training times made full improvement experiments impractical on CPU.

---

## Deviations from the paper

| Original | Our setup | Why |
|----------|-----------|-----|
| 5-model ensemble | Single model | CPU only, training time |
| TensorFlow 1.12 + GPU | TensorFlow 1.15, CPU | CUDA version mismatch on Windows |
| 85 UCR datasets | 3 datasets | Computational constraints |
| 1500 epochs on all datasets | 50 epochs on FordA | Training time on CPU |
| Linux | Windows 10 | Local hardware |

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
├── figures/                      - plots used in the report
├── logs/
│   └── methodology.md            - deviations and setup notes
└── requirements.txt
```

---

## Reference

Fawaz et al. (2020). InceptionTime: Finding AlexNet for time series classification. Data Mining and Knowledge Discovery, 34(6), 1936-1962.
