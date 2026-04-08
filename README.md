# InceptionTime Reproduction

Reproduction of **InceptionTime: Finding AlexNet for Time Series Classification** (Fawaz et al., 2020) as part of a university reproducibility project.

Original paper repo: https://github.com/hfawaz/InceptionTime

---

## What this is

InceptionTime is a deep learning model for time series classification that adapts Google's Inception architecture to 1D temporal data. This repo reproduces the core architecture and trains it on three datasets from the UCR Archive: GunPoint, ECG200, and FordA.

---

## First time setup

Open **Anaconda Prompt** and run:

```bash
conda create -n inception_env python=3.7
conda activate inception_env
```

Navigate to the project folder (change the path to wherever you cloned it):

```bash
cd C:\path\to\InceptionTimeProject
pip install numpy==1.18.5
pip install -r requirements.txt
```

> Install numpy first - other packages need it to build correctly on Windows.

Then move into the inception folder and run training:

```bash
cd inception
python main.py InceptionTime
```

Training runs on CPU and takes a while - roughly 2 hours for GunPoint, longer for FordA. Results are saved to `results/<dataset>/` as they finish.

> The original code was written for TensorFlow 1.x - TensorFlow 2.x will not work without modifications.

---

## Re-running from scratch

The script skips any dataset that already has a results folder. To retrain from scratch, delete the results folder first:

```bash
cd C:\path\to\InceptionTimeProject
rmdir /s /q results
cd inception
python main.py InceptionTime
```

---

## Results

Results are saved to `results/<dataset>/` after training:
- `df_metrics.csv` - final test accuracy
- `history.csv` - loss and accuracy per epoch
- `best_model.hdf5` - saved model weights (not tracked by git)

Previous saved runs are kept in `results_baseline/` for comparison.

---

## Hyperparameter experiments

To run the hyperparameter sweep (batch size, filters, depth, kernel size etc.):

```bash
cd inception
python main.py InceptionTime_xp
```

---

## Deviations from the paper

A few things we had to change due to hardware constraints:

| Original | Our setup | Why |
|----------|-----------|-----|
| 5-model ensemble | Single model | CPU only - training time |
| TensorFlow 1.12 + GPU | TensorFlow 1.15, CPU | CUDA version mismatch on Windows |
| 85 UCR datasets | 3 datasets | Computational constraints |
| Linux | Windows 10 | Local hardware |

See `logs/methodology.md` for more detail.

---

## Project structure

```
InceptionTimeProject/
├── inception/
│   ├── main.py
│   ├── classifiers/
│   │   ├── inception.py
│   │   └── nne.py
│   └── utils/
├── data/
│   ├── GunPoint/
│   ├── ECG200/
│   └── FordA/
├── results/                  - generated after training
├── results (Kasturi)/        - saved results from initial reproduction run
├── logs/
│   └── methodology.md
└── requirements.txt
```

---

## Reference

Fawaz et al. (2020). InceptionTime: Finding AlexNet for time series classification. *Data Mining and Knowledge Discovery*, 34(6), 1936-1962.
