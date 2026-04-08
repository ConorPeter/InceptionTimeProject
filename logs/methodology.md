### Experimental Setup & Deviations from Original Protocol

To reproduce the InceptionTime architecture on a local Windows workstation, several adaptations were necessary to address operating system differences, library versioning, and computational constraints.

#### 1. Hardware & Computational Constraints
*   **Original Protocol:** The authors utilized a GPU cluster to train an ensemble of 5 independent models per dataset.
*   **Our Implementation:** Experiments were conducted on a local CPU-only environment due to hardware limitations. Consequently, we trained a **single high-performance model** per dataset rather than a 5-model ensemble. Despite this, our results on ECG200 (+4.0%) and GunPoint (+0.3%) met or exceeded the paper's ensemble baselines, validating the architectural implementation.

#### 2. Software Environment
The reproduction was performed using a constrained Python environment to match the legacy requirements of the original codebase (TensorFlow 1.x):
*   **OS:** Windows 10/11 (Original code developed for Linux)
*   **Python Version:** 3.7 (via Conda `inception_env`)
*   **Deep Learning Framework:** TensorFlow 1.15.0 with Keras 2.3.1
*   **Key Libraries:** `numpy==1.16.4`, `scikit-learn==0.21.3`, `pandas==0.24.2`
*   **Justification:** Newer versions of TensorFlow (2.x+) contain breaking API changes incompatible with the original repository structure.

#### 3. Code Modifications for Windows Compatibility
Several critical modifications were applied to the original Linux-centric codebase to ensure execution on Windows:
*   **Path Handling:** Hardcoded Unix-style paths (`/home/...`) were replaced with dynamic Windows-compatible paths using raw strings (e.g., `C:/Users/...`).
*   **File Delimiters:** The data loader was explicitly configured to read `.tsv` (Tab-Separated Values) files with `delimiter='\t'`, as the default behavior differed between OS file systems.
*   **GPU Fallback:** The original script contained a hard exit (`exit()`) if no GPU was detected. This was modified to issue a warning and proceed with CPU training to accommodate our local setup.
*   **Output Buffering:** Console output buffering issues common in Windows CMD were managed by ensuring explicit flushes or relying on file-based logging (`history.csv`) for progress tracking.

#### 4. Data & Output Handling
*   **Data Format:** We utilized the full-resolution UCR Archive `.tsv` files rather than compressed or short-version variants to ensure exact reproducibility of the input signal.
*   **Artifacts:** The training process generated standard Keras artifacts including `best_model.hdf5` (model weights), `history.csv` (epoch-by-epoch loss/accuracy), and `df_metrics.csv` (final test performance). These files serve as the primary evidence of successful convergence.

## GPU/CUDA Status
- **Physical GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (Detected)
- **Compute Backend:** CPU-only (GPU libraries not loaded)
- **Reason:** TensorFlow 1.15 requires CUDA 10.0/cuDNN 7.4; system has modern CUDA 12.x drivers which are incompatible.
- **Decision:** Proceed with CPU for Phase 1 (GunPoint baseline). 
- **Impact:** GunPoint training 2hrs on CPU (acceptable). FordA required several hours on CPU.

## Deviation Log (Updated)
| Original Requirement | Actual Implementation | Justification |
|---------------------|----------------------|---------------|
| GPU Acceleration (CUDA 10.0) | CPU-only execution | CUDA version mismatch; GPU setup deferred to maintain environment stability |
| TensorFlow-GPU 1.12.0 | TensorFlow-GPU 1.15.0 (CPU mode) | TF 1.15 maintains architectural fidelity while enabling Windows execution |

## Issue: Epoch Output Delayed/Buffered
- **Symptom:** `iter 0`, `iter 1` headers appeared immediately, but `Epoch 1/1500` logs did not show.
- **Cause:** Windows Command Prompt output buffering with Keras/TensorFlow progress bars.
- **Resolution:** Script is running (confirmed via CPU usage). Epoch logs will appear once buffer flushes.
- **Reference:** InceptionTime Paper Section 3.2 (5-model ensemble trained sequentially).

### Reproduction Results Summary

The following table compares the achieved test accuracy against the baselines reported in the original InceptionTime paper [1].

| Dataset | Samples | Series Length | Our Accuracy | Paper Baseline | Difference | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GunPoint** | 50 | 150 | **0.9933** | 0.990 | +0.0033 | Success |
| **ECG200** | 100 | 96 | **0.9200** | 0.880 | +0.0400 | Exceeds Baseline |
| **FordA** | 3,200 | 500 | **0.9561** | 0.960 | -0.0039 |  Within Tolerance |

*Note: Results represent a single model trained for 1500 epochs. The original paper reports an ensemble of 5 models; our single-model results are highly competitive, particularly on ECG200.*