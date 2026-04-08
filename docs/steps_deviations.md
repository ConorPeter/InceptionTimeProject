Implementation Steps & Deviation Analysis
Steps Taken:
Environment Setup: Created a dedicated Conda environment (inception_env) with Python 3.7 to match the legacy requirements of the original InceptionTime repository.
Data Preparation: Downloaded the UCR Archive datasets (GunPoint, ECG200, FordA) and converted them to .tsv format to ensure compatibility with the data loader.
Code Adaptation: Modified the original Linux-centric codebase to run on Windows:
Updated file paths to use Windows directory structures.
Fixed delimiter issues in readucr utility functions.
Removed hard exits that prevented execution on CPU-only machines.
Execution: Ran the training script for 1500 epochs per dataset. Due to hardware constraints, we executed a single-model training run rather than the full 5-model ensemble specified in the paper.
Validation: Extracted final test accuracies from df_metrics.csv and compared them against the baselines reported in the original paper.


Reason for Deviation (Technical Constraints):
The primary deviation from the original protocol (running on GPU clusters with a 5-model ensemble) was necessitated by CUDA incompatibility with legacy dependencies:
Library Conflict: The original InceptionTime code relies on TensorFlow 1.15 and Keras 2.3. These versions have deprecated support for modern CUDA drivers (CUDA 10.0/10.1+) found on current consumer GPUs.
Runtime Error: Attempts to enable GPU acceleration resulted in CUDA_ERROR_NO_DEVICE and cudart64_100.dll not found errors, as the specific DLLs required by TF 1.15 are absent in modern driver packages.

Resolution & Experimental Scope: To ensure hardware-agnostic reproducibility and bypass legacy GPU dependency issues (CUDA 10.0), all experiments were executed on a CPU infrastructure using the full 1500-epoch protocol. This study validated architectural integrity via single-model instances across three datasets. While the original paper utilizes a 5-model ensemble for marginal accuracy gains, this single-model approach confirms convergence and efficacy. Future work aiming for exact upper-bound metrics should extend this workflow by running five independent iterations (nb_iter_=5) and aggregating predictions.