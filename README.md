EEG Cognitive Workload Classification with EEGNet
A small EEG classification project that distinguishes resting state from mental arithmetic using a simplified EEGNet architecture.

Author
Elham Latif

Overview
I used the PhysioNet EEG During Mental Arithmetic Tasks dataset to classify EEG segments into two conditions:

Low workload: Rest

High workload: Mental Arithmetic

The pipeline covers data downloading, EEG preprocessing, subject-wise train/validation/test splitting, EEGNet training, and final evaluation on a held-out test set.

Project Structure
text
eeg-cognitive-workload/

├── src/
│   ├── download_data.py
│   ├── preprocess.py
│   ├── dataset.py
│   ├── model.py
│   ├── splits.py
│   ├── train.py
│   └── evaluate.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── results/
│   ├── final_test_metrics.json
│   ├── chosen_run.json
│   └── meta_seed42.json
│
├── requirements.txt
├── .gitignore
└── README.md

Source Files

download_data.py — Downloads the EEG recordings from PhysioNet.

preprocess.py — Filters, re-references, normalizes, and segments the EEG recordings.

dataset.py — Provides the PyTorch dataset used for EEG windows.

model.py — Defines the simplified EEGNet architecture.

splits.py — Creates subject-wise train, validation, and test splits.

train.py — Trains EEGNet using the training set and selects the model using validation performance.

evaluate.py — Evaluates the selected model once on the held-out test set.

The raw and processed EEG data are not tracked by Git.

Requirements
Python 3.9+

PyTorch

NumPy

scikit-learn

MNE

Matplotlib

tqdm

CUDA is optional — training also works fine on CPU.

Install the required packages with:

bash
pip install -r requirements.txt
How to Run
1. Download the Dataset
bash
python src/download_data.py
This downloads the EEG recordings from PhysioNet and stores the EDF files in:

text
data/raw/
2. Preprocess the EEG Data
bash
python src/preprocess.py
The preprocessing pipeline includes:

Band-pass filtering: 1–40 Hz

Notch filtering: 50 Hz

Average re-referencing

4-second windows

50% overlap between consecutive windows

Z-score normalization per window

The processed arrays are saved in:

text
data/processed/
├── X.npy
├── y.npy
└── groups.npy
Where:

X.npy contains the EEG windows.

y.npy contains the class labels.

groups.npy contains the subject IDs.

The final processed dataset used in this project contained:

text
X shape: (4266, 21, 2000)
That's 4,266 EEG windows from 36 subjects.

3. Train the Model
bash
python src/train.py --epochs 50 --batch-size 32 --learning-rate 0.001
The split is performed at the subject level using GroupShuffleSplit, giving:

Training set: 2,517 windows

Validation set: 840 windows

Held-out test set: 909 windows

The test set stays untouched during training and model selection.

Training uses inverse-frequency class weights to reduce the effect of class imbalance, and early stopping is based on validation macro F1.

For the reported run, the best validation macro F1 was:

text
0.7007
at epoch 26. The selected checkpoint was:

text
results/best_model_seed42.pt
4. Evaluate the Model
bash
python src/evaluate.py
This loads the selected checkpoint and the threshold determined from the validation set, then evaluates the model once on the held-out test set. The final metrics are saved to:

text
results/final_test_metrics.json
Results
Final evaluation on the held-out test set:

Metric	Value
Accuracy	0.6183
Precision (High)	0.3770
Recall (High)	0.6833
F1 (High)	0.4859
Macro F1	0.5912
ROC-AUC	0.6628
Test windows	909
The classification threshold used for the final test evaluation was:

text
0.5287
This threshold was picked on the validation set and never re-tuned on the test set.

Confusion Matrix
text
                  Predicted
                  Low   High

Actual Low       398    271
Actual High       76    164
The test set was never used for training, early stopping, or threshold selection.

Model
I used a simplified implementation of EEGNet based on:

Lawhern, V. J., et al. (2018).

EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.

Journal of Neural Engineering.

It has four parts:

Temporal convolution

Depthwise spatial convolution

Separable convolution

Two-class output layer

Labels
Label	Condition
0	Rest
1	Mental Arithmetic
Dataset
This project relies on the PhysioNet EEG During Mental Arithmetic Tasks dataset. Each subject has two recordings:

File	                 Condition	           Label
SubjectXX_1.edf	        Resting state	             0
SubjectXX_2.edf	      Mental arithmetic	             1

The task is to classify each segment by cognitive workload condition.

Citation
EEGNet
Lawhern, V. J., et al. (2018).

EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.

Journal of Neural Engineering.

Dataset
Zyma, I., et al. (2019).

Electroencephalograms during Mental Arithmetic Task Performance.

PhysioNet.

License
This project is released under the MIT License.