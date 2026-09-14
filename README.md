EEG Cognitive Workload Classification with EEGNet
Classify mental workload (Rest vs. Mental Arithmetic) from EEG signals using a simplified EEGNet architecture.
Author
Elham Latif
This project processes the PhysioNet EEG During Mental Arithmetic Tasks dataset and trains a convolutional neural network to distinguish between low and high cognitive workload states.
Project Structure
eeg-cognitive-workload/
├── src/
│   ├── download_data.py
│   ├── preprocess.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── data/
│   ├── raw/
│   └── processed/
├── results/
├── requirements.txt
├── .gitignore
└── README.md
•	download_data.py — Downloads the PhysioNet EEG dataset.
•	preprocess.py — Filters, re-references, and segments the EEG recordings.
•	dataset.py — PyTorch Dataset implementation.
•	model.py — EEGNet model implementation.
•	train.py — Trains the model using a subject-independent split.
•	evaluate.py — Evaluates the trained model and generates a confusion matrix.
The data/raw/, data/processed/, and results/ directories are not tracked by Git.
Requirements
•	Python 3.9+
•	CUDA (optional, for GPU training)
Install the required packages:
pip install -r requirements.txt
How to Run
1. Download the dataset
python src/download_data.py
This downloads the EEG recordings from PhysioNet and stores the raw EDF files in:
data/raw/
2. Preprocess the data
python src/preprocess.py
The preprocessing pipeline performs:
•	Band-pass filtering (1–40 Hz)
•	Notch filtering (50 Hz)
•	Average re-referencing
•	Segmentation into 4-second windows with 50% overlap
•	Z-score normalization per window
The processed data are saved as:
data/processed/
├── X.npy
├── y.npy
└── groups.npy
Where:
•	X.npy contains the EEG windows.
•	y.npy contains the class labels.
•	groups.npy contains the subject IDs used for subject-wise splitting.
3. Train the model
python src/train.py --epochs 50 --batch-size 32 --learning-rate 0.001
The training script uses GroupShuffleSplit to prevent EEG windows from the same subject appearing in both the training and validation sets.
The best model is saved to:
results/best_model.pt
4. Evaluate the model
python src/evaluate.py
The evaluation script reports:
•	Accuracy
•	F1-score
•	ROC-AUC
A confusion matrix is saved to:
results/confusion_matrix.png

Results
Results will be added after training on the full dataset.

Model
This project uses a simplified implementation of EEGNet based on:
Lawhern, V. J., et al. (2018).
EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.
Journal of Neural Engineering.
The model consists of:
1.	Temporal convolution
2.	Depthwise spatial convolution
3.	Separable convolution
4.	A two-class output layer
The two classes are:
Label	Condition
0	Rest
1	Mental Arithmetic
Dataset
The project uses the PhysioNet EEG During Mental Arithmetic Tasks dataset.
Each subject has two recordings:
File	Condition	Label
SubjectXX_1.edf	Resting state	0
SubjectXX_2.edf	Mental arithmetic	1
The goal is to classify EEG segments according to cognitive workload condition.
Citation
If you use this project or the dataset, please cite:
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

