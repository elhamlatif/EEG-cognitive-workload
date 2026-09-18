# preprocess.py
# turns the raw EDF files into windows for the model
# filtering + re-referencing + 4-sec windows with 50% overlap
# elham latif

# labels are in the filename:
#   SubjectXX_1.edf -> 0 (rest / low workload)
#   SubjectXX_2.edf -> 1 (math / high workload)
#
# output: data/processed/X.npy, y.npy, groups.npy

# v1   - basic filtering + windows
# v1.1 - added notch filter (was picking up mains hum)
# v1.2 - z-score per window instead of per recording
# v1.3 - switched to average reference

# TODO: try 2-sec windows at some point, 4 might be too long
# FIXME: NOTCH_FREQ is hardcoded to 50, should really be a flag

from pathlib import Path
import glob

import mne
import numpy as np

# mne is way too chatty by default, shut it up
mne.set_log_level("WARNING")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# filter settings
# 1-40 Hz is the usual band for this kind of task
LOW_FREQ = 1.0
HIGH_FREQ = 40.0
NOTCH_FREQ = 50.0  # 50 Hz for EU, 60 for US. we're in EU so 50.

# window settings
WINDOW_SEC = 4.0
OVERLAP = 0.50  # 50% overlap, seemed like a reasonable default


def find_edf_files():
    # get every edf under raw/, sorted so runs are reproducible
    # (glob order is not guaranteed which messed up my early runs)
    pattern = str(RAW_DIR / "**" / "*.edf")
    return sorted(glob.glob(pattern, recursive=True))


def get_label(filename):
    # _1 is rest, _2 is math
    # anything else means the file isnt ours, ignore it
    name = Path(filename).name

    if name.endswith("_1.edf"):
        return 0
    if name.endswith("_2.edf"):
        return 1

    return None  # shouldnt happen but just in case


def get_subject_id(filename):
    # SubjectXX_1.edf -> SubjectXX
    return Path(filename).stem.split("_")[0]


def make_windows(data, sr):
    # cut the recording into overlapping windows
    # data comes in as (n_channels, n_samples)
    window_samples = int(WINDOW_SEC * sr)
    step_samples = int(window_samples * (1 - OVERLAP))

    windows = []
    start = 0

    while start + window_samples <= data.shape[1]:
        windows.append(data[:, start:start + window_samples])
        start += step_samples

    if not windows:
        # file was shorter than one window, return empty
        # (rare but happens with a couple of subjects)
        return np.empty((0, data.shape[0], window_samples))

    return np.stack(windows)


def preprocess_recording(f):
    # load one edf and clean it up
    # f is the full path
    raw = mne.io.read_raw_edf(f, preload=True)

    # bandpass first
    raw.filter(LOW_FREQ, HIGH_FREQ, fir_design="firwin")

    # then kill the mains hum
    # tried None, 45.0 first but 40-50 worked better
    raw.notch_filter(freqs=NOTCH_FREQ)

    # average reference - works fine for this dataset
    # could try CAR or laplacian later if results are bad
    raw.set_eeg_reference("average", projection=False)

    data = raw.get_data()
    sr = raw.info["sfreq"]  # sampling rate

    windows = make_windows(data, sr)

    # z-score each channel inside each window
    # doing it per window (not per recording) keeps things comparable
    # the 1e-8 is just to avoid div by zero on flat channels
    mean = windows.mean(axis=2, keepdims=True)
    std = windows.std(axis=2, keepdims=True)
    windows = (windows - mean) / (std + 1e-8)

    return windows


def main():
    files = find_edf_files()

    if not files:
        print("no edf files in:", RAW_DIR)
        print("run download.py first")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # collect everything here, stack at the end
    all_windows = []
    all_labels = []
    all_subjects = []

    for f in files:
        label = get_label(f)

        if label is None:
            # not a file we recognize, skip it
            continue

        subject = get_subject_id(f)

        try:
            windows = preprocess_recording(f)
        except Exception as e:
            # some files are just broken, dont kill the whole run
            print("skipping", Path(f).name, "-", e)
            continue

        all_windows.append(windows)
        all_labels.extend([label] * len(windows))
        all_subjects.extend([subject] * len(windows))

        cond = "rest" if label == 0 else "math"
        print(
            Path(f).name,
            "->",
            len(windows),
            "windows | subject=" + subject,
            "| condition=" + cond,
        )

    if not all_windows:
        print("no valid edf files found")
        return

    # stack everything into big arrays
    # X is float32 to save memory, y is int64 for the loss fn
    X = np.concatenate(all_windows, axis=0).astype(np.float32)
    y = np.asarray(all_labels, dtype=np.int64)
    groups = np.asarray(all_subjects)

    # save
    np.save(PROCESSED_DIR / "X.npy", X)
    np.save(PROCESSED_DIR / "y.npy", y)
    np.save(PROCESSED_DIR / "groups.npy", groups)

    print()
    print("done. saved", len(X), "windows")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("subjects:", len(set(all_subjects)))


if __name__ == "__main__":
    main()