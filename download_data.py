# download.py
# Downloads EEG Mental Arithmetic dataset from PhysioNet
# Written by Elham Latif

# The dataset has 2 EDF files per subject:
#   SubjectXX_1.edf -> resting
#   SubjectXX_2.edf -> doing math in head
# Both get saved into data/raw/
# Just run this once, then run preprocess.py

from pathlib import Path
import subprocess
import sys

# project root is one level up from this file (scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# PhysioNet link for the eegmat dataset
URL = "https://physionet.org/files/eegmat/1.0.0/"


def download_dataset():
    # make the raw folder if it doesnt exist yet
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Saving files to:", RAW_DIR)

    # wget flags:
    #   -r recursive
    #   -N only newer files
    #   -c continue if interrupted
    #   -np no parent dirs
    #   -P save to this folder
    cmd = [
        "wget",
        "-r",
        "-N",
        "-c",
        "-np",
        "-P",
        str(RAW_DIR),
        URL,
    ]

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        # wget not installed probably
        print("couldnt find wget on this machine")
        print("you can just download it manually from:")
        print(URL)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("download failed:", e)
        print("try downloading manually from:", URL)
        sys.exit(1)

    print("done.")


if __name__ == "__main__":
    download_dataset()