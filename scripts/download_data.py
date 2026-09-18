
# Downloads the EEG Mental Arithmetic dataset from PhysioNet
# Written by Elham Latif

from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

URL = "https://physionet.org/files/eegmat/1.0.0/"


def download_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Saving files to:", RAW_DIR)
    print("Reading file list from PhysioNet...")

    request = Request(
        URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    try:
        with urlopen(request, timeout=30) as response:
            html = response.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print("Could not access PhysioNet:", e)
        return

    # Find EDF files listed in the PhysioNet directory.
    files = sorted(set(re.findall(r'href=["\']([^"\']+\.edf)["\']', html, re.I)))

    if not files:
        print("No EDF files were found on the PhysioNet page.")
        print("You can download them manually from:")
        print(URL)
        return

    print(f"Found {len(files)} EDF files.")

    for filename in files:
        filename = filename.split("/")[-1]
        output_file = RAW_DIR / filename

        if output_file.exists():
            print("Already exists:", filename)
            continue

        file_url = urljoin(URL, filename)
        print("Downloading:", filename)

        try:
            request = Request(
                file_url,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urlopen(request, timeout=120) as response:
                with open(output_file, "wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)

            print("Saved:", output_file)

        except Exception as e:
            print("Download failed:", filename)
            print("Error:", e)

            if output_file.exists():
                output_file.unlink()

    print("Download step finished.")


if __name__ == "__main__":
    download_dataset()
