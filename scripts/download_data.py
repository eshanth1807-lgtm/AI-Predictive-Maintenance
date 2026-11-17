"""Download the original CC BY 4.0 UCI dataset without extracting arbitrary paths."""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = 'https://archive.ics.uci.edu/static/public/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip'
DEST = Path(__file__).resolve().parents[1] / 'data' / 'ai4i2020.csv'


def download():
    if DEST.exists():
        print(f'Already present: {DEST}')
        return
    with urllib.request.urlopen(URL, timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        content = archive.read('ai4i2020.csv')
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_bytes(content)
    print(f'Downloaded {DEST}')


if __name__ == '__main__':
    try:
        download()
    except Exception as exc:
        sys.exit(f'Download failed: {exc}. Download AI4I 2020 from UCI and place ai4i2020.csv in data/.')
