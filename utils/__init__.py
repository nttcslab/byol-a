# utilities

from itertools import chain
from pathlib import Path
import pandas as pd


def flatten_list(lists):
    return list(chain.from_iterable(lists))


def append_to_csv(csv_filename, data):
    filename = Path(csv_filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(filename) if filename.exists() else pd.DataFrame()
    df.append(data, ignore_index=True).to_csv(filename, index=False)
