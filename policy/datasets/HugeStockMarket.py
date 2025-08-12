import os
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class HugeStockMarketDataset(Dataset):
    """Dataset of OHLC sequences with next-day move labels.

    Each sample contains a sequence of ``ts_len`` days with four features
    (``Open``, ``High``, ``Low``, ``Close``).  The target is a binary label
    indicating whether the closing price on the following day is above the
    opening price.  Additionally, the raw price difference (close - open) is
    returned for optional regression training.
    """

    def __init__(self, dataset_path: str, ts_len: int) -> None:
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            import kagglehub
            # Download latest version
            self.dataset_path = kagglehub.dataset_download("borismarjanovic/price-volume-data-for-all-us-stocks-etfs")
            print("Path to dataset files:", self.dataset_path)

        root = self.dataset_path / "Data" / "Stocks"
        files = os.listdir(root)
    
        self.objects = []
        for f in tqdm(files[:100]):
            try:
                dt = pd.read_csv(os.path.join(root, f))["Open High Low Close".split()]

                for i in range(len(dt) - ts_len - 1):
                    end = dt.iloc[i + ts_len]
                    delta = end["Close"] - end["Open"]
                    self.objects.append(
                        (
                            dt.iloc[i : i + ts_len].to_numpy(),
                            int(delta > 0),
                            float(delta),
                        )
                    )
            except Exception:
                print(f"Problem in {f}")

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.objects)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        dt = self.objects[idx]
        return {
            "data": torch.tensor(dt[0], dtype=torch.float32),
            "class": torch.tensor(dt[1], dtype=torch.long),
            "regr": torch.tensor(dt[2], dtype=torch.float32),
        }

