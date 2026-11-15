from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

class MyDataset(Dataset):
    def __init__(self, frames, window_seconds=3.0):
        processed_dir = Path("data/processed")
        self.window_counts = []
        self.offsets = []
        npzs = sorted(processed_dir.glob("*.npz"))
        for npz in npzs:
            npz_ = np.load(npz)
            self.window_counts.append(len(npz_["keypoints"]) - int(npz_["fps"] * window_seconds))
            self.offsets = np.cumsum([0] + self.window_counts)
    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        video_idx = np.searchsorted(
            self.offsets,
            idx,
            side="right"
        ) - 1

        local_idx = idx - self.offsets[video_idx]
        start_frame = local_idx * self.stride


MyDataset(30)