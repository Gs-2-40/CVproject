from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch

def resample_keypoints(keypoints, target_frames=30):
    n_frames, n_joints, n_features = keypoints.shape

    old_time = np.linspace(0, 1, n_frames)
    new_time = np.linspace(0, 1, target_frames)

    result = np.empty(
        (target_frames, n_joints, n_features),
        dtype=np.float32
    )

    for joint in range(n_joints):
        for coordinate in range(2):
            values = keypoints[:, joint, coordinate]
            valid = np.isfinite(values)

            if valid.sum() < 2:
                result[:, joint, coordinate] = np.nan
                continue

            result[:, joint, coordinate] = np.interp(
                new_time,
                old_time[valid],
                values[valid],
                left=np.nan,
                right=np.nan
            )

        # Confidence не является координатой.
        # Не интерполируем его как положение сустава.
        nearest = np.rint(
            np.linspace(0, n_frames - 1, target_frames)
        ).astype(int)

        result[:, joint, 2] = keypoints[nearest, joint, 2]

    return result

class MyDataset(Dataset):
    def __init__(self, stride=5, window_frames=30, window_seconds=3):
        self.stride = stride
        self.window_frames = window_frames
        processed_dir = Path("data/processed")
        self.window_sizes = []
        self.offsets = []
        self.files = sorted(processed_dir.glob("*.npz"))
        for npz in self.files:
            npz_ = np.load(npz)
            n_frames = len(npz_["keypoints"])
            window_size = round(float(npz_["fps"]) * window_seconds)

            count = max(
                0,
                (n_frames - window_size) // self.stride + 1
            )

            self.window_sizes.append(count)
        self.offsets = np.cumsum([0] + self.window_sizes)
    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, idx):
        video_idx = np.searchsorted(
            self.offsets,
            idx,
            side="right"
        ) - 1

        local_idx = idx - self.offsets[video_idx]
        start_frame = local_idx * self.stride
        with np.load(self.files[video_idx], allow_pickle=False) as data:
            end_frame = start_frame + self.window_sizes[video_idx]

            X = data["keypoints"][start_frame:end_frame]
            X = resample_keypoints(X, target_frames=self.window_frames)
            X = np.nan_to_num(X, nan=0.0)
            y = data["labels"][end_frame - 1, 0]
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.tensor(
            {"no_fall": 0, "fall": 1}[y],
            dtype=torch.long
        )

        return X, y

if __name__ == "__main__":
    dataset = MyDataset(
        stride=5,
        window_frames=30,
        window_seconds=3
    )

    print("Всего окон:", len(dataset))
    print("Окон по видео:", dataset.window_sizes)
    print("Смещения:", dataset.offsets)

    X, y = dataset[0]

    print("X:", X.shape, X.dtype)
    print("y:", y)
    print("NaN:", torch.isnan(X).sum().item())