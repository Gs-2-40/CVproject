from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch
from itertools import combinations
from torch.utils.data import Subset

def choose_val_videos(counts, val_ratio=0.2):
    n = len(counts)

    if n < 2:
        raise ValueError("Для разделения нужны минимум два видео")

    target = sum(counts) * val_ratio
    best_error = float("inf")
    best = None

    for k in range(1, n):
        for combination in combinations(range(n), k):
            current = sum(counts[i] for i in combination)
            error = abs(current - target)

            if error < best_error:
                best_error = error
                best = combination

    return set(best)
def split_dataset(dataset, val_ratio=0.2):
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio должен быть между 0 и 1")

    counts = dataset.window_counts

    if len(counts) > 18:
        raise ValueError("Слишком много видео для полного перебора")

    val_videos = choose_val_videos(counts, val_ratio)

    train_indices = []
    val_indices = []

    for video_idx in range(len(counts)):
        start = int(dataset.offsets[video_idx])
        end = int(dataset.offsets[video_idx + 1])

        if video_idx in val_videos:
            val_indices.extend(range(start, end))
        else:
            train_indices.extend(range(start, end))

    if not train_indices or not val_indices:
        raise ValueError("Обучающая или валидационная выборка пуста")

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices)
    )
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
        self.window_counts = []
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

            self.window_sizes.append(window_size)
            self.window_counts.append(count)
        self.offsets = np.cumsum([0] + self.window_counts)
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

    train_dataset, val_dataset = split_dataset(
        dataset,
        val_ratio=0.2
    )

    print("Всего окон:", len(dataset))
    print("Обучение:", len(train_dataset))
    print("Валидация:", len(val_dataset))