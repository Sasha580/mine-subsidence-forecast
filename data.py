import pandas as pd
import torch

from torch.utils.data import Dataset


class ProfilesDataset(Dataset):
    """
    sett_xt_z: torch.FloatTensor (timesteps, mine_len) standardized
    n_hist: number of past profiles used as input
    t_start: the index of the first profile to use as input
    t_end: index of the last profile to use as input (excluded)
    """

    def __init__(self, sett_xt_z, n_hist: int, t_start: int, t_end: int):
        assert 0 <= t_start < t_end <= sett_xt_z.shape[0], "Bad t_start/t_end"
        assert n_hist > 0, "n_hist must be > 0"
        assert (t_start + n_hist) < t_end, "Not enough timesteps for the given n_hist and split"

        self.sett_xt_z = sett_xt_z
        self.n_hist = n_hist
        self.t_start = t_start
        self.t_end = t_end
        self.indices = list(range(t_start + n_hist, t_end))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        sample = self.sett_xt_z[t - self.n_hist:t, :]          # [n_hist, mine_len]
        target = self.sett_xt_z[t, :]                          # [mine_len]
        return sample, target


class Standardizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean


def split_time_series(data, n_hist, val_steps=1, test_steps=1, start_index=0):
    total_steps = data.shape[0]

    train_end = total_steps - val_steps - test_steps
    val_end = total_steps - test_steps

    train_ds = ProfilesDataset(data, n_hist, start_index, train_end)
    val_ds = ProfilesDataset(data, n_hist, train_end - n_hist, val_end)
    test_ds = ProfilesDataset(data, n_hist, val_end - n_hist, total_steps)

    return train_ds, val_ds, test_ds


def load_data(path, n_hist, val_steps=1, test_steps=1, start_index=0):
    df = pd.read_csv(path, delimiter=";", decimal=",")

    # First column is x-axis (keep for plotting)
    x = df.iloc[:, 0].to_numpy()

    # Remaining columns: profiles over time [time_steps, mine_len]
    y_raw = torch.tensor(df.iloc[:, 1:].to_numpy().T, dtype=torch.float32)

    total_steps = y_raw.shape[0]
    train_end = total_steps - val_steps - test_steps
    y_train_raw = y_raw[:train_end]

    # Compute per-x statistics on the training set only
    mean_x = y_train_raw.mean(dim=0)
    std_x = y_train_raw.std(dim=0)

    # Standardize the entire series using train stats
    scaler = Standardizer(mean_x, std_x)
    y_std = scaler.encode(y_raw)

    # Build datasets
    train_ds, val_ds, test_ds = split_time_series(
        y_std,
        n_hist=n_hist,
        val_steps=val_steps,
        test_steps=test_steps,
        start_index=start_index,
    )

    return train_ds, val_ds, test_ds, scaler

