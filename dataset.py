"""
dataset.py
==========
PyTorch Dataset classes and data-loading utilities for all model families.

Expects processed JSON files produced by preprocess.py.  Each file contains:
  { "<match_key>": { "df": [...], "X": [...], "Y": [...], "GNN": [...] } }
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from pathlib import Path


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_processed_matches(data_dir: str) -> dict:
    """Load all per-match JSON files from a directory into a single dict."""
    data = {}
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(data_dir, fname)) as f:
            match_data = json.load(f)
        data.update(match_data)
    return data


def chronological_split(
    data: dict,
    match_dates: dict,
    train_end: str,
    val_end: str,
) -> tuple:
    """
    Split matches into train/val/test by tournament date string (YYYY-MM-DD).
    match_dates: { match_key: "YYYY-MM-DD" }
    """
    train, val, test = {}, {}, {}
    for key, match in data.items():
        date = match_dates.get(key)
        if date is None:
            train[key] = match
            continue
        if date <= train_end:
            train[key] = match
        elif date <= val_end:
            val[key] = match
        else:
            test[key] = match
    return train, val, test


def flatten_to_states(data: dict, representation: str = "X") -> tuple:
    """Flatten all game states across matches into flat lists."""
    states, labels = [], []
    for match in data.values():
        for state, label in zip(match[representation], match["Y"]):
            states.append(state)
            labels.append(label)
    return states, labels


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class SetNormaliser:
    """Per-feature mean/std normaliser fitted on the training set."""

    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, states: list) -> "SetNormaliser":
        all_vecs = np.concatenate([np.array(s) for s in states], axis=0)
        self.mean = all_vecs.mean(axis=0)
        self.std  = all_vecs.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, state: list) -> np.ndarray:
        arr = np.array(state, dtype=np.float32)
        return (arr - self.mean) / self.std

    def save(self, path: str):
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str) -> "SetNormaliser":
        npz = np.load(path)
        obj = cls()
        obj.mean = npz["mean"]
        obj.std  = npz["std"]
        return obj


# ---------------------------------------------------------------------------
# Vector Dataset (LightGBM)
# ---------------------------------------------------------------------------

class VectorDataset:
    """Produces a DataFrame + label array for sklearn-compatible models."""

    def __init__(self, data: dict):
        import pandas as pd
        rows, labels = flatten_to_states(data, "df")
        self.df     = pd.DataFrame(rows)
        self.labels = np.array(labels, dtype=np.int32)

    def get_features(self) -> tuple:
        drop_cols = [c for c in ["ctWin", "roundNum", "tick"] if c in self.df.columns]
        X = self.df.drop(columns=drop_cols)
        y = self.df["ctWin"].values if "ctWin" in self.df.columns else self.labels
        return X, y


# ---------------------------------------------------------------------------
# Set Dataset (DeepSets / SetTransformer)
# ---------------------------------------------------------------------------

class SetDataset(Dataset):
    """
    Each sample is a (n_players x 32) float tensor and a scalar label.
    Use make_batches() for the grouped-by-alive-count batching ESTA uses.
    """

    def __init__(self, states: list, labels: list, normaliser: SetNormaliser = None):
        self.normaliser = normaliser
        self.samples = []
        for state, label in zip(states, labels):
            if normaliser is not None:
                arr = normaliser.transform(state).astype(np.float32)
            else:
                arr = np.array(state, dtype=np.float32)
            self.samples.append((arr, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        arr, label = self.samples[idx]
        return (
            torch.tensor(arr,   dtype=torch.float32),
            torch.tensor([label], dtype=torch.float32),
        )

    def make_batches(self, batch_size: int, shuffle: bool = True) -> list:
        """
        Group samples by number of alive players, then create fixed-size
        batches within each group.  Mirrors ESTA's train_batch approach.
        Returns list of (X_tensor, Y_tensor) ready for the forward pass.
        """
        by_size = {}
        for x, y in self.samples:
            n = x.shape[0]
            by_size.setdefault(n, []).append((x, y))

        if shuffle:
            for n in by_size:
                random.shuffle(by_size[n])

        batches = []
        for n, group in by_size.items():
            xs = np.array([g[0] for g in group])
            ys = np.array([g[1] for g in group])
            total = len(group)
            for i in range(0, total, batch_size):
                xb = torch.tensor(xs[i:i+batch_size], dtype=torch.float32)
                yb = torch.tensor(ys[i:i+batch_size], dtype=torch.float32).unsqueeze(1)
                batches.append((xb, yb))

        if shuffle:
            random.shuffle(batches)

        return batches


# ---------------------------------------------------------------------------
# Graph Dataset (GCN / GAT)
# ---------------------------------------------------------------------------

class GraphDataset(Dataset):
    """
    Each sample is a PyG Data object.
    Use with torch_geometric.loader.DataLoader for automatic batching.
    """

    def __init__(self, graphs: list):
        self.data_list = []
        for g in graphs:
            x          = torch.tensor(g["x"],         dtype=torch.float)
            edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
            edge_attr  = torch.tensor(g["edge_attr"],  dtype=torch.float)
            y          = torch.tensor([g["y"]],        dtype=torch.long)
            self.data_list.append(
                Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# ---------------------------------------------------------------------------
# Temporal Graph Dataset (Temporal GNN)
# ---------------------------------------------------------------------------

class TemporalGraphDataset(Dataset):
    """
    Each sample is a sequence of PyG Data objects (one per game state)
    representing a full round in chronological order.
    """

    def __init__(self, data: dict, max_len: int = 60):
        self.sequences = []
        self.max_len   = max_len

        for match in data.values():
            # Group states by round number
            rounds = {}
            for gnn, label, df_row in zip(match["GNN"], match["Y"], match["df"]):
                rnd = df_row.get("roundNum", 0)
                rounds.setdefault(rnd, []).append((gnn, label))

            for rnd, states in rounds.items():
                if not states:
                    continue
                label  = states[-1][1]
                graphs = []
                for g, _ in states[:max_len]:
                    x          = torch.tensor(g["x"],         dtype=torch.float)
                    edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
                    edge_attr  = torch.tensor(g["edge_attr"],  dtype=torch.float)
                    graphs.append(
                        Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                    )
                self.sequences.append((graphs, label))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def temporal_collate_fn(batch):
    """
    Custom collate for TemporalGraphDataset.
    Returns (timestep_batches, masks, labels) where:
      timestep_batches[t]: PyG Batch for all sequences at step t (None if none)
      masks[t]:            list of bool, True where sequence has a state at t
      labels:              LongTensor of shape (B,)
    """
    max_len = max(len(seq) for seq, _ in batch)
    labels  = torch.tensor([label for _, label in batch], dtype=torch.long)

    timestep_batches, masks = [], []
    for t in range(max_len):
        graphs_at_t, mask_at_t = [], []
        for seq, _ in batch:
            if t < len(seq):
                graphs_at_t.append(seq[t])
                mask_at_t.append(True)
            else:
                mask_at_t.append(False)
        timestep_batches.append(
            Batch.from_data_list(graphs_at_t) if graphs_at_t else None
        )
        masks.append(mask_at_t)

    return timestep_batches, masks, labels


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_datasets(
    train_data: dict,
    val_data:   dict,
    test_data:  dict,
    temporal:   bool = False,
    max_seq_len: int = 60,
) -> dict:
    """
    Build and return all dataset objects for one map.

    Returns:
      {
        "vector":   { "train": VectorDataset, "val": ..., "test": ... },
        "set":      { "train": SetDataset, "val": ..., "test": ...,
                      "normaliser": SetNormaliser },
        "graph":    { "train": GraphDataset, "val": ..., "test": ... },
        "temporal": { ... }   (only if temporal=True)
      }
    """
    # Vector
    vector = {
        split: VectorDataset(d)
        for split, d in [("train", train_data), ("val", val_data), ("test", test_data)]
    }

    # Set
    train_states, train_labels = flatten_to_states(train_data, "X")
    val_states,   val_labels   = flatten_to_states(val_data,   "X")
    test_states,  test_labels  = flatten_to_states(test_data,  "X")

    normaliser = SetNormaliser().fit(train_states)
    set_datasets = {
        "train":      SetDataset(train_states, train_labels, normaliser),
        "val":        SetDataset(val_states,   val_labels,   normaliser),
        "test":       SetDataset(test_states,  test_labels,  normaliser),
        "normaliser": normaliser,
    }

    # Graph
    train_graphs, _ = flatten_to_states(train_data, "GNN")
    val_graphs,   _ = flatten_to_states(val_data,   "GNN")
    test_graphs,  _ = flatten_to_states(test_data,  "GNN")

    graph_datasets = {
        "train": GraphDataset(train_graphs),
        "val":   GraphDataset(val_graphs),
        "test":  GraphDataset(test_graphs),
    }

    result = {"vector": vector, "set": set_datasets, "graph": graph_datasets}

    if temporal:
        result["temporal"] = {
            "train": TemporalGraphDataset(train_data, max_len=max_seq_len),
            "val":   TemporalGraphDataset(val_data,   max_len=max_seq_len),
            "test":  TemporalGraphDataset(test_data,  max_len=max_seq_len),
        }

    return result
