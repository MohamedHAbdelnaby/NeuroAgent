
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, f1_score, confusion_matrix,
)
import nibabel as nib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MANIFEST = PROJECT_ROOT / "preprocessing" / "manifests" / "all_subjects.csv"
GT_DIR   = PROJECT_ROOT / "eval" / "ground_truth"
OUT_DIR  = PROJECT_ROOT / "baselines" / "results"

TARGET_SHAPE = (64, 80, 64)

def _load_volume(path: str | Path, target_shape=TARGET_SHAPE) -> np.ndarray | None:
    try:
        nii  = nib.load(str(path))
        data = np.asarray(nii.dataobj).astype(np.float32)
        if data.any():
            lo, hi = np.percentile(data[data > 0], [1, 99])
            data = np.clip(data, lo, hi)
        brain = data[data > 0]
        if len(brain) > 0:
            data = (data - brain.mean()) / (brain.std() + 1e-8)
        tensor  = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(tensor, size=target_shape, mode="trilinear",
                                align_corners=False)
        return resized.squeeze(0).numpy().astype(np.float32)
    except Exception as e:
        print(f"  Warning: failed to load {path}: {e}", file=sys.stderr)
        return None

class ADHDDataset(Dataset):

    def __init__(self, manifest: pd.DataFrame, labels: pd.DataFrame):
        self.manifest = manifest[manifest["dataset"] == "ADHD-200"].copy()
        self.labels   = labels.set_index("subject_id")
        self.items    = self._build_items()

    def _build_items(self):
        items = []
        for _, row in self.manifest.iterrows():
            sid = row["subject_id"]
            if sid not in self.labels.index:
                continue
            t1 = row.get("t1_path", "")
            if not t1 or not Path(str(t1)).exists():
                continue
            items.append({"subject_id": sid, "t1_path": str(t1),
                          "label": int(self.labels.loc[sid, "label"])})
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        vol  = _load_volume(item["t1_path"])
        if vol is None:
            vol = np.zeros((1, *TARGET_SHAPE), dtype=np.float32)
        return torch.from_numpy(vol).float(), torch.tensor(item["label"], dtype=torch.long)

class TumorDataset(Dataset):

    MODALITIES = ["t1_path", "t1c_path", "t2w_path", "t2f_path"]

    def __init__(self, manifest: pd.DataFrame, labels: pd.DataFrame):
        self.manifest = manifest[manifest["dataset"] == "BraTS-GLI"].copy()
        self.labels   = labels.dropna(subset=["grade_label"]).set_index("subject_id")
        self.items    = self._build_items()

    def _build_items(self):
        items = []
        for _, row in self.manifest.iterrows():
            sid = row["subject_id"]
            if sid not in self.labels.index:
                continue
            paths = [str(row.get(m, "") or "") for m in self.MODALITIES]
            if not any(Path(p).exists() for p in paths):
                continue
            items.append({
                "subject_id": sid,
                "paths": paths,
                "label": int(self.labels.loc[sid, "grade_label"]),
            })
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        channels = []
        for p in item["paths"]:
            vol = _load_volume(p) if p and Path(p).exists() else None
            if vol is None:
                vol = np.zeros((1, *TARGET_SHAPE), dtype=np.float32)
            channels.append(vol)
        tensor = np.concatenate(channels, axis=0)
        return torch.from_numpy(tensor).float(), torch.tensor(item["label"], dtype=torch.long)

class StrokeDataset(Dataset):

    LAT_MAP = {"left": 0, "right": 1, "bilateral": -1}

    def __init__(self, manifest: pd.DataFrame, labels: pd.DataFrame):
        self.manifest = manifest[manifest["dataset"] == "ATLAS-v2"].copy()
        self.labels   = labels.dropna(subset=["lateralization"]).set_index("subject_id")
        self.items    = self._build_items()

    def _build_items(self):
        items = []
        for _, row in self.manifest.iterrows():
            sid = row["subject_id"]
            if sid not in self.labels.index:
                continue
            t1 = str(row.get("t1_path", "") or "")
            if not t1 or not Path(t1).exists():
                continue
            lat_str = str(self.labels.loc[sid, "lateralization"])
            label   = self.LAT_MAP.get(lat_str, -1)
            if label < 0:
                continue
            items.append({"subject_id": sid, "t1_path": t1, "label": label})
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        vol  = _load_volume(item["t1_path"])
        if vol is None:
            vol = np.zeros((1, *TARGET_SHAPE), dtype=np.float32)
        return torch.from_numpy(vol).float(), torch.tensor(item["label"], dtype=torch.long)

class ResBlock3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.net(x) + x)

class CNN3D(nn.Module):

    def __init__(self, in_channels: int = 1, n_classes: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            ResBlock3D(32),
            nn.Conv3d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            ResBlock3D(64),
            nn.Conv3d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            ResBlock3D(128),
            nn.Conv3d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

def _split(items: list, val_frac=0.15, test_frac=0.15, seed=42) -> tuple[list, list, list]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items))
    n_test = int(len(items) * test_frac)
    n_val  = int(len(items) * val_frac)
    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return ([items[i] for i in train_idx],
            [items[i] for i in val_idx],
            [items[i] for i in test_idx])

def _metrics(y_true, y_pred_prob, n_classes=2) -> dict:
    y_pred = np.argmax(y_pred_prob, axis=1)
    result = {
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if n_classes == 2 and len(np.unique(y_true)) > 1:
        try:
            result["auc"] = round(float(roc_auc_score(y_true, y_pred_prob[:, 1])), 4)
        except Exception:
            pass
    return result

def train(condition: str, epochs: int, batch_size: int, lr: float,
          checkpoint_path: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    manifest = pd.read_csv(MANIFEST)

    if condition == "adhd":
        labels  = pd.read_csv(GT_DIR / "adhd_labels.csv")
        dataset = ADHDDataset(manifest, labels)
        in_ch   = 1
    elif condition == "tumor":
        labels  = pd.read_csv(GT_DIR / "tumor_labels.csv")
        dataset = TumorDataset(manifest, labels)
        in_ch   = 4
    elif condition == "stroke":
        print("Stroke lateralization is not trainable: local ATLAS-v2 subset contains "
              "only left-hemisphere cases (all label=0). Skipping.")
        return {"skipped": "all-left dataset, no right-hemisphere cases available"}
    else:
        raise ValueError(f"Unknown condition: {condition}")

    print(f"Dataset size: {len(dataset)} samples")
    if len(dataset) == 0:
        print("No data found, check ground-truth CSVs and manifest paths.")
        return {}

    train_items, val_items, test_items = _split(dataset.items)
    print(f"Split: train={len(train_items)}, val={len(val_items)}, test={len(test_items)}")

    def _subset(ds, items):
        ds_copy = type(ds).__new__(type(ds))
        ds_copy.__dict__.update(ds.__dict__)
        ds_copy.items = items
        return ds_copy

    train_loader = DataLoader(_subset(dataset, train_items), batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(_subset(dataset, val_items),   batch_size=batch_size,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(_subset(dataset, test_items),  batch_size=batch_size,
                              num_workers=4, pin_memory=True)

    model = CNN3D(in_channels=in_ch, n_classes=2).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    labels_in_train = [item["label"] for item in train_items]
    counts = np.bincount(labels_in_train, minlength=2).astype(float)
    weights = torch.tensor(1.0 / (counts + 1e-8), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1, best_epoch = 0.0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for X, y in val_loader:
                logits = model(X.to(device))
                val_preds.append(F.softmax(logits, dim=1).cpu().numpy())
                val_true.extend(y.numpy().tolist())
        val_preds = np.concatenate(val_preds)
        val_m = _metrics(val_true, val_preds)
        val_f1 = val_m["f1_macro"]
        print(f"  Epoch {epoch:3d}  loss={train_loss/len(train_loader):.4f}  "
              f"val_f1={val_f1:.4f}  val_bal_acc={val_m['balanced_accuracy']:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            torch.save(model.state_dict(), str(checkpoint_path))
            print(f"  New best, saved to {checkpoint_path}")

    print(f"\nLoading best checkpoint (epoch {best_epoch})...")
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for X, y in test_loader:
            logits = model(X.to(device))
            test_preds.append(F.softmax(logits, dim=1).cpu().numpy())
            test_true.extend(y.numpy().tolist())
    test_preds = np.concatenate(test_preds)
    test_m = _metrics(test_true, test_preds)
    print("\nTest results:")
    print(json.dumps(test_m, indent=2))
    return test_m

def evaluate_checkpoint(condition: str, checkpoint_path: Path, batch_size: int) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = pd.read_csv(MANIFEST)

    if condition == "adhd":
        labels  = pd.read_csv(GT_DIR / "adhd_labels.csv")
        dataset = ADHDDataset(manifest, labels)
        in_ch   = 1
    elif condition == "tumor":
        labels  = pd.read_csv(GT_DIR / "tumor_labels.csv")
        dataset = TumorDataset(manifest, labels)
        in_ch   = 4
    elif condition == "stroke":
        print("Stroke lateralization not evaluable: all-left dataset.")
        return {}
    else:
        raise ValueError(f"Unknown condition: {condition}")

    model = CNN3D(in_channels=in_ch, n_classes=2).to(device)
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    preds, true = [], []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X.to(device))
            preds.append(F.softmax(logits, dim=1).cpu().numpy())
            true.extend(y.numpy().tolist())
    preds = np.concatenate(preds)
    m = _metrics(true, preds)
    print(json.dumps(m, indent=2))
    return m

def main():
    parser = argparse.ArgumentParser(description="3D CNN baseline for NeuroAgent eval")
    parser.add_argument("--condition", required=True, choices=["adhd", "tumor", "stroke"])
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch-size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--out-dir",     default=str(OUT_DIR))
    parser.add_argument("--eval-only",   action="store_true")
    parser.add_argument("--checkpoint",  default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else \
                out_dir / f"{args.condition}_cnn_best.pt"

    if args.eval_only:
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}")
            sys.exit(1)
        metrics = evaluate_checkpoint(args.condition, ckpt_path, args.batch_size)
    else:
        metrics = train(args.condition, args.epochs, args.batch_size, args.lr, ckpt_path)

    results_path = out_dir / f"{args.condition}_cnn_results.json"
    results_path.write_text(json.dumps({"condition": args.condition, **metrics}, indent=2))
    print(f"\nSaved results to {results_path}")

if __name__ == "__main__":
    main()
