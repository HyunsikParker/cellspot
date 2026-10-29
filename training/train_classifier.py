"""Name the device in a webcam photo: MobileNetV3-Large (ImageNet weights) fine-tuned on crops around the labelled
device in XBAT+ RGB (15 battery-containing device types).

The number of epochs is chosen on a device-disjoint validation slice of the training set; the model is then retrained
on the whole training set for that many epochs and evaluated once on the held-out test set. Writes the metrics and the weights;
export_onnx.py turns them into ONNX for the browser.
"""
import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from common import RGB_CLASSES, device_fold, device_split, items

SIZE = 224  # input side; --size changes it for a run
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def square_crop(img, box, margin, jitter=None):
    """A square around the labelled device, `margin` of its longer side on each side."""
    W, H = img.size
    _, x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2 * W, (y0 + y1) / 2 * H
    side = max((x1 - x0) * W, (y1 - y0) * H) * (1 + 2 * margin)
    if jitter:
        cx += jitter.uniform(-0.08, 0.08) * side
        cy += jitter.uniform(-0.08, 0.08) * side
    return img.crop((round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2)))


class Crops(torch.utils.data.Dataset):
    def __init__(self, rows, train):
        self.rows, self.train = rows, train
        self.seed, self.r = None, None
        norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
        self.tf = T.Compose([T.Resize((SIZE, SIZE)), T.RandomHorizontalFlip(), T.RandomRotation(12),
                             T.ColorJitter(0.35, 0.35, 0.3, 0.04), *norm]) if train else T.Compose([T.Resize((SIZE, SIZE)), *norm])

    def __len__(self):
        return len(self.rows)

    @property
    def rng(self):
        # torch gives every worker, in every epoch, its own initial seed; follow it so crops differ between workers.
        seed = torch.initial_seed() % 2 ** 32
        if seed != self.seed:
            self.seed, self.r = seed, random.Random(seed)
        return self.r

    def __getitem__(self, i):
        path, boxes = self.rows[i]
        img = Image.open(path).convert("RGB")
        margin = self.rng.uniform(0.05, 0.35) if self.train else 0.15
        return self.tf(square_crop(img, boxes[0], margin, self.rng if self.train else None)), boxes[0][0]


def model_for(n):
    m = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, n)
    return m


BALANCE = 0.5  # rare device types are drawn count**-BALANCE times as often; --balance changes it


def loader(rows, train, batch=32):
    ds = Crops(rows, train)
    if not train:
        return torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=4)
    counts = Counter(r[1][0][0] for r in rows)
    weights = [counts[r[1][0][0]] ** -BALANCE for r in rows]
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(rows), replacement=True)
    return torch.utils.data.DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=4, drop_last=True)


@torch.no_grad()
def predict(model, rows, dev):
    model.eval()
    probs = []
    for x, _ in loader(rows, False):
        probs.append(torch.softmax(model(x.to(dev)), 1).cpu())
    return torch.cat(probs).numpy()


def scores(probs, rows):
    y = np.array([r[1][0][0] for r in rows])
    pred = probs.argmax(1)
    top2 = (np.argsort(-probs, 1)[:, :2] == y[:, None]).any(1)
    per_class = {}
    for c, name in enumerate(RGB_CLASSES):
        mask = y == c
        if mask.any():
            per_class[name] = {"n": int(mask.sum()), "recall": round(float((pred[mask] == c).mean()), 3)}
    confusion = np.zeros((len(RGB_CLASSES), len(RGB_CLASSES)), dtype=int)
    for t, p in zip(y, pred):
        confusion[t, p] += 1
    return {"n": int(len(y)), "accuracy": round(float((pred == y).mean()), 4), "top2": round(float(top2.mean()), 4),
            "macro_recall": round(float(np.mean([v["recall"] for v in per_class.values()])), 4),
            "per_class": per_class, "confusion": confusion.tolist()}


def train(rows, epochs, dev, val_rows=None, log=print, lr=5e-4):
    torch.manual_seed(0)
    model = model_for(len(RGB_CLASSES)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    data = loader(rows, True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(data), pct_start=0.15)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in data:
            opt.zero_grad()
            loss = lossf(model(x.to(dev)), y.to(dev))
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        row = {"epoch": ep, "loss": round(total / len(data), 4), "s": round(time.time() - t0, 1)}
        if val_rows:
            s = scores(predict(model, val_rows, dev), val_rows)
            row.update(val_accuracy=s["accuracy"], val_macro_recall=s["macro_recall"])
        history.append(row)
        log(json.dumps(row))
    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default="runs/classifier")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--balance", type=float, default=0.5)
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    global SIZE, BALANCE
    SIZE, BALANCE = args.size, args.balance
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows, test = items("RGB", "train"), items("RGB", "test")
    if args.folds:
        # Cross-validation on the training devices only: one fold's best-epoch validation scores, no test set.
        fit, val = device_fold(rows, lambda r: r[1][0][0], args.fold, args.folds)
        _, hist = train(fit, args.epochs, dev, val, lr=args.lr)
        best = max(hist, key=lambda h: (h["val_macro_recall"], h["val_accuracy"], h["epoch"]))
        last = hist[-1]
        (out / f"cv_fold{args.fold}.json").write_text(json.dumps({"settings": {"lr": args.lr, "size": SIZE, "balance": BALANCE},
                                                                "fold": args.fold, "folds": args.folds, "val": len(val), "best": best, "last": last}))
        print(json.dumps({"fold": args.fold, "best": best, "last": last}))
        return
    fit, val = device_split(rows, lambda r: r[1][0][0], 0.15, seed=0)
    print(json.dumps({"train": len(fit), "val": len(val), "test": len(test), "device": dev}))

    _, hist = train(fit, args.epochs, dev, val, lr=args.lr)
    # Choose the epoch count by validation macro recall (ties go to the later, smoother epoch).
    best = max(hist, key=lambda h: (h["val_macro_recall"], h["val_accuracy"], h["epoch"]))
    model, _ = train(rows, best["epoch"], dev, lr=args.lr)
    result = {"settings": {"lr": args.lr, "size": SIZE, "balance": BALANCE}, "chosen_epochs": best["epoch"],
              "validation_at_choice": best, "test": scores(predict(model, test, dev), test),
              "classes": RGB_CLASSES, "train_images": len(rows), "test_images": len(test)}
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    torch.save(model.state_dict(), out / "classifier.pth")

    print(json.dumps({k: result["test"][k] for k in ("n", "accuracy", "top2", "macro_recall")}))


if __name__ == "__main__":
    main()
