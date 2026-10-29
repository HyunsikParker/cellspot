"""Find batteries inside devices on X-ray images: Faster R-CNN MobileNetV3-Large FPN (COCO weights) fine-tuned on XBAT+
X-ray images (17 classes: 15 device types, Battery and NoBattery, an empty battery compartment).

Trains on VQ train (it contains HQ train) minus a device-disjoint validation slice, which is used only to pick the
score threshold for the facility view (the lowest threshold that keeps validation precision at or above --precision).
Reports AP50 per class and image-level battery detection on the device-disjoint VQ and HQ test sets.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor

from common import BATTERY, NO_BATTERY, XRAY_CLASSES, device_split, items

SIZE = 640


def device_class(row):
    return next((c for c, *_ in row[1] if c not in (BATTERY, NO_BATTERY)), -1)


class XRays(torch.utils.data.Dataset):
    def __init__(self, rows, train):
        self.rows, self.train = rows, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, boxes = self.rows[i]
        img = Image.open(path).convert("RGB")
        b = np.array([[x0, y0, x1, y1] for _, x0, y0, x1, y1 in boxes], dtype=np.float32).reshape(-1, 4) * SIZE
        labels = np.array([c + 1 for c, *_ in boxes], dtype=np.int64)  # 0 is background
        if self.train:
            r = random.Random(torch.initial_seed() + i)
            if r.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                b[:, [0, 2]] = SIZE - b[:, [2, 0]]
            if r.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                b[:, [1, 3]] = SIZE - b[:, [3, 1]]
            img = ImageEnhance.Contrast(img).enhance(r.uniform(0.75, 1.3))
            img = ImageEnhance.Brightness(img).enhance(r.uniform(0.8, 1.2))
        keep = (b[:, 2] - b[:, 0] > 1) & (b[:, 3] - b[:, 1] > 1)
        return to_tensor(img), {"boxes": torch.from_numpy(b[keep]), "labels": torch.from_numpy(labels[keep])}


def collate(batch):
    return tuple(zip(*batch))


def build(num_classes):
    m = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1,
                                          min_size=SIZE, max_size=SIZE, box_detections_per_img=50)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, num_classes + 1)
    return m


@torch.no_grad()
def detect(model, rows, dev):
    model.eval()
    out = []
    data = torch.utils.data.DataLoader(XRays(rows, False), batch_size=8, num_workers=4, collate_fn=collate)
    for imgs, _ in data:
        for d in model([x.to(dev) for x in imgs]):
            out.append({k: v.cpu().numpy() for k, v in d.items()})
    return out


def iou(a, b):
    x0, y0 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x1, y1 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter)


def ap50(dets, rows, cls):
    """All-point interpolated average precision at IoU 0.5 for one class (VOC style)."""
    gts = [np.array([[x0, y0, x1, y1] for c, x0, y0, x1, y1 in r[1] if c == cls]).reshape(-1, 4) * SIZE for r in rows]
    n_gt = sum(len(g) for g in gts)
    if n_gt == 0:
        return None
    cand = [(s, i, b) for i, d in enumerate(dets) for b, l, s in zip(d["boxes"], d["labels"], d["scores"]) if l == cls + 1]
    cand.sort(key=lambda t: -t[0])
    used = [np.zeros(len(g), bool) for g in gts]
    tp = np.zeros(len(cand))
    for k, (_, i, b) in enumerate(cand):
        if len(gts[i]):
            o = iou(b, gts[i])
            j = int(o.argmax())
            if o[j] >= 0.5 and not used[i][j]:
                used[i][j], tp[k] = True, 1
    if not len(cand):
        return 0.0
    ctp = np.cumsum(tp)
    recall, precision = ctp / n_gt, ctp / np.arange(1, len(cand) + 1)
    mrec, mpre = np.concatenate([[0], recall, [1]]), np.concatenate([[0], precision, [0]])
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def battery_flags(dets, rows, threshold):
    """Image level: does the image show a battery? Truth is any Battery box; the model says yes above threshold."""
    truth = np.array([any(c == BATTERY for c, *_ in r[1]) for r in rows])
    best = np.array([max([s for l, s in zip(d["labels"], d["scores"]) if l == BATTERY + 1], default=0.0) for d in dets])
    said = best >= threshold
    tp, fp, fn = int((said & truth).sum()), int((said & ~truth).sum()), int((~said & truth).sum())
    return {"threshold": threshold, "images": int(len(rows)), "with_battery": int(truth.sum()), "found": tp,
            "missed": fn, "false_alarms": fp, "recall": round(tp / max(1, tp + fn), 4),
            "precision": round(tp / max(1, tp + fp), 4)}, best, truth


def pick_threshold(best, truth, min_precision):
    """Lowest threshold whose validation precision stays at or above min_precision (more batteries found)."""
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        said = best >= t
        if said.sum() and (said & truth).sum() / said.sum() >= min_precision:
            return float(t)
    return 0.5


def evaluate(model, rows, dev, threshold):
    dets = detect(model, rows, dev)
    per_class = {XRAY_CLASSES[c]: ap50(dets, rows, c) for c in range(len(XRAY_CLASSES))}
    present = {k: round(v, 4) for k, v in per_class.items() if v is not None}
    flags, _, _ = battery_flags(dets, rows, threshold)
    return {"images": len(rows), "ap50": present, "map50": round(float(np.mean(list(present.values()))), 4),
            "battery_ap50": present.get("Battery"), "no_battery_ap50": present.get("NoBattery"), "battery_images": flags}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--precision", type=float, default=0.9)
    ap.add_argument("--out", default="runs/detector")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    rows = [r for r in items("VQ", "train") if r[1]]
    fit, val = device_split(rows, device_class, 0.1, seed=0)
    tests = {"VQ_test": items("VQ", "test"), "HQ_test": items("HQ", "test")}
    print(json.dumps({"train": len(fit), "val": len(val), **{k: len(v) for k, v in tests.items()}, "device": dev}))

    model = build(len(XRAY_CLASSES)).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)
    data = torch.utils.data.DataLoader(XRays(fit, True), batch_size=args.batch, shuffle=True, num_workers=4,
                                       collate_fn=collate, drop_last=True)
    steps = args.epochs * len(data)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.01, total_steps=steps, pct_start=0.05)
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for imgs, targets in data:
            loss = sum(model([x.to(dev) for x in imgs], [{k: v.to(dev) for k, v in t.items()} for t in targets]).values())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            sched.step()
            total += loss.item()
        print(json.dumps({"epoch": ep, "loss": round(total / len(data), 4), "s": round(time.time() - t0, 1)}), flush=True)

    vdets = detect(model, val, dev)
    _, vbest, vtruth = battery_flags(vdets, val, 0.5)
    threshold = pick_threshold(vbest, vtruth, args.precision)
    val_flags, _, _ = battery_flags(vdets, val, threshold)
    result = {"epochs": args.epochs, "train_images": len(fit), "threshold": threshold, "validation": val_flags,
              "classes": XRAY_CLASSES, **{k: evaluate(model, v, dev, threshold) for k, v in tests.items()}}
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    torch.save(model.state_dict(), out / "detector.pth")
    print(json.dumps({k: {"map50": result[k]["map50"], "battery_ap50": result[k]["battery_ap50"],
                          "battery_images": result[k]["battery_images"]} for k in tests}))



if __name__ == "__main__":
    main()
