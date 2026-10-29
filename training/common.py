"""XBAT+ 1.0 (resized release) loading.

The release ships YOLO label files but no class names. The orders below were checked against the images and the
per-class counts: the RGB classes are the 15 device types in case-insensitive alphabetical order, and the X-ray classes
add "Battery" (a battery inside the device) and "NoBattery" (an empty battery compartment) in the same ordering.
"""
import os
from pathlib import Path

RGB_CLASSES = ["AlarmClock", "ComputerMouse", "E-Razor", "E-Toothbrush", "FireAlarm", "FlashLight", "GameController",
               "iPad", "KidToy", "MobilePhone", "NewsRadio", "NightTorch", "PhotoCamera", "RemoteControl", "Thermometer"]
XRAY_CLASSES = ["AlarmClock", "Battery", "ComputerMouse", "E-Razor", "E-Toothbrush", "FireAlarm", "FlashLight",
                "GameController", "iPad", "KidToy", "MobilePhone", "NewsRadio", "NightTorch", "NoBattery", "PhotoCamera",
                "RemoteControl", "Thermometer"]
BATTERY, NO_BATTERY = XRAY_CLASSES.index("Battery"), XRAY_CLASSES.index("NoBattery")

DATA = Path(os.environ.get("XBAT_DIR", Path(__file__).resolve().parents[2] / "cellspot-data" / "xbat"))


def split_dirs(kind, split):
    """kind is RGB, HQ or VQ; split is train or test."""
    sub = DATA / f"res_XBAT+_v1.0_{kind}_{split}" / f"{split}_data"
    return sub / "images", sub / "labels"


def read_labels(path):
    """(class, x0, y0, x1, y1) in 0-1 image coordinates. A few rows are polygons; they become their bounding box."""
    boxes = []
    for line in path.read_text().splitlines():
        v = line.split()
        if not v:
            continue
        c, nums = int(float(v[0])), [float(x) for x in v[1:]]
        if len(nums) == 4:
            x, y, w, h = nums
            x0, y0, x1, y1 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        else:
            xs, ys = nums[0::2], nums[1::2]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        boxes.append((c, max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)))
    return boxes


def items(kind, split):
    images, labels = split_dirs(kind, split)
    out = []
    for img in sorted(images.glob("*.jpg")):
        lab = labels / f"{img.stem}.txt"
        out.append((img, read_labels(lab) if lab.exists() else []))
    return out


def device_id(path):
    """The leading number names the physical device; the same device appears in the RGB, HQ and VQ sets."""
    return Path(path).name.split("_")[0]


def device_split(rows, label_of, fraction, seed):
    """Hold out about `fraction` of the devices of each class, so validation never shares a device with training."""
    import random
    rng = random.Random(seed)
    by_class = {}
    for r in rows:
        by_class.setdefault(label_of(r), set()).add(device_id(r[0]))
    held = set()
    for devices in by_class.values():
        devices = sorted(devices)
        rng.shuffle(devices)
        held.update(devices[:max(1, round(len(devices) * fraction))] if len(devices) >= 4 else [])
    train = [r for r in rows if device_id(r[0]) not in held]
    val = [r for r in rows if device_id(r[0]) in held]
    return train, val


def device_fold(rows, label_of, fold, folds, seed=0):
    """Fold `fold` of `folds` device-disjoint folds, stratified by class: (train, validation)."""
    import random
    rng = random.Random(seed)
    by_class = {}
    for r in rows:
        by_class.setdefault(label_of(r), set()).add(device_id(r[0]))
    held = set()
    for devices in by_class.values():
        devices = sorted(devices)
        rng.shuffle(devices)
        held.update(d for i, d in enumerate(devices) if i % folds == fold)
    return [r for r in rows if device_id(r[0]) not in held], [r for r in rows if device_id(r[0]) in held]
