"""Copy the exported models and their metrics into web/models, and pick test photos and test X-rays for the demo
(web/samples, with manifest.json). Samples come only from the XBAT+ test splits, which share no device with training.

usage: python make_web_assets.py CLASSIFIER_RUN DETECTOR_RUN
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from common import BATTERY, NO_BATTERY, RGB_CLASSES, XRAY_CLASSES, device_id, items

WEB = Path(__file__).resolve().parents[1] / "web"
CHECK = ["MobilePhone", "RemoteControl", "E-Toothbrush", "KidToy", "GameController", "FireAlarm"]


def photo_crop(img, box, margin=0.15):
    """Square crop around the device that stays inside the photo: the release pads photos to 640x640 with flat bars,
    which would otherwise show up in the demo thumbnails."""
    a = np.asarray(img.convert("L"), dtype=np.float32)
    rows = np.where(a.std(1) > 3)[0]  # padding bars are flat; photo rows are not
    cols = np.where(a.std(0) > 3)[0]
    top, bottom, left, right = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    W, H = img.size
    _, x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2 * W, (y0 + y1) / 2 * H
    side = min(max((x1 - x0) * W, (y1 - y0) * H) * (1 + 2 * margin), right - left, bottom - top)
    cx = min(max(cx, left + side / 2), right - side / 2)
    cy = min(max(cy, top + side / 2), bottom - side / 2)
    return img.crop((round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2)))


def main():
    cls_run, det_run = Path(sys.argv[1]), Path(sys.argv[2])
    models = WEB / "models"
    models.mkdir(parents=True, exist_ok=True)
    for run, name in ((cls_run, "classifier"), (det_run, "detector")):
        if not (run / f"{name}.onnx").exists():
            print(f"skipped {name}: no {run / f'{name}.onnx'} yet")
            continue
        shutil.copyfile(run / f"{name}.onnx", models / f"{name}.onnx")
        metrics = json.loads((run / "metrics.json").read_text())
        if name == "detector":
            metrics["test_devices"] = len({device_id(p) for p, _ in items("VQ", "test")})
        (models / f"metrics-{name}.json").write_text(json.dumps(metrics, indent=1))

    manifest = {"source": "XBAT+ 1.0 test splits (Zenodo 18022530, CC BY 4.0)", "check": [], "xray": []}
    out = WEB / "samples" / "check"
    out.mkdir(parents=True, exist_ok=True)
    taken = set()
    for path, boxes in items("RGB", "test"):
        name = RGB_CLASSES[boxes[0][0]]
        if name in CHECK and name not in taken:
            taken.add(name)
            crop = photo_crop(Image.open(path).convert("RGB"), boxes[0]).resize((448, 448), Image.LANCZOS)
            file = f"{name.lower()}.jpg"
            crop.save(out / file, quality=88)
            manifest["check"].append({"file": file, "device": name, "source": path.name})
    manifest["check"].sort(key=lambda s: CHECK.index(s["device"]))

    out = WEB / "samples" / "xray"
    out.mkdir(parents=True, exist_ok=True)
    rows = items("HQ", "test")
    with_battery = [r for r in rows if any(c == BATTERY for c, *_ in r[1])]
    without = [r for r in rows if not any(c == BATTERY for c, *_ in r[1])]
    picks, seen = [], set()
    # Eight with a battery and four without, one device type each where possible.
    for group, n in ((with_battery, 8), (without, 4)):
        got = 0
        for path, boxes in group:
            dev = next((XRAY_CLASSES[c] for c, *_ in boxes if c not in (BATTERY, NO_BATTERY)), None)
            if dev and (dev, n) not in seen and got < n:
                seen.add((dev, n))
                picks.append((path, boxes, dev))
                got += 1
    for k, (path, boxes, dev) in enumerate(picks, 1):
        file = f"xray-{k:02d}.jpg"
        shutil.copyfile(path, out / file)
        manifest["xray"].append({"file": file, "device": dev, "battery": any(c == BATTERY for c, *_ in boxes),
                                 "empty_compartment": any(c == NO_BATTERY for c, *_ in boxes), "source": path.name})
    (WEB / "samples" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"check": [s["device"] for s in manifest["check"]],
                      "xray": [(s["device"], s["battery"]) for s in manifest["xray"]]}))


if __name__ == "__main__":
    main()
