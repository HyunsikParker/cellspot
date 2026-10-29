"""How the item check behaves on the held-out test photos with the exported model and the app's rule: name the device
when the top probability is at least CONFIDENT (0.6, fixed before looking at the test set), otherwise ask the question.
Every XBAT+ device type has a battery, so a confident wrong name still gives the right advice; the split shows how often
a resident would see a name, a wrong name, or the question.

usage: python check_policy.py RUN_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from common import items
from train_classifier import MEAN, SIZE, STD, square_crop

CONFIDENT = 0.6


def main():
    run = Path(sys.argv[1])
    size = json.loads((run / "metrics.json").read_text()).get("settings", {}).get("size", SIZE)
    sess = ort.InferenceSession(str(run / "classifier.onnx"))
    named = wrong = asked = 0
    for path, boxes in items("RGB", "test"):
        crop = square_crop(Image.open(path).convert("RGB"), boxes[0], 0.15).resize((size, size))
        x = ((np.asarray(crop, np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        p = sess.run(None, {"image": x})[0][0]
        if p.max() >= CONFIDENT:
            named += 1
            wrong += int(p.argmax() != boxes[0][0])
        else:
            asked += 1
    result = {"threshold": CONFIDENT, "photos": named + asked, "named": named, "named_wrong": wrong, "asked": asked}
    metrics = json.loads((run / "metrics.json").read_text())
    metrics["app_rule"] = result
    (run / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
