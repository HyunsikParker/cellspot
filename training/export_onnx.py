"""Turn the trained weights into ONNX for the browser and check that onnxruntime gives the same answers as PyTorch on
a few test images. Builds the architectures without pretrained weights, so nothing is downloaded here.

usage: python export_onnx.py RUN_DIR [classifier|detector ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import mobilenet_v3_large
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms.functional import to_tensor

from common import RGB_CLASSES, XRAY_CLASSES, items
from train_classifier import MEAN, SIZE as CLS_SIZE, STD, square_crop
from train_detector import SIZE as DET_SIZE


def run_size(run):
    return json.loads((run / "metrics.json").read_text()).get("settings", {}).get("size", CLS_SIZE)


def classifier(run):
    size = run_size(run)
    m = mobilenet_v3_large(weights=None, num_classes=len(RGB_CLASSES))
    m.load_state_dict(torch.load(run / "classifier.pth", map_location="cpu"))
    model = nn.Sequential(m, nn.Softmax(dim=1)).eval()
    path = run / "classifier.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, size, size), path, input_names=["image"], output_names=["probs"],
                      dynamic_axes={"image": {0: "n"}, "probs": {0: "n"}}, opset_version=17, dynamo=False)
    sess = ort.InferenceSession(str(path))
    worst = 0.0
    for img_path, boxes in items("RGB", "test")[:8]:
        crop = square_crop(Image.open(img_path).convert("RGB"), boxes[0], 0.15).resize((size, size))
        x = ((np.asarray(crop, np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        with torch.no_grad():
            ref = model(torch.from_numpy(x)).numpy()
        worst = max(worst, float(np.abs(sess.run(None, {"image": x})[0] - ref).max()))
    print(f"classifier.onnx {path.stat().st_size / 1e6:.1f} MB, max |onnx - torch| = {worst:.2e}")


def detector(run):
    m = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None, num_classes=len(XRAY_CLASSES) + 1,
                                          min_size=DET_SIZE, max_size=DET_SIZE, box_detections_per_img=50)
    m.load_state_dict(torch.load(run / "detector.pth", map_location="cpu"))
    m.eval()
    path = run / "detector.onnx"
    # Trace with a real X-ray: a blank image yields no proposals and the trace then fixes degenerate shapes.
    example = to_tensor(Image.open(items("HQ", "train")[0][0]).convert("RGB"))
    torch.onnx.export(m, ([example],), path, input_names=["image"], output_names=["boxes", "labels", "scores"],
                      opset_version=17, dynamo=False)
    sess = ort.InferenceSession(str(path))
    worst, same = 0.0, True
    for img_path, _ in items("HQ", "test")[:4]:
        x = to_tensor(Image.open(img_path).convert("RGB"))
        with torch.no_grad():
            ref = m([x])[0]
        boxes, labels, scores = sess.run(None, {"image": x.numpy()})
        same &= len(labels) == len(ref["labels"]) and bool((labels == ref["labels"].numpy()).all())
        if len(boxes) and same:
            worst = max(worst, float(np.abs(boxes - ref["boxes"].numpy()).max()))
    print(f"detector.onnx {path.stat().st_size / 1e6:.1f} MB, same labels: {same}, max box |onnx - torch| = {worst:.2e} px")


if __name__ == "__main__":
    run = Path(sys.argv[1])
    for what in sys.argv[2:] or ["classifier", "detector"]:
        {"classifier": classifier, "detector": detector}[what](run)
