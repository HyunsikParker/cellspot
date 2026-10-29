"""Shrink the detector for the web: static int8 quantization (QDQ) of its convolutions and matrix products, calibrated
on training X-rays. Prints the size and how often the int8 model still flags the same images as the float model.

usage: python quantize_detector.py RUN_DIR [CALIBRATION_IMAGES]
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image

from common import BATTERY, items


def tensor(path):
    return (np.asarray(Image.open(path).convert("RGB"), np.float32) / 255).transpose(2, 0, 1)


class Reader(CalibrationDataReader):
    def __init__(self, paths):
        self.it = iter(paths)

    def get_next(self):
        p = next(self.it, None)
        return None if p is None else {"image": tensor(p)}


def flags(sess, rows, threshold):
    out = []
    for p, _ in rows:
        boxes, labels, scores = sess.run(None, {"image": tensor(p)})
        out.append(bool(((labels == BATTERY + 1) & (scores >= threshold)).any()))
    return np.array(out)


def main():
    run = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    src, pre, dst = run / "detector.onnx", run / "detector.pre.onnx", run / "detector.int8.onnx"
    quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
    calib = [p for p, b in items("VQ", "train") if b][:: max(1, 1700 // n)][:n]
    quantize_static(str(pre), str(dst), Reader(calib), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, op_types_to_quantize=["Conv", "MatMul", "Gemm"], per_channel=True)
    test = items("HQ", "test")
    a, b = flags(ort.InferenceSession(str(src)), test, 0.5), flags(ort.InferenceSession(str(dst)), test, 0.5)
    truth = np.array([any(c == BATTERY for c, *_ in r[1]) for r in test])
    print(f"float {src.stat().st_size / 1e6:.1f} MB, int8 {dst.stat().st_size / 1e6:.1f} MB; same flag on {int((a == b).sum())}/{len(test)}; "
          f"recall float {(a & truth).sum()}/{truth.sum()}, int8 {(b & truth).sum()}/{truth.sum()}")


if __name__ == "__main__":
    main()
