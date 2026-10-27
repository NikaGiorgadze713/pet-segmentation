import argparse
import os
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import (
    CalibrationDataReader, QuantFormat, QuantType, quantize_static,
)

from src.dataset import PetSegmentation
from src.model import get_model


def load_model(model_name, checkpoint):
    model = get_model(model_name)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    return model


def export_onnx(model, path, img_size=256):
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model, dummy, path,
        input_names=["image"], output_names=["logits"],
        opset_version=17, dynamo=False,
    )


class CalibReader(CalibrationDataReader):
    def __init__(self, n=64):
        ds = PetSegmentation("data", split="trainval")
        self.items = iter([ds[i][0].unsqueeze(0).numpy() for i in range(n)])

    def get_next(self):
        x = next(self.items, None)
        return None if x is None else {"image": x}


def quantize(fp32_path, int8_path):
    quantize_static(
        fp32_path, int8_path, CalibReader(),
        quant_format=QuantFormat.QDQ, per_channel=True,
        weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
    )


def make_runner(kind, path, model_name):
    if kind == "pytorch":
        model = load_model(model_name, path)

        def run(x):
            with torch.no_grad():
                return model(torch.from_numpy(x)).numpy()
        return run

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return lambda x: sess.run(None, {"image": x})[0]


def iou(logits, mask, eps=1e-7):
    pred = (logits > 0).astype(np.float32)
    inter = (pred * mask).sum()
    union = pred.sum() + mask.sum() - inter
    return (inter + eps) / (union + eps)


def benchmark(run, samples, n_speed=50):
    x = samples[0][0]
    for _ in range(5):
        run(x)
    start = time.perf_counter()
    for _ in range(n_speed):
        run(x)
    ms = (time.perf_counter() - start) / n_speed * 1000

    scores = [iou(run(x), m) for x, m in samples]
    return ms, float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resnet-checkpoint", required=True)
    parser.add_argument("--mobilenet-checkpoint", required=True)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--out-dir", default="exports")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    variants = []
    for model_name, ckpt in [
        ("unet_resnet34", args.resnet_checkpoint),
        ("unet_mobilenet", args.mobilenet_checkpoint),
    ]:
        short = model_name.replace("unet_", "")
        fp32 = f"{args.out_dir}/{short}_fp32.onnx"
        int8 = f"{args.out_dir}/{short}_int8.onnx"
        print(f"Exporting {short} to ONNX")
        export_onnx(load_model(model_name, ckpt), fp32)
        print(f"Quantizing {short} to INT8")
        quantize(fp32, int8)
        variants += [
            (f"{short} pytorch fp32", "pytorch", ckpt, model_name),
            (f"{short} onnx fp32", "onnx", fp32, model_name),
            (f"{short} onnx int8", "onnx", int8, model_name),
        ]

    test_ds = PetSegmentation("data", split="test")
    samples = []
    for i in range(args.n_test):
        image, mask = test_ds[i]
        samples.append((image.unsqueeze(0).numpy(), mask.unsqueeze(0).numpy()))

    rows = []
    for name, kind, path, model_name in variants:
        print(f"Benchmarking {name}")
        run = make_runner(kind, path, model_name)
        ms, test_iou = benchmark(run, samples)
        rows.append({
            "version": name,
            "size_mb": round(os.path.getsize(path) / 1e6, 1),
            "cpu_ms_per_image": round(ms, 1),
            "test_iou": round(test_iou, 4),
        })

    table = pd.DataFrame(rows)
    table.to_csv("results/optimization.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()