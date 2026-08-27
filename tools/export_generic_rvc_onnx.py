#!/usr/bin/env python3
"""Build the voice-neutral RVC v2/F0/40k synthesizer used by PTHChanger.

The ONNX model contains random placeholder initializers with the exact same
names/shapes as the supported RVC checkpoint. Android replaces those
initializers from the user's .pth before creating the ORT session.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import onnx
import torch
from torch import nn

# JayModern-compatible RVC v2, F0, 40 kHz profile.
CONFIG = [
    1025, 32, 192, 192, 768, 2, 6, 3, 0,
    "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000,
]

NORMAL_INPUTS = {"phone", "phone_lengths", "pitch", "pitchf", "sid", "rnd"}


class InferenceWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, phone, phone_lengths, pitch, pitchf, sid, rnd):
        g = self.model.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.model.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * rnd * 0.66666) * x_mask
        z = self.model.flow(z_p, x_mask, g=g, reverse=True)
        return self.model.dec(z * x_mask, pitchf, g=g)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rvc-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    rvc_root = Path(args.rvc_root).resolve()
    sys.path.insert(0, str(rvc_root))
    from infer.module.models import SynthesizerTrnMs768NSFsid

    torch.manual_seed(1234)
    model = SynthesizerTrnMs768NSFsid(*CONFIG, is_half=False)
    if hasattr(model, "enc_q"):
        del model.enc_q
    model.eval().float()
    wrapper = InferenceWrapper(model).eval()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    t = 16
    inputs = (
        torch.zeros(1, t, 768, dtype=torch.float32),
        torch.tensor([t], dtype=torch.long),
        torch.ones(1, t, dtype=torch.long),
        torch.full((1, t), 120.0, dtype=torch.float32),
        torch.tensor([1], dtype=torch.long),
        torch.zeros(1, 192, t, dtype=torch.float32),
    )

    print("Exporting generic RVC synthesizer...")
    torch.onnx.export(
        wrapper,
        inputs,
        str(out),
        input_names=["phone", "phone_lengths", "pitch", "pitchf", "sid", "rnd"],
        output_names=["waveform"],
        dynamic_axes={
            "phone": {1: "frames"},
            "pitch": {1: "frames"},
            "pitchf": {1: "frames"},
            "rnd": {2: "frames"},
            "waveform": {2: "samples"},
        },
        opset_version=17,
        do_constant_folding=False,
        keep_initializers_as_inputs=True,
        dynamo=False,
    )

    graph = onnx.load(str(out))
    onnx.checker.check_model(graph)

    state = wrapper.state_dict()
    initializer_names = {x.name for x in graph.graph.initializer}
    weights = []
    unmatched_initializers = []

    for init in graph.graph.initializer:
        name = init.name
        checkpoint_name = None
        if name in state:
            checkpoint_name = name.removeprefix("model.")
        elif f"model.{name}" in state:
            checkpoint_name = name
        if checkpoint_name is None:
            unmatched_initializers.append(name)
            continue
        tensor = state[name if name in state else f"model.{name}"]
        weights.append({
            "onnxName": name,
            "checkpointName": checkpoint_name,
            "shape": list(tensor.shape),
            "dtype": "float32",
        })

    expected = {
        key.removeprefix("model.")
        for key in state.keys()
        if key.startswith("model.")
    }
    mapped = {w["checkpointName"] for w in weights}
    missing = sorted(expected - mapped)

    print(f"State tensors: {len(expected)}")
    print(f"Mapped initializers: {len(weights)}")
    print(f"Unmatched ONNX initializers: {len(unmatched_initializers)}")
    print(f"State tensors absent from ONNX: {len(missing)}")
    if unmatched_initializers:
        print("UNMATCHED_INITIALIZERS", unmatched_initializers[:30])
    if missing:
        print("MISSING_STATE", missing[:30])
        raise RuntimeError("Not every inference checkpoint tensor maps to an ONNX initializer")

    manifest = {
        "format": 1,
        "version": "v2",
        "f0": True,
        "sampleRate": 40000,
        "speakerCount": 109,
        "normalInputs": ["phone", "phone_lengths", "pitch", "pitchf", "sid", "rnd"],
        "weights": weights,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Generic synth: {out} ({out.stat().st_size / 1024**2:.1f} MiB)")
    print(f"Manifest: {manifest_path}")

    # CPU smoke test. This proves the generated graph itself is executable.
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "phone": np.zeros((1, t, 768), np.float32),
        "phone_lengths": np.asarray([t], np.int64),
        "pitch": np.ones((1, t), np.int64),
        "pitchf": np.full((1, t), 120.0, np.float32),
        "sid": np.asarray([1], np.int64),
        "rnd": np.zeros((1, 192, t), np.float32),
    }
    result = sess.run(["waveform"], ort_inputs)[0]
    print("ORT synth smoke output:", result.shape, result.dtype, float(np.max(np.abs(result))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
