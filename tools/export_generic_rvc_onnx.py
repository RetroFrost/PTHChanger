#!/usr/bin/env python3
"""Build the voice-neutral RVC v2/F0/40k synthesizer used by PTHChanger.

Upstream RVC relative-attention code converts sequence lengths to Python ints
while tracing. A nominally dynamic legacy ONNX export therefore bakes reshape
sizes and breaks at any length other than the traced one. PTHChanger avoids
that trap deliberately: it exports a fixed 400-frame (4 second) graph, while
Android pads every chunk to 400 frames and crops the synthesized result back
to the chunk's real length. Long files are handled by overlapping chunks.

The ONNX model contains placeholder initializers with the exact same
names/shapes as the supported RVC checkpoint. Android replaces all 457
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

FIXED_FRAMES = 400
SAMPLES_PER_FRAME = 400

CONFIG = [
    1025, 32, 192, 192, 768, 2, 6, 3, 0,
    "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000,
]


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


def make_inputs(np, actual_frames: int = FIXED_FRAMES):
    assert 1 <= actual_frames <= FIXED_FRAMES
    phone = np.zeros((1, FIXED_FRAMES, 768), np.float32)
    pitch = np.ones((1, FIXED_FRAMES), np.int64)
    pitchf = np.zeros((1, FIXED_FRAMES), np.float32)
    rnd = np.zeros((1, 192, FIXED_FRAMES), np.float32)
    phone[:, :actual_frames, :] = 0.01
    pitch[:, :actual_frames] = 128
    pitchf[:, :actual_frames] = 120.0
    return {
        "phone": phone,
        "phone_lengths": np.asarray([actual_frames], np.int64),
        "pitch": pitch,
        "pitchf": pitchf,
        "sid": np.asarray([1], np.int64),
        "rnd": rnd,
    }


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

    t = FIXED_FRAMES
    inputs = (
        torch.zeros(1, t, 768, dtype=torch.float32),
        torch.tensor([t], dtype=torch.long),
        torch.full((1, t), 128, dtype=torch.long),
        torch.full((1, t), 120.0, dtype=torch.float32),
        torch.tensor([1], dtype=torch.long),
        torch.zeros(1, 192, t, dtype=torch.float32),
    )

    print(f"Exporting fixed {FIXED_FRAMES}-frame generic RVC synthesizer...")
    torch.onnx.export(
        wrapper,
        inputs,
        str(out),
        input_names=["phone", "phone_lengths", "pitch", "pitchf", "sid", "rnd"],
        output_names=["waveform"],
        opset_version=17,
        do_constant_folding=False,
        keep_initializers_as_inputs=True,
        dynamo=False,
    )

    graph = onnx.load(str(out))
    onnx.checker.check_model(graph)

    state = wrapper.state_dict()
    weights = []
    unmatched_initializers = []
    onnx_to_state = {}

    for init in graph.graph.initializer:
        name = init.name
        state_name = None
        checkpoint_name = None
        if name in state:
            state_name = name
            checkpoint_name = name.removeprefix("model.")
        elif f"model.{name}" in state:
            state_name = f"model.{name}"
            checkpoint_name = name
        if checkpoint_name is None:
            unmatched_initializers.append(name)
            continue
        tensor = state[state_name]
        onnx_to_state[name] = state_name
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
    if len(weights) != 457:
        raise RuntimeError(f"Expected 457 runtime RVC tensors, got {len(weights)}")

    manifest = {
        "format": 2,
        "version": "v2",
        "f0": True,
        "sampleRate": 40000,
        "speakerCount": 109,
        "fixedFrames": FIXED_FRAMES,
        "samplesPerFrame": SAMPLES_PER_FRAME,
        "normalInputs": ["phone", "phone_lengths", "pitch", "pitchf", "sid", "rnd"],
        "weights": weights,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Generic synth: {out} ({out.stat().st_size / 1024**2:.1f} MiB)")
    print(f"Manifest: {manifest_path}")

    import numpy as np
    import onnxruntime as ort

    base = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    for actual_frames in (FIXED_FRAMES, 173):
        result = base.run(["waveform"], make_inputs(np, actual_frames))[0]
        expected_samples = FIXED_FRAMES * SAMPLES_PER_FRAME
        if result.shape != (1, 1, expected_samples):
            raise RuntimeError(
                f"Fixed RVC graph failed: {result.shape}, expected (1,1,{expected_samples})"
            )
        if not np.isfinite(result).all():
            raise RuntimeError(f"Non-finite RVC output for actual_frames={actual_frames}")
        cropped = result[:, :, : actual_frames * SAMPLES_PER_FRAME]
        print(
            "ORT fixed smoke:", actual_frames, result.shape,
            "cropped", cropped.shape, "peak", float(np.max(np.abs(cropped)))
        )

    opts = ort.SessionOptions()
    keepalive = []
    for spec in weights:
        state_name = onnx_to_state[spec["onnxName"]]
        array = state[state_name].detach().cpu().numpy().astype(np.float32, copy=False)
        value = ort.OrtValue.ortvalue_from_numpy(array)
        keepalive.append((array, value))
        opts.add_initializer(spec["onnxName"], value)
    overridden = ort.InferenceSession(str(out), sess_options=opts, providers=["CPUExecutionProvider"])
    result = overridden.run(["waveform"], make_inputs(np, 251))[0]
    if result.shape != (1, 1, FIXED_FRAMES * SAMPLES_PER_FRAME) or not np.isfinite(result).all():
        raise RuntimeError(f"Runtime initializer override smoke failed: {result.shape}")
    print("ORT 457-initializer override smoke:", result.shape, float(np.max(np.abs(result))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
