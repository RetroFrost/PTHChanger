# PTHChanger

Offline Android RVC voice conversion focused on RVC v2 F0 `.pth` checkpoints and Android Vulkan/Mali GPUs.

## Current app flow

1. Select an RVC v2 `.pth` model.
2. Select an audio file supported by Android MediaCodec (WAV, MP3, M4A/AAC and device-supported formats).
3. Choose pitch and speaker ID.
4. Convert offline.
5. Play, save or share the WAV output.

The app parses PyTorch ZIP/pickle checkpoints on-device with a restricted checkpoint reader; it does not execute Python or arbitrary pickle globals.

## Runtime pack

The source compiles without committing the large shared neural runtime binaries. Actual conversion requires these APK assets:

- `app/src/main/assets/runtime/contentvec_v2.onnx`
- `app/src/main/assets/runtime/rmvpe.onnx`
- `app/src/main/assets/runtime/rvc_v2_f0_40k_s109_vulkan.pte`
- `app/src/main/assets/runtime/rvc_weights_manifest.json`

The `.pte` is voice-neutral: the selected `.pth` supplies the RVC synthesizer weights at runtime. Users never need to select or convert to `.pte` themselves.

## Build

GitHub Actions builds a debug APK on every push to `main` and exposes it as the `PTHChanger-debug` artifact.
