# ONNX-Stego

ONNX-Stego is a Python proof-of-concept for hiding short authenticated messages
inside `float32` ONNX model weights. It embeds bits in the least significant
mantissa bit of selected weights, while using modern authenticated encryption
for the hidden payload.

This project demonstrates that neural-network model files can act as covert carriers for authenticated messages, while preserving model validity and minimizing numerical impact. It provides a reproducible ONNX proof, cryptographic payload protection, and reference-based natural weight selection to emulate hiding inside fine-tuning noise.

The project is built for defensive research, watermarking experiments,
provenance notes, and steganography analysis on neural-network artifacts.

This is a Side-Project created mainly because of https://github.com/X-3306/Project-Onyx

## What Makes It Different

- **The carrier is a real ML artifact**: the payload is embedded in ordinary
  `.onnx` weights, so the resulting file remains a valid model rather than a
  custom container disguised as one.
- **Cryptography is separated from steganography**: the hidden bytes are
  protected with ChaCha20-Poly1305 before embedding. The model hides where the
  ciphertext is stored; the AEAD layer protects the message itself.
- **Positions are keyed and reproducible**: sender and receiver derive the same
  embedding positions from the master key using a ChaCha20-backed deterministic
  CSPRNG with unbiased sampling. (Header positions are static per master key, payload positions are re-rerived per message from the master key and a fresh per-message salt)
- **Reference-aware embedding is supported**: natural selection can restrict
  writes to weights that already differ from a reference model, matching the
  practical "hide inside fine-tuning changes" threat model.
- **ONNX storage details are handled explicitly**: the implementation supports
  both `raw_data` and `float_data`, and normalizes weight bytes as little-endian
  `float32` before touching mantissa bits.
- **The proof is measurable**: tests and scripts check extraction, wrong-key rejection, tamper detection, ONNX validity, simple LSB statistics, and inference drift.
  
READ `SECURITY.md` for better understanding implementation, security and known-limitations.

## Current Capabilities

- Generate a 256-bit master key.
- Inspect the available `float32` LSB capacity of an ONNX model.
- Embed text messages into ONNX `float32` weights.
- Extract messages with the same master key.
- Detect wrong keys or modified payload bits through AEAD authentication.
- Use either:
  - `uniform` selection: keyed positions across all float32 weights.
  - `natural` selection: keyed positions only inside weights whose absolute
    delta from a reference model is above a configured threshold.
- Sanitize common ONNX metadata fields before writing the stego model.

## Recommended Real-World Flow

ONNX-Stego is strongest when used with a legitimate model update workflow.

> **Note:** `squeezenet1.0-12-cover.onnx` is a deterministic fine-tune 
> surrogate for reproducibility only. Do not use it as a real embedding 
> example in operational scenarios.

### What Sender A Does

1. Choose a public reference model, for example a model from ONNX Model Zoo or a
   model exported from PyTorch/TensorFlow to ONNX.
2. Fine-tune that model on a normal task. This creates a **cover model** whose
   weights naturally differ from the public reference.
3. Share, out of band, with Receiver B:
   - the exact reference model identity or file hash,
   - the 256-bit master key,
   - the `natural-min-abs-delta` threshold,
   - the ONNX-Stego version or commit.
4. Embed the message into the fine-tuned cover model using natural selection:

```powershell
onnx-stego embed `
  --model cover-finetuned.onnx `
  --output stego.onnx `
  --key "<MASTER_KEY>" `
  --message "short authenticated message" `
  --selection-mode natural `
  --reference-model reference.onnx `
  --natural-min-abs-delta 1e-5
```

5. Send only `stego.onnx` through the ordinary channel.

### What Receiver B Needs

B needs:

- `stego.onnx`
- the same 256-bit master key
- the same reference model
- the same natural-selection threshold
- this tool

Extraction:

```powershell
onnx-stego extract `
  --model stego.onnx `
  --key "<MASTER_KEY>" `
  --selection-mode natural `
  --reference-model reference.onnx `
  --natural-min-abs-delta 1e-5
```

If the key is wrong, the reference model differs, the threshold differs, or the
payload bits were modified, extraction fails authentication.

### Why Fine-Tuning Matters

Uniform LSB embedding works technically, but it modifies positions across the
whole model without explaining why those positions should have changed. In a
better operational setup, A first fine-tunes the model for a plausible task.
Fine-tuning changes many weights for a legitimate reason. Natural selection then
restricts hidden bits to weights that already changed relative to the reference
model, making the stego edits hide inside an existing model-update story.

This project does not perform fine-tuning for you. It expects A (You) to bring a
reference (base) model and a fine-tuned cover model.

## Installation

```powershell
python -m pip install -e .
```

For development and empirical verification:

```powershell
python -m pip install -e ".[dev]"
```

## Quick Start

Generate a key:

```powershell
onnx-stego keygen
```

Inspect a model:

```powershell
onnx-stego inspect model.onnx
```

Embed a message with uniform selection:

```powershell
onnx-stego embed `
  --model model.onnx `
  --output stego.onnx `
  --key "<MASTER_KEY>" `
  --message "hello from inside the weights"
```

Extract it:

```powershell
onnx-stego extract --model stego.onnx --key "<MASTER_KEY>"
```

## Publication Proof

The `proof/` directory contains a complete public demonstration.

Model used:

- `squeezenet1.0-12` from ONNX Model Zoo on Hugging Face
- Source: <https://huggingface.co/onnxmodelzoo/squeezenet1.0-12>
- License: Apache-2.0, as inherited from ONNX Model Zoo
- Reason for my choice: it is public, compact enough for a repository, works with
  ONNX tooling, small enough to upload on github and has about 1.24 million `float32` weights, which keeps the example embedding density low.

Proof files:

- `proof/squeezenet1.0-12-reference.onnx`: downloaded public reference model
- `proof/squeezenet1.0-12-cover.onnx`: deterministic fine-tune surrogate used
  only for reproducible publication proof
- `proof/squeezenet1.0-12-stego.onnx`: final model containing the hidden message
- `proof/manifest.json`: hashes, key, threshold, and extraction command

The embedded public demo message is:

```text
SECRET OF X-3306
```

Extract it:

```powershell
onnx-stego extract `
  --model proof/squeezenet1.0-12-stego.onnx `
  --key onxs1_AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8 `
  --selection-mode natural `
  --reference-model proof/squeezenet1.0-12-reference.onnx `
  --natural-min-abs-delta 1e-5
```

Rebuild the proof:

```powershell
python scripts/build_proof.py
```

The included proof uses a deterministic fine-tune surrogate so the repository
can be reproduced without a dataset or GPU. In real use, replace that surrogate
with ACTUAL fine-tuning.

## Validation

Run the test suite:

```powershell
python -m pytest
```

Run the smoke test:

```powershell
python scripts/empirical_demo.py
```

Checks cover:

- message round-trip
- wrong-key rejection
- tamper rejection
- ONNX checker validity
- conversion from `float_data` to `raw_data`
- little-endian `float32` handling
- simple LSB ratio and chi-square statistics
- inference output drift on a synthetic ONNX model
- natural selection against a reference model

## What Observer C Can Check, Known Limitations and Key Considerations
is in `SECURITY.md`.

## Responsible Use

Use this project for legitimate research, watermarking, provenance experiments,
and defensive analysis. Do not use it to bypass monitoring, policy, law, or the
consent of systems and people that handle the model files.

## Development

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python scripts/empirical_demo.py
python scripts/build_proof.py
```

## License

Code in this repository is released under the MIT License. The included proof
model is derived from ONNX Model Zoo's SqueezeNet model and is documented in
`proof/manifest.json` as Apache-2.0 sourced material.
