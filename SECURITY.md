- Payload encryption uses ChaCha20-Poly1305 AEAD.

- Positions come from a ChaCha20-backed deterministic CSPRNG with domain separation.

- HKDF-SHA256 with domain separation

- `float_data` tensors are converted to little-endian `raw_data` before bit
manipulation.

- AAD binds the payload to a header: PAYLOAD_AAD = b"onnx-stego:v1:payload:" + header_plaintext (this protects against cross-session cut-and-paste attacks)

- Random nonces (12 B) + salt (16 B) per message (96-bit nonce space, birthday bound ~2⁴⁸ uses)

- Natural selection can restrict embedding to weights that already changed
relative to a reference model.

### Why Fine-Tuning Matters

Uniform LSB embedding works technically, but it modifies positions across the
whole model without explaining why those positions should have changed. In a
better operational setup, A first fine-tunes the model for a plausible task.
Fine-tuning changes many weights for a legitimate reason. Natural selection then
restricts hidden bits to weights that already changed relative to the reference
model, making the stego edits hide inside an existing model-update 'story'.

This project does not perform fine-tuning for you. It expects A (You) to bring a
reference (base) model and a fine-tuned cover model.

## What Observer C Can Check

A passive observer without the key cannot extract the message. C can still run
steganalysis-style checks:

- validate the ONNX model with `onnx.checker`
- inspect metadata fields
- compare the model to a known reference, if available (that's why fine-tuning is important, fine-tuning provides statistical cover: LSB changes are indistinguishable from fine-tuning noise in changed weights)
- inspect LSB one/zero ratios and chi-square statistics (note: CSPRNG-based position selection significantly weakens sequential-embedding attacks like the classic chi-square test)
- compare weight-delta distributions against normally fine-tuned models
- run inference drift checks
- test whether quantization or model conversion destroys hidden data

The important caveat: empirical tests can show that this implementation works
and that no obvious anomaly appears in the tested setup. They cannot prove
universal undetectability against every model, dataset, fine-tuning process, or
adversary.

## Known Limitations and Key Considerations

- Only `float32` ONNX initializers are supported.
- External ONNX tensor data is not supported.
- Quantization, pruning, lossy conversion, or weight noise can destroy the
  hidden payload.
- Natural selection requires B to have the same reference model and threshold.
- The public proof cover model is a deterministic fine-tune surrogate, not a
  task-trained SqueezeNet checkpoint.
- This is mainly research software. Do not treat it as a formal steganographic proof.
- A defender (C) with access to multiple model versions or historical 
  checkpoints can perform delta-distribution analysis and detect 
  statistical fingerprints in LSB patterns across weight updates.
- static per master key
- Embedding capacity is bounded by the number of changed weights 
  relative to the reference model and the chosen LSB depth.
- Capacity depends on model size: 1 bit per float32 weight, so a 
  1M-weight model holds up to ~125 KB (less with natural selection).

## Natural Selection

Natural selection requires two models:

- a public reference model
- a cover model, usually produced by fine-tuning the reference

ONNX-Stego computes `abs(cover_weight - reference_weight)` for each `float32`
weight and keeps only positions above a configured threshold. Header and payload
positions are then sampled from that candidate pool with the normal keyed
sampler.

This means Receiver B must have the same reference model and threshold. If those
differ, authentication fails because B reads the wrong candidate positions.

## Limits Of Empirical Claims

The test suite and proof demonstrate correctness, authentication failure on
wrong keys, small numerical deltas, and no obvious anomaly in simple LSB tests.
They do not prove universal undetectability tho.

## Responsible Use

Use this project for legitimate research, watermarking, provenance experiments,
and defensive analysis. Do not use it to bypass monitoring, policy, law, or the
consent of systems and people that handle the model files.

## License

Code in this repository is released under the MIT License. The included proof
model is derived from ONNX Model Zoo's SqueezeNet model and is documented in
`proof/manifest.json` as Apache-2.0 sourced material.
