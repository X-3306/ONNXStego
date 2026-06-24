# ONNX-Stego Public Proof

This directory contains a complete public demonstration of ONNX-Stego.

Hidden message:

```text
SECRET OF X-3306
```

Model:

- `squeezenet1.0-12`
- Source: <https://huggingface.co/onnxmodelzoo/squeezenet1.0-12>
- License: Apache-2.0, as inherited from ONNX Model Zoo

Files:

- `squeezenet1.0-12-reference.onnx`: public reference model
- `squeezenet1.0-12-cover.onnx`: deterministic fine-tune surrogate
- `squeezenet1.0-12-stego.onnx`: model containing the hidden proof message
- `manifest.json`: hashes, public proof key, threshold, and extraction command

(REMEMBER, `squeezenet1.0-12-cover.onnx` IS JUST DETERMINISTIC SIMULATION AS A PoC FOR PUBLICATION. DO NOT USE IT IN YOUR LAB AS A REAL EMBEDDING EXAMPLE.)

BEFORE EXTRACT, CHANGE THIS TO YOUR OWN PATH in manifest.json:

`input_path": "C:\\Users\\YOU\\YOUR\\REPO_PATH\\proof\\squeezenet1.0-12-cover.onnx",` 

`"output_path": "C:\\Users\\YOU\\YOUR\\REPO_PATH\\proof\\squeezenet1.0-12-stego.onnx",`

## Extract:

```powershell
onnx-stego extract `
  --model proof/squeezenet1.0-12-stego.onnx `
  --key onxs1_AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8 `
  --selection-mode natural `
  --reference-model proof/squeezenet1.0-12-reference.onnx `
  --natural-min-abs-delta 1e-5
```

The proof key is intentionally public. It is not a real secret👍.
