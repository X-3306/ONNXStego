"""
REMEMBER, THIS IS JUST DETERMINISTIC SIMULATION AS A PoC FOR THIS PUBLICATION. 
DO NOT USE IT AS A REAL EMBEDDING EXAMPLE. 

This script builds a proof of concept for the onnx-stego publication. 
It downloads a reference model, builds a cover model by fine-tuning it 
with a deterministic surrogate, embeds a secret message, and verifies that 
the message can be extracted. It also generates a manifest with relevant information 
about the proof, including the SHA256 hashes of the reference, cover, and stego models, 
as well as the command to extract the message from the stego model. The proof is designed 
to be reproducible and verifiable by others, but it is not intended for real-world use. 
The embedding process is done using the natural selection mode with a specified minimum 
absolute delta, which ensures that only weights that have changed significantly from the 
reference model are considered for embedding. The script also includes a function to compute 
the SHA256 hash of a file, which is used to verify the integrity of the models. Overall, this
script serves as a demonstration of the capabilities of the onnx-stego library and provides 
a starting point for further research and development in the field of model steganography.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import onnx  # noqa: E402

from onnxstego.core import embed_message, extract_message  # noqa: E402
from onnxstego.crypto import parse_master_key  # noqa: E402
from onnxstego.onnx_io import Float32WeightStore  # noqa: E402
from onnxstego.selection import NaturalWeightSelector, SelectionMode  # noqa: E402

MODEL_URL = (
    "https://huggingface.co/onnxmodelzoo/squeezenet1.0-12/resolve/main/"
    "squeezenet1.0-12.onnx"
)
MODEL_SOURCE = "https://huggingface.co/onnxmodelzoo/squeezenet1.0-12"
MODEL_LICENSE = "Apache-2.0"
PROOF_MESSAGE = b"SECRET OF X-3306"
PROOF_KEY = "onxs1_AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8"
NATURAL_MIN_ABS_DELTA = 1e-5


def main() -> int:
    proof_dir = ROOT / "proof"
    proof_dir.mkdir(exist_ok=True)

    reference_path = proof_dir / "squeezenet1.0-12-reference.onnx"
    cover_path = proof_dir / "squeezenet1.0-12-cover.onnx"
    stego_path = proof_dir / "squeezenet1.0-12-stego.onnx"
    manifest_path = proof_dir / "manifest.json"

    urllib.request.urlretrieve(MODEL_URL, reference_path)
    build_cover_model(reference_path, cover_path)

    key = parse_master_key(PROOF_KEY)
    report = embed_message(
        cover_path,
        stego_path,
        key,
        PROOF_MESSAGE,
        reference_model_path=reference_path,
        selection_mode=SelectionMode.NATURAL,
        natural_min_abs_delta=NATURAL_MIN_ABS_DELTA,
    )
    recovered = extract_message(
        stego_path,
        key,
        reference_model_path=reference_path,
        selection_mode=SelectionMode.NATURAL,
        natural_min_abs_delta=NATURAL_MIN_ABS_DELTA,
    )
    if recovered != PROOF_MESSAGE:
        raise RuntimeError("proof round-trip failed")

    candidates = NaturalWeightSelector.from_models(
        cover_path,
        reference_path,
        min_abs_delta=NATURAL_MIN_ABS_DELTA,
    ).count

    manifest = {
        "message": PROOF_MESSAGE.decode("ascii"),
        "proof_key": PROOF_KEY,
        "model_source": MODEL_SOURCE,
        "model_license": MODEL_LICENSE,
        "selection_mode": SelectionMode.NATURAL.value,
        "natural_min_abs_delta": NATURAL_MIN_ABS_DELTA,
        "candidate_weights": candidates,
        "embed_report": report.to_dict(),
        "files": {
            reference_path.name: sha256_file(reference_path),
            cover_path.name: sha256_file(cover_path),
            stego_path.name: sha256_file(stego_path),
        },
        "extract_command": (
            "onnx-stego extract --model proof/squeezenet1.0-12-stego.onnx "
            f"--key {PROOF_KEY} --selection-mode natural "
            "--reference-model proof/squeezenet1.0-12-reference.onnx "
            f"--natural-min-abs-delta {NATURAL_MIN_ABS_DELTA}"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


def build_cover_model(reference_path: Path, cover_path: Path) -> None:
    model = onnx.load(reference_path)
    store = Float32WeightStore.from_model(model)
    weights = store.to_float_array()

    # Deterministic fine-tune surrogate for publication proof:
    # small, structured deltas create a cover model with many already-changed
    # weights. Real deployments should replace this with actual task fine-tuning.
    indices = np.arange(weights.size, dtype=np.float64)
    signs = np.where((indices.astype(np.int64) % 2) == 0, 1.0, -1.0)
    deltas = signs * (1.0e-4 + (indices % 17) * 1.0e-7)
    tuned = (weights.astype(np.float64) + deltas).astype("<f4")

    cursor = 0
    for segment in store.segments:
        count = segment.count
        segment.raw[:] = tuned[cursor : cursor + count].astype("<f4").tobytes()
        cursor += count
    store.flush()
    onnx.checker.check_model(model)
    onnx.save(model, cover_path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
