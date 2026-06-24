from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import onnx

from onnxstego.bits import bits_to_bytes, bytes_to_bits
from onnxstego.crypto import (
    HEADER_ENVELOPE_SIZE,
    AuthenticationError,
    decrypt_header,
    decrypt_message,
    derive_payload_keys,
    derive_static_keys,
    encrypt_message,
)
from onnxstego.onnx_io import Float32WeightStore, ModelCapacity, inspect_capacity, sanitize_metadata
from onnxstego.positions import sample_unique
from onnxstego.selection import NaturalWeightSelector, SelectionMode

HEADER_POSITION_DOMAIN = b"header-v1"
PAYLOAD_POSITION_DOMAIN = b"payload-v1"


class CapacityError(Exception):
    """Raised when a model has too few float32 weights for a message."""


@dataclass(frozen=True)
class EmbedReport:
    input_path: str
    output_path: str
    capacity_bits: int
    required_bits: int
    header_bits: int
    payload_bits: int
    changed_weights: int
    embedding_density: float
    selection_mode: str
    candidate_weights: int

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def inspect_model(model_path: str | Path) -> ModelCapacity:
    return inspect_capacity(model_path)


def embed_message(
    model_path: str | Path,
    output_path: str | Path,
    master_key: bytes,
    message: bytes | str,
    *,
    sanitize: bool = True,
    reference_model_path: str | Path | None = None,
    selection_mode: SelectionMode | str = SelectionMode.UNIFORM,
    natural_min_abs_delta: float = 1e-6,
) -> EmbedReport:
    message_bytes = message.encode("utf-8") if isinstance(message, str) else message
    mode = SelectionMode(selection_mode)
    model = onnx.load(str(model_path))
    store = Float32WeightStore.from_model(model)
    candidates = _candidate_positions(
        model_path,
        reference_model_path,
        mode,
        natural_min_abs_delta=natural_min_abs_delta,
        total_weights=store.total_weights,
    )
    envelope = encrypt_message(master_key, message_bytes)

    header_bits = bytes_to_bits(envelope.header_bytes)
    payload_bits = bytes_to_bits(envelope.payload_ciphertext)
    required_bits = len(header_bits) + len(payload_bits)
    if required_bits > len(candidates):
        raise CapacityError(
            f"candidate capacity is {len(candidates)} bits, but message requires {required_bits} bits"
        )

    static_keys = derive_static_keys(master_key)
    header_candidate_offsets = sample_unique(
        len(candidates),
        len(header_bits),
        static_keys.header_position_key,
        HEADER_POSITION_DOMAIN,
    )
    header_positions = [candidates[offset] for offset in header_candidate_offsets]
    payload_keys = derive_payload_keys(master_key, envelope.header.salt)
    payload_candidate_offsets = sample_unique(
        len(candidates),
        len(payload_bits),
        payload_keys.position_key,
        PAYLOAD_POSITION_DOMAIN,
        exclude=header_candidate_offsets,
    )
    payload_positions = [candidates[offset] for offset in payload_candidate_offsets]

    changed_weights = 0
    for position, bit in zip(header_positions, header_bits, strict=True):
        changed_weights += int(store.set_lsb(position, bit))
    for position, bit in zip(payload_positions, payload_bits, strict=True):
        changed_weights += int(store.set_lsb(position, bit))

    store.flush()
    if sanitize:
        sanitize_metadata(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))

    return EmbedReport(
        input_path=str(model_path),
        output_path=str(output_path),
        capacity_bits=store.total_weights,
        required_bits=required_bits,
        header_bits=len(header_bits),
        payload_bits=len(payload_bits),
        changed_weights=changed_weights,
        embedding_density=required_bits / store.total_weights if store.total_weights else 0.0,
        selection_mode=mode.value,
        candidate_weights=len(candidates),
    )


def extract_message(
    model_path: str | Path,
    master_key: bytes,
    *,
    reference_model_path: str | Path | None = None,
    selection_mode: SelectionMode | str = SelectionMode.UNIFORM,
    natural_min_abs_delta: float = 1e-6,
) -> bytes:
    mode = SelectionMode(selection_mode)
    model = onnx.load(str(model_path))
    store = Float32WeightStore.from_model(model)
    candidates = _candidate_positions(
        model_path,
        reference_model_path,
        mode,
        natural_min_abs_delta=natural_min_abs_delta,
        total_weights=store.total_weights,
    )
    header_bit_count = HEADER_ENVELOPE_SIZE * 8
    if len(candidates) < header_bit_count:
        raise CapacityError(
            f"candidate capacity is {len(candidates)} bits, but header requires "
            f"{header_bit_count} bits"
        )

    static_keys = derive_static_keys(master_key)
    header_candidate_offsets = sample_unique(
        len(candidates),
        header_bit_count,
        static_keys.header_position_key,
        HEADER_POSITION_DOMAIN,
    )
    header_positions = [candidates[offset] for offset in header_candidate_offsets]
    header_bytes = bits_to_bytes(store.get_lsb(position) for position in header_positions)
    header = decrypt_header(master_key, header_bytes)

    payload_bit_count = header.payload_ciphertext_len * 8
    if payload_bit_count > len(candidates) - header_bit_count:
        raise AuthenticationError("authenticated payload length exceeds model capacity")

    payload_keys = derive_payload_keys(master_key, header.salt)
    payload_candidate_offsets = sample_unique(
        len(candidates),
        payload_bit_count,
        payload_keys.position_key,
        PAYLOAD_POSITION_DOMAIN,
        exclude=header_candidate_offsets,
    )
    payload_positions = [candidates[offset] for offset in payload_candidate_offsets]
    payload_ciphertext = bits_to_bytes(store.get_lsb(position) for position in payload_positions)
    return decrypt_message(master_key, header_bytes, payload_ciphertext)


def _candidate_positions(
    model_path: str | Path,
    reference_model_path: str | Path | None,
    mode: SelectionMode,
    *,
    natural_min_abs_delta: float,
    total_weights: int,
) -> list[int]:
    if mode == SelectionMode.UNIFORM:
        return list(range(total_weights))
    if reference_model_path is None:
        raise ValueError("natural selection requires --reference-model")
    selector = NaturalWeightSelector.from_models(
        model_path,
        reference_model_path,
        min_abs_delta=natural_min_abs_delta,
    )
    return selector.candidate_indices()
