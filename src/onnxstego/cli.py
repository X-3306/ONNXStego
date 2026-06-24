from __future__ import annotations

import argparse
import sys
from pathlib import Path

from onnxstego.core import CapacityError, embed_message, extract_message, inspect_model
from onnxstego.crypto import AuthenticationError, format_master_key, generate_master_key, parse_master_key
from onnxstego.selection import SelectionMode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="onnx-stego")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("keygen", help="generate a 256-bit master key")

    inspect_parser = subparsers.add_parser("inspect", help="show float32 LSB capacity")
    inspect_parser.add_argument("model", type=Path)

    embed_parser = subparsers.add_parser("embed", help="embed a message into an ONNX model")
    embed_parser.add_argument("--model", required=True, type=Path)
    embed_parser.add_argument("--output", required=True, type=Path)
    embed_parser.add_argument("--key", required=True)
    embed_parser.add_argument(
        "--selection-mode",
        choices=[mode.value for mode in SelectionMode],
        default=SelectionMode.UNIFORM.value,
    )
    embed_parser.add_argument("--reference-model", type=Path)
    embed_parser.add_argument("--natural-min-abs-delta", type=float, default=1e-6)
    message_group = embed_parser.add_mutually_exclusive_group(required=True)
    message_group.add_argument("--message")
    message_group.add_argument("--input-file", type=Path)
    embed_parser.add_argument("--no-sanitize", action="store_true")

    extract_parser = subparsers.add_parser("extract", help="extract a message from an ONNX model")
    extract_parser.add_argument("--model", required=True, type=Path)
    extract_parser.add_argument("--key", required=True)
    extract_parser.add_argument(
        "--selection-mode",
        choices=[mode.value for mode in SelectionMode],
        default=SelectionMode.UNIFORM.value,
    )
    extract_parser.add_argument("--reference-model", type=Path)
    extract_parser.add_argument("--natural-min-abs-delta", type=float, default=1e-6)
    extract_parser.add_argument("--output-file", type=Path)

    args = parser.parse_args(argv)

    try:
        if args.command == "keygen":
            print(format_master_key(generate_master_key()))
            return 0
        if args.command == "inspect":
            capacity = inspect_model(args.model)
            print(f"float32 weights: {capacity.float32_weights}")
            print(f"capacity bits:   {capacity.capacity_bits}")
            print(f"capacity bytes:  {capacity.capacity_bytes}")
            print(f"float32 tensors: {capacity.tensors}")
            return 0
        if args.command == "embed":
            key = parse_master_key(args.key)
            message = args.input_file.read_bytes() if args.input_file else args.message.encode("utf-8")
            report = embed_message(
                args.model,
                args.output,
                key,
                message,
                sanitize=not args.no_sanitize,
                reference_model_path=args.reference_model,
                selection_mode=args.selection_mode,
                natural_min_abs_delta=args.natural_min_abs_delta,
            )
            print(f"written: {report.output_path}")
            print(f"required bits: {report.required_bits}")
            print(f"changed weights: {report.changed_weights}")
            print(f"embedding density: {report.embedding_density:.6%}")
            print(f"selection mode: {report.selection_mode}")
            print(f"candidate weights: {report.candidate_weights}")
            return 0
        if args.command == "extract":
            key = parse_master_key(args.key)
            message = extract_message(
                args.model,
                key,
                reference_model_path=args.reference_model,
                selection_mode=args.selection_mode,
                natural_min_abs_delta=args.natural_min_abs_delta,
            )
            if args.output_file:
                args.output_file.write_bytes(message)
            else:
                sys.stdout.buffer.write(message)
                sys.stdout.buffer.write(b"\n")
            return 0
    except (AuthenticationError, CapacityError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.error("unknown command")
    return 2
