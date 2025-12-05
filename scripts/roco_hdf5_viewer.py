#!/usr/bin/env python3
"""Small helper to download, inspect, and export rocochallenge2025 HDF5 logs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    import h5py  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("h5py is required. pip install h5py") from exc

try:
    from huggingface_hub import hf_hub_download
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "huggingface_hub is required. pip install huggingface_hub"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and export rocochallenge2025 HDF5 recordings."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help=(
            "HDF5 paths. If not found locally, they are pulled from the HF dataset "
            "repo (e.g., gearbox_assembly_demos/data_20251127_212217.hdf5)."
        ),
    )
    parser.add_argument(
        "--repo",
        default="rocochallenge2025/rocochallenge2025",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--local-dir",
        default="data/cache",
        help="Local cache directory for downloads.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list dataset keys (no export).",
    )
    parser.add_argument(
        "--rgb-key",
        help="Dataset key to use for RGB export (auto-detected if omitted).",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="How many seconds to export.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="FPS for MP4 export.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Hard cap on exported frames (overrides seconds*fps if smaller).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=5,
        help="Save this many PNG frames alongside the video (0 to skip).",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Do not write MP4 (still saves PNG frames if --frames > 0).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/exports",
        help="Where to place exported media.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce log verbosity.",
    )
    return parser.parse_args()


def download_if_needed(repo: str, fname: str, local_dir: str) -> Path:
    """Download from HF repo if file is not present locally."""
    candidate = Path(fname)
    if candidate.exists():
        return candidate

    path = hf_hub_download(
        repo_id=repo,
        filename=fname,
        local_dir=local_dir,
        repo_type="dataset",
    )
    return Path(path)


def list_datasets(handle: h5py.File) -> List[Tuple[str, Sequence[int], str]]:
    rows: List[Tuple[str, Sequence[int], str]] = []

    def _visitor(name: str, obj: h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            rows.append((name, obj.shape, str(obj.dtype)))

    handle.visititems(_visitor)
    return sorted(rows, key=lambda x: x[0])


def tag_for(name: str) -> str:
    lname = name.lower()
    tags: List[str] = []
    if "rgb" in lname or "color" in lname:
        tags.append("rgb?")
    if "depth" in lname:
        tags.append("depth?")
    if "action" in lname:
        tags.append("action?")
    if "joint" in lname or "jnt" in lname:
        tags.append("joint?")
    if "state" in lname or "obs" in lname:
        tags.append("state?")
    if "ee" in lname:
        tags.append("ee?")
    return f" [{' '.join(tags)}]" if tags else ""


def print_structure(rows: Iterable[Tuple[str, Sequence[int], str]]) -> None:
    print("=== Dataset keys ===")
    for name, shape, dtype in rows:
        print(f"- {name} | {shape} | {dtype}{tag_for(name)}")


def pick_rgb_key(rows: Iterable[Tuple[str, Sequence[int], str]], override: str | None) -> str | None:
    if override:
        return override
    candidates: List[str] = []
    for name, shape, _ in rows:
        if len(shape) == 4 and "rgb" in name.lower():
            candidates.append(name)
    return candidates[0] if candidates else None


def normalize_to_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype == np.uint8:
        out = arr
    else:
        arr = arr.astype("float32")
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        out = np.clip(arr * 255.0, 0, 255).astype("uint8")

    if out.shape[-1] == 4:
        out = out[..., :3]
    if out.shape[-1] == 1:
        out = np.repeat(out, 3, axis=-1)
    return out


def export_media(
    data: np.ndarray,
    dest_dir: Path,
    base_name: str,
    fps: int,
    seconds: float,
    max_frames: int | None,
    save_frames: int,
    skip_video: bool,
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    total = data.shape[0]
    target = min(int(seconds * fps), total)
    if max_frames is not None:
        target = min(target, max_frames)

    if target <= 0:
        print("No frames to export.")
        return

    frames = [normalize_to_uint8(data[i]) for i in range(target)]

    if not skip_video:
        if cv2 is None:
            print("OpenCV not available; skipping video export (pip install opencv-python).")
        else:
            h, w, _ = frames[0].shape
            mp4_path = dest_dir / f"{base_name}.mp4"
            writer = cv2.VideoWriter(
                str(mp4_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h),
            )
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"Wrote video: {mp4_path}")

    if save_frames > 0:
        try:
            import imageio.v2 as imageio  # type: ignore
        except Exception:
            print("imageio not available; skipping PNG frames (pip install imageio).")
            return
        for idx in range(min(save_frames, len(frames))):
            png_path = dest_dir / f"{base_name}_frame{idx:04d}.png"
            imageio.imwrite(png_path, frames[idx])
        print(f"Saved {min(save_frames, len(frames))} PNG frame(s) to {dest_dir}")


def process_file(
    repo: str,
    local_dir: str,
    fname: str,
    args: argparse.Namespace,
) -> None:
    path = download_if_needed(repo, fname, local_dir)
    if not args.quiet:
        print(f"\n=== Processing {path} ===")

    with h5py.File(path, "r") as handle:
        rows = list_datasets(handle)
        print_structure(rows)
        if args.list_only:
            return

        rgb_key = pick_rgb_key(rows, args.rgb_key)
        if rgb_key is None:
            print("No RGB-like dataset found. Use --rgb-key to select one manually.")
            return

        data = handle[rgb_key]
        base = Path(path).stem + "_" + rgb_key.replace("/", "_")
        dest = Path(args.output_dir) / base
        export_media(
            data=data,
            dest_dir=dest,
            base_name="preview",
            fps=args.fps,
            seconds=args.seconds,
            max_frames=args.max_frames,
            save_frames=args.frames,
            skip_video=args.skip_video,
        )


def main() -> None:
    args = parse_args()
    for fname in args.files:
        process_file(args.repo, args.local_dir, fname, args)


if __name__ == "__main__":
    main()
