import argparse
import os
from pathlib import Path

import soundfile
import torch
from tqdm import tqdm

from onsets_and_frames.constants import (
    HOP_LENGTH,
    MEL_FMAX,
    MEL_FMIN,
    N_MELS,
    SAMPLE_RATE,
    WINDOW_LENGTH,
)
from onsets_and_frames.mel import MelSpectrogram


AUDIO_EXTENSIONS = {".flac", ".wav"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute audio features and save them as .pt files."
    )

    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to dataset root, for example data/MAESTRO or data/MAPS.",
    )

    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="Optional dataset group(s), for example train validation test.",
    )

    parser.add_argument(
        "--feature-type",
        default="mel",
        choices=["mel"],
        help="Feature type to compute. Currently only mel is supported.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing feature files.",
    )

    return parser.parse_args()


def find_audio_files(dataset_path: Path, groups):
    if groups:
        search_roots = [dataset_path / group for group in groups]
    else:
        search_roots = [dataset_path]

    audio_files = []

    for root in search_roots:
        if not root.exists():
            raise FileNotFoundError(f"Path does not exist: {root}")

        for path in root.rglob("*"):
            if path.suffix.lower() in AUDIO_EXTENSIONS:
                audio_files.append(path)

    return sorted(audio_files)


def feature_output_path(audio_path: Path, feature_type: str) -> Path:
    return audio_path.with_suffix(f".{feature_type}.pt")


def build_feature_extractor(feature_type: str):
    if feature_type == "mel":
        extractor = MelSpectrogram(
            N_MELS,
            SAMPLE_RATE,
            WINDOW_LENGTH,
            HOP_LENGTH,
            mel_fmin=MEL_FMIN,
            mel_fmax=MEL_FMAX,
        )
        extractor.eval()
        return extractor

    raise ValueError(f"Unsupported feature type: {feature_type}")


def load_audio(audio_path: Path) -> torch.Tensor:
    audio, sample_rate = soundfile.read(str(audio_path), dtype="int16")

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Unexpected sample rate for {audio_path}: "
            f"expected {SAMPLE_RATE}, got {sample_rate}"
        )

    audio = torch.ShortTensor(audio)

    if audio.ndim == 2:
        audio = audio.mean(dim=1).short()

    audio = audio.float().div_(32768.0)

    return audio


def compute_features(audio: torch.Tensor, feature_type: str, extractor) -> torch.Tensor:
    if feature_type == "mel":
        with torch.no_grad():
            # Matches dataset.py:
            # result["features"] = self.mel(result["audio"].unsqueeze(0)[:, :-1]).squeeze(0)
            return extractor(audio.unsqueeze(0)[:, :-1]).squeeze(0)

    raise ValueError(f"Unsupported feature type: {feature_type}")


def preprocess_file(audio_path: Path, feature_type: str, extractor, overwrite: bool) -> str:
    output_path = feature_output_path(audio_path, feature_type)

    if output_path.exists() and not overwrite:
        return "skipped"

    audio = load_audio(audio_path)
    features = compute_features(audio, feature_type, extractor)

    torch.save(features.cpu(), output_path)

    return "saved"


def main():
    args = parse_args()

    audio_files = find_audio_files(args.dataset_path, args.groups)

    if not audio_files:
        raise RuntimeError(f"No audio files found under {args.dataset_path}")

    extractor = build_feature_extractor(args.feature_type)

    saved = 0
    skipped = 0

    for audio_path in tqdm(audio_files, desc=f"Precomputing {args.feature_type}"):
        status = preprocess_file(
            audio_path=audio_path,
            feature_type=args.feature_type,
            extractor=extractor,
            overwrite=args.overwrite,
        )

        if status == "saved":
            saved += 1
        elif status == "skipped":
            skipped += 1

    print(f"Done. Saved: {saved}, skipped: {skipped}, total: {len(audio_files)}")


if __name__ == "__main__":
    main()