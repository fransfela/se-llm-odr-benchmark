"""Pipeline integrity checks for data/enhanced/ WAV files."""

import pytest
import numpy as np
import soundfile as sf
from pathlib import Path

ROOT = Path(__file__).parent.parent
ENHANCED_DIR = ROOT / "data" / "enhanced"


def _all_wavs():
    return list(ENHANCED_DIR.rglob("*.wav"))


def test_sample_rate_consistency():
    wavs = _all_wavs()
    if not wavs:
        pytest.skip("data/enhanced/ contains no WAV files yet")

    failures = []
    for path in wavs:
        sr = sf.info(path).samplerate
        if sr != 16000:
            print(f"FAIL sample rate: {path} — {sr}Hz")
            failures.append(path)

    assert not failures, f"{len(failures)} file(s) not at 16kHz: {failures}"


def test_no_clipping():
    wavs = _all_wavs()
    if not wavs:
        pytest.skip("data/enhanced/ contains no WAV files yet")

    failures = []
    for path in wavs:
        audio, _ = sf.read(path, dtype="float32", always_2d=False)
        peak = float(np.max(np.abs(audio)))
        if peak > 1.0:
            print(f"FAIL clipping: {path} — peak={peak:.6f}")
            failures.append((path, peak))

    assert not failures, f"{len(failures)} file(s) clipped: {failures}"


def test_duration_match():
    if not ENHANCED_DIR.exists():
        pytest.skip("data/enhanced/ does not exist yet")

    # Collect {clip_id: {condition: duration_s}}
    durations: dict[str, dict[str, float]] = {}
    for path in _all_wavs():
        condition = path.parent.name
        clip_id = path.stem
        info = sf.info(path)
        dur = info.frames / info.samplerate
        durations.setdefault(clip_id, {})[condition] = dur

    if not durations:
        pytest.skip("data/enhanced/ contains no WAV files yet")

    TOLERANCE_S = 0.1
    failures = []
    for clip_id, cond_map in durations.items():
        if "clean" not in cond_map:
            continue
        ref_dur = cond_map["clean"]
        for condition, dur in cond_map.items():
            if condition == "clean":
                continue
            delta = abs(dur - ref_dur)
            if delta > TOLERANCE_S:
                print(
                    f"FAIL duration: clip={clip_id} condition={condition} "
                    f"delta={delta:.3f}s (clean={ref_dur:.3f}s, this={dur:.3f}s)"
                )
                failures.append((clip_id, condition, delta))

    assert not failures, f"{len(failures)} duration mismatch(es): {failures}"
