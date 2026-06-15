"""Compute all 22 audio quality metrics for every DNS blind test clip, joined with P.808 MOS."""

import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

from src.metrics.run_all_metrics import (
    _compute_distortion,
    _compute_perceptual,
    _compute_non_intrusive,
    _compute_special,
    _load,
    ERROR_LOG,
    MODEL_VERSIONS,
)
from src.data.dns_utils import load_dns_mos_labels, load_dns_clip_pairs, get_dns_condition_list

ROOT        = Path(__file__).parent.parent.parent
METRICS_CSV = ROOT / "results" / "metrics_dns.csv"
DNS_DIR     = ROOT / "data" / "raw" / "dns_blind_test"

FIELDNAMES = [
    "clip_id", "condition_name", "mos_source", "sig_mos", "bak_mos", "ovrl_mos",
    "snr", "si_snr", "si_sdr", "sdr",
    "pesq", "stoi", "visqol", "warpq",
    "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "nisqa", "noresqa", "noresqa_mos",
    "scoreq_nr", "nomad", "squim_mos", "squim_stoi", "squim_pesq", "squim_si_sdr",
    "srmr", "emotion_sim",
]


def compute_dns_metrics(
    clip_id: str,
    condition_name: str,
    enhanced_path: str | Path,
    clean_path: str | Path,
    mos_entry: dict | None,
) -> dict:
    """Compute all 22 metrics for one DNS clip; join P.808 MOS if available."""
    enh_path = Path(enhanced_path)
    ref_path = Path(clean_path)

    enh = _load(enh_path)
    ref = _load(ref_path)
    n   = min(len(enh), len(ref))
    enh, ref = enh[:n], ref[:n]

    row = {"clip_id": clip_id, "condition_name": condition_name}

    if mos_entry:
        row["mos_source"] = mos_entry.get("mos_source", "")
        row["sig_mos"]    = mos_entry.get("sig",  float("nan"))
        row["bak_mos"]    = mos_entry.get("bak",  float("nan"))
        row["ovrl_mos"]   = mos_entry.get("ovrl", float("nan"))
    else:
        row["mos_source"] = ""
        row["sig_mos"]    = float("nan")
        row["bak_mos"]    = float("nan")
        row["ovrl_mos"]   = float("nan")
        ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(ERROR_LOG, "a") as f:
            f.write(json.dumps({"clip_id": clip_id, "condition": condition_name,
                                "category": "mos_lookup", "error": "no MOS entry"}) + "\n")

    row.update(_compute_distortion(enh, ref, clip_id, condition_name))
    row.update(_compute_perceptual(enh_path, ref_path, enh, ref, 16000, clip_id, condition_name))
    row.update(_compute_non_intrusive(enh, ref, enh_path, 16000, clip_id, condition_name))
    row.update(_compute_special(enh, ref, 16000, clip_id, condition_name))
    return row


def _row_worker(args: tuple) -> dict:
    clip_id, condition_name, enh_path, clean_path, mos_entry = args
    return compute_dns_metrics(clip_id, condition_name, enh_path, clean_path, mos_entry)


def _existing_keys() -> set:
    if not METRICS_CSV.exists():
        return set()
    with open(METRICS_CSV, newline="") as f:
        return {(r["clip_id"], r["condition_name"]) for r in csv.DictReader(f)}


def main() -> None:
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    mos_labels = {}
    mos_csv = DNS_DIR / "dns_mos_labels.csv"
    if mos_csv.exists():
        mos_labels = load_dns_mos_labels(mos_csv)

    clip_pairs = load_dns_clip_pairs(DNS_DIR)
    existing   = _existing_keys()
    write_hdr  = not METRICS_CSV.exists()

    jobs = []
    for pair in clip_pairs:
        cid       = pair["clip_id"]
        clean_p   = pair["clean_path"]
        if clean_p is None:
            continue
        # dns_noisy baseline
        if (cid, "dns_noisy") not in existing:
            mos_e = mos_labels.get(cid)
            jobs.append((cid, "dns_noisy", pair["noisy_path"], clean_p, mos_e))
        # each enhanced version
        for sys_name, enh_p in pair["enhanced_paths"].items():
            cname = f"dns_{sys_name}" if not sys_name.startswith("dns_") else sys_name
            if (cid, cname) not in existing:
                mos_e = mos_labels.get(cid)
                jobs.append((cid, cname, enh_p, clean_p, mos_e))

    with open(METRICS_CSV, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_hdr:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=4) as pool:
            for row in tqdm(pool.map(_row_worker, jobs), total=len(jobs), desc="dns-metrics"):
                writer.writerow(row)
                csvfile.flush()


if __name__ == "__main__":
    main()
