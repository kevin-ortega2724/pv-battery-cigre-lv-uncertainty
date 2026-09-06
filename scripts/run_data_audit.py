import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.data.household import aggregate_15min, load_raw, write_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    args = parser.parse_args()
    config = yaml.safe_load(Path("configs/data.yaml").read_text(encoding="utf-8"))
    source = Path(args.input or config["raw_path"])
    if not source.is_file():
        raise SystemExit(f"Raw dataset not found: {source}. Supply --input; the raw file must remain outside Git.")
    frame = load_raw(source)
    report = write_audit(frame, "results/metrics")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    provenance = {
        "file_name": source.name,
        "size_bytes": source.stat().st_size,
        "sha256": digest.hexdigest(),
        "raw_file_modified": False,
    }
    Path("results/metrics/data_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    processed = aggregate_15min(frame, config["minimum_observations_per_interval"])
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    processed.to_csv("data/processed/household_15min.csv")
    print({**report, "valid_15min_intervals": len(processed), "sha256": provenance["sha256"]})


if __name__ == "__main__":
    main()
