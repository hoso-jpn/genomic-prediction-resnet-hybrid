"""Measure loader-only wall/RSS/disk in an isolated process (synthetic inputs)."""

import argparse
import gzip
import hashlib
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import adzuki_gs_panel_data as gs  # noqa: E402


def generate(directory, samples, markers):
    cohort = "benchmark"
    names = [f"S{i}" for i in range(samples)]
    paths = [
        directory / (cohort + suffix)
        for suffix in (
            gs.MATRIX_SUFFIX,
            gs.SAMPLE_METADATA_SUFFIX,
            gs.VARIANT_METADATA_SUFFIX,
        )
    ]
    with gzip.open(paths[0], "wt") as matrix, paths[2].open("w") as variants:
        matrix.write("variant_key\t" + "\t".join(names) + "\n")
        variants.write("variant_index\tvariant_key\n")
        row = "\t".join(str(i % 3 - 1) for i in range(samples))
        for index in range(markers):
            key = f"Chr1:{index + 1}:A:T"
            matrix.write(f"{key}\t{row}\n")
            variants.write(f"{index}\t{key}\n")
    paths[1].write_text(
        "sample_index\tsample_id\n"
        + "".join(f"{i}\t{name}\n" for i, name in enumerate(names))
    )
    manifest = {
        "schema_version": 2,
        "cohort_id": cohort,
        "parameters": {"sample_ploidy": 2},
        "genotype_encoding": {
            "schema": gs.SUPPORTED_ENCODING_SCHEMA,
            "matrix_orientation": gs.EXPECTED_ORIENTATION,
            "missing_token": gs.EXPECTED_MISSING_TOKEN,
            "ploidy": gs.EXPECTED_PLOIDY,
            "dosage_by_genotype": gs.EXPECTED_DOSAGES,
        },
        "checksums": {
            p.name: "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
    }
    (directory / (cohort + gs.MANIFEST_SUFFIX)).write_text(json.dumps(manifest))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--markers", type=int, default=1000)
    parser.add_argument("--load", type=Path)
    args = parser.parse_args()
    if args.load:
        before = sum(p.stat().st_size for p in args.load.iterdir())
        start = time.perf_counter()
        panel = gs.load_gs_panel(args.load)
        print(
            json.dumps(
                {
                    "samples": panel.genotypes.shape[0],
                    "markers": panel.genotypes.shape[1],
                    "wall_seconds": time.perf_counter() - start,
                    "peak_rss_kib_linux": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss,
                    "temporary_disk_bytes": sum(
                        p.stat().st_size for p in args.load.iterdir()
                    )
                    - before,
                }
            )
        )
    else:
        with tempfile.TemporaryDirectory() as temporary:
            generate(Path(temporary), args.samples, args.markers)
            subprocess.run([sys.executable, __file__, "--load", temporary], check=True)


if __name__ == "__main__":
    main()
