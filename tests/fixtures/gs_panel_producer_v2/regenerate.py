"""Rebuild synthetic fixtures using the pinned upstream producer checkout.

Usage: python tests/fixtures/gs_panel_producer_v2/regenerate.py /path/to/producer
No network access, real cohort data, variant calling, or containers are used.
"""

import argparse
import gzip
import hashlib
import json
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

PRODUCER_COMMIT = "3158ca50c2c13c31bdc80db302c7df4bbb5670bf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("producer", type=Path)
    producer = parser.parse_args().producer.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(producer), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != PRODUCER_COMMIT:
        parser.error(f"expected producer commit {PRODUCER_COMMIT}, got {commit}")
    sys.path.insert(0, str(producer / "bin"))
    manifest_builder = runpy.run_path(str(producer / "bin/build_gs_panel_manifest.py"))
    root = Path(__file__).resolve().parent
    header = (
        "##fileformat=VCFv4.2\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t001\tNA\tsample3\n"
    )
    variants = (
        "Chr1\t100\t.\tA\tT\t100\tPASS\t.\tGT\t0/0\t0|1\t1/1\n"
        "Chr1\t200\t.\tG\tC\t100\tPASS\t.\tGT\t./.\t1|1\t0/0\n"
    )
    for cohort, rows in (("producer", variants), ("empty", "")):
        output = root / cohort
        output.mkdir(exist_ok=True)
        names = {
            "matrix": f"{cohort}.gs_panel.genotype_matrix.tsv.gz",
            "sample-metadata": f"{cohort}.gs_panel.sample_metadata.tsv",
            "variant-metadata": f"{cohort}.gs_panel.variant_metadata.tsv",
        }
        with tempfile.TemporaryDirectory() as scratch:
            temporary = Path(scratch)
            vcf = temporary / "synthetic.vcf.gz"
            vcf.write_bytes(gzip.compress((header + rows).encode(), mtime=0))
            command = [
                sys.executable,
                str(producer / "bin/build_gs_panel.py"),
                "--gs-pass-vcf",
                str(vcf),
                "--cohort-id",
                cohort,
                "--sample-ploidy",
                "2",
                "--genotype-accounting-output",
                str(temporary / "accounting.tsv"),
                "--genotype-accounting-summary-output",
                str(temporary / "summary.txt"),
            ]
            for option, name in names.items():
                command.extend([f"--{option}-output", str(output / name)])
            subprocess.run(command, check=True, capture_output=True, text=True)
        checksums = {
            name: "sha256:" + hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in names.values()
        }
        manifest = manifest_builder["build_manifest"](
            cohort_id=cohort,
            pipeline_version="synthetic-fixture",
            git_commit=commit,
            containers={
                process: "fixture/unused:synthetic"
                for process in manifest_builder["CONTAINER_PROCESS_NAMES"]
            },
            sample_ploidy=2,
            snp_filter_params={},
            panel_status="populated" if rows else "empty",
            checksums=checksums,
            run_id=f"synthetic-{cohort}",
            generated_at="2026-09-10T00:00:00Z",
        )
        (output / f"{cohort}.gs_panel.manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
