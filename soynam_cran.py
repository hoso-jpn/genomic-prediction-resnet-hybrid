"""Build a reproducible canonical dataset from the CRAN SoyNAM 1.6.2 release.

The published R package is the data source of record (Issue #6). This module
pins that release by SHA-256, runs ``scripts/build_soynam_canonical.R`` to fit
the phenotype model and export intermediates, then turns those intermediates
into the per-family gzip TSV layout the verified loader already reads, plus a
provenance manifest.

What this builder deliberately does not do:

* it does not use the NAM package, or SoyNAM's own ``BLUP()`` genotype path
  (MAF filtering, Markov imputation, duplicate removal). Genotypes are the
  quality-assured ``gen.qa`` matrix exactly as distributed;
* it does not relabel the numeric dosages as ``A``/``H``/``B``. No mapping
  from 0/1/2 to allele letters is documented, so the numeric representation
  is carried through as its own input contract;
* it never writes raw or derived individual-level data into the repository.
  Outputs go to an explicit ``--output-dir`` that is expected to be ignored
  by git, and no sample identifier is printed to stdout.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R_SCRIPT = ROOT / "scripts" / "build_soynam_canonical.R"

SOURCE_URL = "https://cran.r-project.org/src/contrib/SoyNAM_1.6.2.tar.gz"
SOURCE_SHA256 = "0bd87f7b101456006a42a11679809349f7545b95dee0b43d954aa5b642995aaa"
SOURCE_VERSION = "1.6.2"
SOURCE_PUBLICATION_DATE = "2022-01-04"
SOURCE_PACKAGE_LICENSE = "GPL-3"
DATASET_ID = "cran-soynam-1.6.2-yield-blup-false"
TRAIT = "Yld (kg/ha)"
FORMULA = "yield ~ (1 | environ) + (1 | strain)"
USE_CHECK_REASON = (
    "use.check=TRUE was rejected: in the raw data.line object every line "
    "carries the single set code '2A', so the check covariate is absent for "
    "the six environments without 2A checks (10,355 observations dropped "
    "with no documented intent) and the remaining rows receive another "
    "set's check value rather than their own. No statement in the package "
    "documents that exclusion. use.check=FALSE drops only missing-yield rows "
    "and keeps the environment random effect."
)
REDISTRIBUTION_NOTE = (
    "GPL-3 is the license of the CRAN package. The redistribution terms of "
    "the underlying SoyBase SoyNAM data are unverified; no raw or derived "
    "individual-level data is committed to this repository."
)
GENOTYPE_ENCODING_NAME = "cran-numeric-dosage"
GENOTYPE_ENCODING_NOTE = (
    "0/1/2 dosages relative to the founder parent, as distributed in gen.qa; "
    "missing values are written as NA. Not relabelled to A/H/B."
)

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = f"soynam-cran-{SOURCE_VERSION}-manifest.json"
MANIFEST_GLOB = "soynam-cran-*-manifest.json"

PHENOTYPE_SUFFIX = "_phenotype_data.tsv.gz"
GENOTYPE_SUFFIX = "_SNP_genotype_Wm82.a1.tsv.gz"
SAMPLE_COLUMN = "Corrected Strain"
PHENOTYPE_COLUMN = TRAIT
MARKER_ID_HEADER = "marker_id"
MISSING_GENOTYPE_TOKEN = "NA"
VALID_GENOTYPE_TOKENS = frozenset({"0", "1", "2", MISSING_GENOTYPE_TOKEN})


@dataclass(frozen=True)
class CanonicalExpectations:
    """Reference values a finished build must reproduce.

    The defaults describe the audited CRAN 1.6.2 result. Tests that exercise
    the builder on small synthetic intermediates pass their own expectations
    (or ``None`` for a field) instead of the production constants.
    """

    samples: int | None = None
    families: int | None = None
    markers: int | None = None
    family_numbers: tuple[int, ...] | None = None
    marker_list_sha256: str | None = None
    sample_list_sha256: str | None = None
    joined_phenotype_sha256: str | None = None
    genotype_missing_rate: float | None = None
    missing_rate_tolerance: float = 5e-9


CRAN_EXPECTATIONS = CanonicalExpectations(
    samples=5142,
    families=39,
    markers=4312,
    family_numbers=(
        2,
        3,
        4,
        5,
        6,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        17,
        18,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        48,
        50,
        54,
        64,
    ),
    marker_list_sha256=(
        "2f24bd7524cd3b7cf003fc8f5c99dffd540a03707d0f64e0f04c07dad001d7d4"
    ),
    sample_list_sha256=(
        "d1a43e4ac11abdd70fbbd99b96cb06e468ff130c5d02cba2e51e3245e3dcdfaf"
    ),
    joined_phenotype_sha256=(
        "22a632e2f8a1cea001bf5c4466c472d23df0331253e1ef7c97ca228664663c75"
    ),
    genotype_missing_rate=0.25647186,
    missing_rate_tolerance=5e-9,
)


class BuildError(RuntimeError):
    """A build precondition or output check failed."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1_048_576) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(lines: Iterable[str], *, trailing_newline: bool) -> str:
    """Hash text lines joined by LF, with or without a trailing newline.

    Both conventions are recorded because the audit pinned both: the joined
    form without a trailing newline, and the one-per-line file form with it.
    """
    body = "\n".join(lines)
    if trailing_newline and body:
        body += "\n"
    return sha256_bytes(body.encode("utf-8"))


def canonical_json_hash(payload: object) -> str:
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def family_label(family_number: int) -> str:
    if family_number < 0:
        raise BuildError(f"family numbers must be non-negative: {family_number}")
    return f"NAM{family_number:02d}"


def phenotype_filename(label: str) -> str:
    return f"{label}{PHENOTYPE_SUFFIX}"


def genotype_filename(label: str, markers: int) -> str:
    return f"{label}_{markers}{GENOTYPE_SUFFIX}"


def write_deterministic_gzip(path: Path, text: str) -> None:
    """Write UTF-8 text as gzip with no embedded name and a fixed timestamp.

    Rebuilding the same dataset must produce byte-identical files, so the
    gzip header must not carry the source filename or the current time.
    """
    payload = text.encode("utf-8")
    with (
        path.open("wb") as raw,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
        ) as handle,
    ):
        handle.write(payload)


def download_source(url: str, destination: Path) -> Path:
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return destination


def resolve_source_tarball(
    *,
    source_tarball: Path | None,
    cache_dir: Path | None,
    url: str = SOURCE_URL,
    expected_sha256: str = SOURCE_SHA256,
) -> Path:
    """Return a tarball whose checksum matches, fetching it only if needed.

    A supplied ``--source-tarball`` is used as-is and never triggers a network
    request. The checksum is verified before the archive is unpacked or handed
    to R, so a mismatched download is rejected before any code runs on it.
    """
    if source_tarball is not None:
        tarball = Path(source_tarball)
        if not tarball.is_file():
            raise BuildError(f"source tarball not found: {tarball.name}")
    else:
        directory = Path(cache_dir) if cache_dir is not None else Path.cwd()
        directory.mkdir(parents=True, exist_ok=True)
        tarball = directory / Path(url).name
        if not tarball.is_file():
            download_source(url, tarball)

    actual = sha256_file(tarball)
    if actual != expected_sha256:
        raise BuildError(
            f"source tarball checksum mismatch for '{tarball.name}': "
            f"expected {expected_sha256}, found {actual}"
        )
    return tarball


def run_r_builder(
    *, rscript: str, tarball: Path, workdir: Path, intermediate_dir: Path
) -> None:
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    command = [
        rscript,
        "--vanilla",
        str(R_SCRIPT),
        f"--tarball={tarball}",
        f"--workdir={workdir}",
        f"--outdir={intermediate_dir}",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise BuildError(
            "the R builder failed "
            f"(exit {result.returncode}): {result.stderr.strip()[-2000:]}"
        )


@dataclass(frozen=True)
class Intermediates:
    sample_ids: tuple[str, ...]
    families: tuple[int, ...]
    values: tuple[str, ...]
    marker_ids: tuple[str, ...]
    genotypes: tuple[tuple[str, ...], ...]  # marker rows x sample columns
    summary: dict[str, str]


def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise BuildError(f"the R builder did not produce '{path.name}'")
    text = path.read_text(encoding="utf-8")
    return text.split("\n")[:-1] if text.endswith("\n") else text.split("\n")


def read_intermediates(intermediate_dir: Path) -> Intermediates:
    phenotype_lines = _read_lines(intermediate_dir / "phenotype.tsv")
    if len(phenotype_lines) < 2:
        raise BuildError("the phenotype intermediate has no data rows")
    header = phenotype_lines[0].split("\t")
    if header != ["strain", "family", "value"]:
        raise BuildError(f"unexpected phenotype intermediate header: {header}")

    sample_ids: list[str] = []
    families: list[int] = []
    values: list[str] = []
    for line in phenotype_lines[1:]:
        fields = line.split("\t")
        if len(fields) != 3:
            raise BuildError("malformed row in the phenotype intermediate")
        strain, family, value = fields
        if not strain.strip():
            raise BuildError("the phenotype intermediate contains an empty strain ID")
        try:
            families.append(int(family))
        except ValueError as error:
            raise BuildError(
                f"non-integer family in the intermediate: {family!r}"
            ) from error
        float(value)  # reject a non-numeric adjusted phenotype early
        sample_ids.append(strain)
        values.append(value)

    genotype_lines = _read_lines(intermediate_dir / "genotype.tsv")
    if len(genotype_lines) < 2:
        raise BuildError("the genotype intermediate has no marker rows")
    genotype_header = genotype_lines[0].split("\t")
    if genotype_header[0] != MARKER_ID_HEADER:
        raise BuildError(
            f"unexpected genotype intermediate header: {genotype_header[0]!r}"
        )
    genotype_samples = genotype_header[1:]
    if genotype_samples != sample_ids:
        raise BuildError(
            "phenotype and genotype intermediates disagree on the sample list; "
            "the join must be performed by strain ID in the R builder"
        )

    marker_ids: list[str] = []
    rows: list[tuple[str, ...]] = []
    for line in genotype_lines[1:]:
        fields = line.split("\t")
        marker = fields[0]
        row = tuple(fields[1:])
        if not marker.strip():
            raise BuildError("the genotype intermediate contains an empty marker ID")
        if len(row) != len(sample_ids):
            raise BuildError(
                f"marker row width does not match the sample count for '{marker}'"
            )
        unknown = sorted(set(row) - VALID_GENOTYPE_TOKENS)
        if unknown:
            raise BuildError(f"unexpected genotype values for '{marker}': {unknown}")
        marker_ids.append(marker)
        rows.append(row)

    summary: dict[str, str] = {}
    for line in _read_lines(intermediate_dir / "build-summary.tsv"):
        if not line:
            continue
        key, _, value = line.partition("\t")
        summary[key] = value

    return Intermediates(
        sample_ids=tuple(sample_ids),
        families=tuple(families),
        values=tuple(values),
        marker_ids=tuple(marker_ids),
        genotypes=tuple(rows),
        summary=summary,
    )


def validate_intermediates(data: Intermediates) -> None:
    # Family consistency is checked before the duplicate-ID rule so that a
    # repeated strain carrying two different families is reported as the
    # family conflict it is, rather than as a plain duplicate.
    by_strain: dict[str, int] = {}
    for strain, family in zip(data.sample_ids, data.families):
        if by_strain.setdefault(strain, family) != family:
            # The identifier is withheld deliberately: build logs must not
            # enumerate individual samples.
            raise BuildError(
                "a strain maps to more than one family in the intermediate dataset"
            )
    if len(set(data.sample_ids)) != len(data.sample_ids):
        raise BuildError("duplicate sample IDs in the intermediate dataset")
    if len(set(data.marker_ids)) != len(data.marker_ids):
        raise BuildError("duplicate marker IDs in the intermediate dataset")
    if sorted(data.sample_ids) != list(data.sample_ids):
        raise BuildError("intermediate samples must already be sorted by ascending ID")
    if not data.sample_ids or not data.marker_ids:
        raise BuildError("the intermediate dataset is empty")


def order_samples(data: Intermediates) -> list[int]:
    """Order samples by family number, then by ascending sample ID."""
    return sorted(
        range(len(data.sample_ids)),
        key=lambda index: (data.families[index], data.sample_ids[index]),
    )


def build_family_tables(data: Intermediates) -> list[tuple[int, list[int]]]:
    grouped: dict[int, list[int]] = {}
    for index in order_samples(data):
        grouped.setdefault(data.families[index], []).append(index)
    return [(family, grouped[family]) for family in sorted(grouped)]


def render_phenotype(data: Intermediates, indices: Sequence[int]) -> str:
    lines = [f"{SAMPLE_COLUMN}\t{PHENOTYPE_COLUMN}"]
    lines.extend(f"{data.sample_ids[i]}\t{data.values[i]}" for i in indices)
    return "\n".join(lines) + "\n"


def render_genotype(data: Intermediates, indices: Sequence[int]) -> str:
    header = "\t".join([MARKER_ID_HEADER, *(data.sample_ids[i] for i in indices)])
    lines = [header]
    for marker, row in zip(data.marker_ids, data.genotypes):
        lines.append("\t".join([marker, *(row[i] for i in indices)]))
    return "\n".join(lines) + "\n"


def genotype_missing_rate(data: Intermediates) -> float:
    total = len(data.marker_ids) * len(data.sample_ids)
    missing = sum(row.count(MISSING_GENOTYPE_TOKEN) for row in data.genotypes)
    return missing / total


def check_expectations(
    data: Intermediates,
    hashes: dict[str, str],
    expectations: CanonicalExpectations,
) -> None:
    families = sorted(set(data.families))
    checks: list[tuple[str, object, object]] = [
        ("sample count", expectations.samples, len(data.sample_ids)),
        ("family count", expectations.families, len(families)),
        ("marker count", expectations.markers, len(data.marker_ids)),
        (
            "family numbers",
            None
            if expectations.family_numbers is None
            else list(expectations.family_numbers),
            families,
        ),
        (
            "marker ID list SHA-256",
            expectations.marker_list_sha256,
            hashes["marker_id_list_sha256"],
        ),
        (
            "sample ID list SHA-256",
            expectations.sample_list_sha256,
            hashes["sample_id_list_sha256"],
        ),
        (
            "joined phenotype SHA-256",
            expectations.joined_phenotype_sha256,
            hashes["joined_phenotype_sha256"],
        ),
    ]
    for label, expected, actual in checks:
        if expected is not None and expected != actual:
            raise BuildError(
                f"{label} does not match the audited reference: "
                f"expected {expected}, found {actual}"
            )

    if expectations.genotype_missing_rate is not None:
        actual_rate = genotype_missing_rate(data)
        if (
            abs(actual_rate - expectations.genotype_missing_rate)
            > expectations.missing_rate_tolerance
        ):
            raise BuildError(
                "genotype missing rate does not match the audited reference: "
                f"expected {expectations.genotype_missing_rate}, found {actual_rate}"
            )


def dataset_hashes(data: Intermediates) -> dict[str, str]:
    by_sample_id = sorted(
        range(len(data.sample_ids)), key=lambda index: data.sample_ids[index]
    )
    sample_ids = [data.sample_ids[i] for i in by_sample_id]
    joined_phenotype = [f"{data.sample_ids[i]}\t{data.values[i]}" for i in by_sample_id]
    return {
        "marker_id_list_sha256": sha256_lines(data.marker_ids, trailing_newline=False),
        "marker_id_file_sha256": sha256_lines(data.marker_ids, trailing_newline=True),
        "sample_id_list_sha256": sha256_lines(
            sorted(sample_ids), trailing_newline=True
        ),
        "joined_phenotype_sha256": sha256_lines(
            joined_phenotype, trailing_newline=True
        ),
    }


def builder_source_checksums() -> dict[str, str]:
    return {
        path.name: sha256_file(path) for path in (Path(__file__).resolve(), R_SCRIPT)
    }


def build_manifest(
    data: Intermediates,
    hashes: dict[str, str],
    outputs: dict[str, str],
    *,
    source_sha256: str,
    url: str,
    now: datetime | None = None,
) -> dict:
    families = sorted(set(data.families))
    summary = data.summary
    content = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "source": {
            "url": url,
            "package": "SoyNAM",
            "version": SOURCE_VERSION,
            "tarball_sha256": source_sha256,
            "cran_publication_date": SOURCE_PUBLICATION_DATE,
            "package_license": SOURCE_PACKAGE_LICENSE,
            "redistribution_terms": REDISTRIBUTION_NOTE,
        },
        "phenotype": {
            "trait": TRAIT,
            "formula": FORMULA,
            "reml": True,
            "use_check": False,
            "use_check_rationale": USE_CHECK_REASON,
            "environments": _as_int(summary.get("environments")),
            "source_phenotype_rows": _as_int(summary.get("source_phenotype_rows")),
            "dropped_missing_yield": _as_int(summary.get("dropped_missing_yield")),
            "model_frame_rows": _as_int(summary.get("model_frame_rows")),
        },
        "environment": {
            "r_version": summary.get("r_version"),
            "lme4_version": summary.get("lme4_version"),
            "matrix_version": summary.get("matrix_version"),
            "blas": summary.get("blas"),
            "lapack": summary.get("lapack"),
        },
        "dataset": {
            "samples": len(data.sample_ids),
            "families": len(families),
            "family_numbers": families,
            "markers": len(data.marker_ids),
            "genotype_encoding": GENOTYPE_ENCODING_NAME,
            "genotype_encoding_note": GENOTYPE_ENCODING_NOTE,
            "genotype_missing_rate": genotype_missing_rate(data),
        },
        "checksums": {
            **hashes,
            "files": outputs,
            "builder_sources": builder_source_checksums(),
        },
    }
    manifest = dict(content)
    manifest["content_hash"] = canonical_json_hash(content)
    moment = now if now is not None else datetime.now(UTC)
    manifest["created_at"] = moment.strftime("%Y-%m-%dT%H:%M:%SZ")
    return manifest


def _as_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def verify_outputs(output_dir: Path, data: Intermediates) -> None:
    """Re-read the finished dataset through the production loader."""
    import soynam_data

    files = soynam_data.list_family_files(output_dir)
    expected_families = len(set(data.families))
    if len(files) != expected_families:
        raise BuildError(
            f"expected {expected_families} family file pairs, found {len(files)}"
        )
    dataset = soynam_data.load_soynam_dataset(output_dir, family_files=files)
    if dataset.genotypes.shape != (len(data.sample_ids), len(data.marker_ids)):
        raise BuildError(
            "the written dataset does not reload at the expected dimensions"
        )
    if list(dataset.marker_names) != list(data.marker_ids):
        raise BuildError("marker order changed between writing and reloading")
    if sorted(dataset.sample_names) != sorted(data.sample_ids):
        raise BuildError("sample set changed between writing and reloading")


def write_dataset(data: Intermediates, output_dir: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    marker_count = len(data.marker_ids)
    for family, indices in build_family_tables(data):
        label = family_label(family)
        phenotype_path = output_dir / phenotype_filename(label)
        genotype_path = output_dir / genotype_filename(label, marker_count)
        write_deterministic_gzip(phenotype_path, render_phenotype(data, indices))
        write_deterministic_gzip(genotype_path, render_genotype(data, indices))
        outputs[phenotype_path.name] = sha256_file(phenotype_path)
        outputs[genotype_path.name] = sha256_file(genotype_path)
    return outputs


def publish_dataset(staging: Path, output_dir: Path) -> None:
    """Publish a verified dataset without exposing a partial target.

    The final directory is first copied to a hidden sibling of ``output_dir``.
    Because that sibling is on the same filesystem, the last ``replace`` is a
    single directory rename.  A copy or rename failure therefore leaves either
    the caller's original empty directory or no target at all, never a prefix
    of the dataset's files.
    """
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{output_dir.name}-publish-", dir=output_dir.parent
        ) as publish_scratch:
            ready = Path(publish_scratch) / "dataset"
            shutil.copytree(staging, ready)
            if output_dir.exists():
                output_dir.rmdir()
            ready.replace(output_dir)
    except OSError as error:
        raise BuildError(
            f"failed to publish the completed dataset to '{output_dir.name}'"
        ) from error


def build(
    *,
    output_dir: Path,
    source_tarball: Path | None = None,
    cache_dir: Path | None = None,
    rscript: str = "Rscript",
    expectations: CanonicalExpectations = CRAN_EXPECTATIONS,
    intermediate_dir: Path | None = None,
    url: str = SOURCE_URL,
    expected_sha256: str = SOURCE_SHA256,
    now: datetime | None = None,
) -> dict:
    """Produce the canonical dataset and its manifest in ``output_dir``.

    Everything is staged in a temporary directory and moved into place only
    after the outputs reload correctly, so a failed build never leaves a
    half-written dataset that looks finished.

    ``intermediate_dir`` skips the R step and consumes existing intermediate
    files; it exists so the assembly, ordering, and manifest logic can be
    tested without CRAN or R.
    """
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise BuildError(
            f"output directory '{output_dir.name}' is not empty; "
            "point --output-dir at a new directory"
        )

    with tempfile.TemporaryDirectory(prefix="soynam-cran-") as scratch:
        scratch_path = Path(scratch)
        if intermediate_dir is None:
            tarball = resolve_source_tarball(
                source_tarball=source_tarball,
                cache_dir=cache_dir,
                url=url,
                expected_sha256=expected_sha256,
            )
            source_sha256 = sha256_file(tarball)
            staged_intermediates = scratch_path / "intermediates"
            run_r_builder(
                rscript=rscript,
                tarball=tarball,
                workdir=scratch_path / "work",
                intermediate_dir=staged_intermediates,
            )
        else:
            staged_intermediates = Path(intermediate_dir)
            source_sha256 = expected_sha256

        data = read_intermediates(staged_intermediates)
        validate_intermediates(data)
        hashes = dataset_hashes(data)
        check_expectations(data, hashes, expectations)

        staging = scratch_path / "dataset"
        staging.mkdir()
        outputs = write_dataset(data, staging)
        verify_outputs(staging, data)

        manifest = build_manifest(
            data,
            hashes,
            outputs,
            source_sha256=source_sha256,
            url=url,
            now=now,
        )
        (staging / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        publish_dataset(staging, output_dir)

    return manifest


def find_dataset_manifest(data_dir: Path) -> Path | None:
    """Return the canonical manifest in a data directory, if there is one.

    Synthetic fixtures and the legacy raw layout have no manifest, which is
    not an error: those datasets simply carry no CRAN provenance record.
    """
    matches = sorted(Path(data_dir).glob(MANIFEST_GLOB))
    if len(matches) > 1:
        raise ValueError(
            "multiple canonical dataset manifests found in the data directory: "
            f"{[path.name for path in matches]}"
        )
    return matches[0] if matches else None


def describe_dataset_manifest(data_dir: Path) -> dict[str, str] | None:
    """Describe the canonical manifest by filename and checksum, never by path."""
    manifest = find_dataset_manifest(data_dir)
    if manifest is None:
        return None
    return {"filename": manifest.name, "sha256": sha256_file(manifest)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new (or empty) directory that receives the canonical dataset",
    )
    parser.add_argument(
        "--source-tarball",
        type=Path,
        help="local SoyNAM source tarball; skips the download entirely",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="directory the source tarball is downloaded into when not supplied",
    )
    parser.add_argument(
        "--rscript", default="Rscript", help="Rscript executable to run the R builder"
    )
    args = parser.parse_args(argv)

    try:
        manifest = build(
            output_dir=args.output_dir,
            source_tarball=args.source_tarball,
            cache_dir=args.cache_dir,
            rscript=args.rscript,
        )
    except BuildError as error:
        print(f"build failed: {error}", file=sys.stderr)
        return 1

    dataset = manifest["dataset"]
    print(
        f"wrote {dataset['samples']} samples, {dataset['families']} families, "
        f"{dataset['markers']} markers"
    )
    print(f"manifest content hash: {manifest['content_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
