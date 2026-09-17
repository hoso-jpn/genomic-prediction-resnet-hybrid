"""Assemble the canonical dataset from small synthetic intermediates.

No CRAN download, no R, and no real SoyNAM data: the R step is bypassed by
handing the builder intermediate files it would otherwise have produced, so
the ordering, determinism, manifest, and rejection rules can be checked
without shipping individual-level data in the repository.
"""

import gzip
import json

import numpy as np
import pytest

import soynam_cran
import soynam_data

MARKERS = ["Gm01_100_A_C", "Gm01_200_T_G", "Gm02_300_C_T"]


def write_intermediates(
    directory,
    *,
    samples=(("S0001", 2), ("S0002", 2), ("S0003", 3), ("S0004", 3)),
    markers=MARKERS,
    genotypes=None,
    summary_overrides=None,
):
    directory.mkdir(parents=True, exist_ok=True)
    sample_ids = [sample for sample, _ in samples]
    phenotype_rows = ["strain\tfamily\tvalue"]
    for index, (sample, family) in enumerate(samples):
        phenotype_rows.append(f"{sample}\t{family}\t{3000.0 + index:.17g}")
    (directory / "phenotype.tsv").write_text(
        "\n".join(phenotype_rows) + "\n", encoding="utf-8"
    )

    if genotypes is None:
        cycle = ["0", "1", "2", "NA"]
        genotypes = [
            [cycle[(row + column) % len(cycle)] for column in range(len(sample_ids))]
            for row in range(len(markers))
        ]
    genotype_rows = ["\t".join(["marker_id", *sample_ids])]
    for marker, row in zip(markers, genotypes):
        genotype_rows.append("\t".join([marker, *row]))
    (directory / "genotype.tsv").write_text(
        "\n".join(genotype_rows) + "\n", encoding="utf-8"
    )

    summary = {
        "r_version": "4.5.3",
        "lme4_version": "2.0.6",
        "matrix_version": "1.7.5",
        "blas": "libopenblasp-r0.3.34.so",
        "lapack": "libopenblasp-r0.3.34.so",
        "environments": "18",
        "source_phenotype_rows": "60744",
        "dropped_missing_yield": "354",
        "model_frame_rows": "60390",
    }
    summary.update(summary_overrides or {})
    (directory / "build-summary.tsv").write_text(
        "\n".join(f"{key}\t{value}" for key, value in summary.items()) + "\n",
        encoding="utf-8",
    )
    return directory


def build(tmp_path, *, name="dataset", intermediates=None, expectations=None, **kwargs):
    source = intermediates or write_intermediates(tmp_path / "intermediates")
    return soynam_cran.build(
        output_dir=tmp_path / name,
        intermediate_dir=source,
        expectations=expectations or soynam_cran.CanonicalExpectations(),
        **kwargs,
    )


def read_gzip_text(path):
    with gzip.open(path, mode="rt", encoding="utf-8", newline="") as handle:
        return handle.read()


def test_source_checksum_mismatch_is_rejected_before_running_r(tmp_path):
    tarball = tmp_path / "SoyNAM_1.6.2.tar.gz"
    tarball.write_bytes(b"not the published release")
    with pytest.raises(soynam_cran.BuildError, match="checksum mismatch"):
        soynam_cran.resolve_source_tarball(source_tarball=tarball, cache_dir=None)


def test_missing_local_tarball_is_rejected(tmp_path):
    with pytest.raises(soynam_cran.BuildError, match="not found"):
        soynam_cran.resolve_source_tarball(
            source_tarball=tmp_path / "absent.tar.gz", cache_dir=None
        )


def test_nonempty_output_directory_is_refused(tmp_path):
    output = tmp_path / "dataset"
    output.mkdir()
    (output / "existing.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(soynam_cran.BuildError, match="not empty"):
        build(tmp_path, name="dataset")
    assert (output / "existing.txt").read_text(encoding="utf-8") == "keep me"


def test_failed_build_leaves_no_partial_output(tmp_path):
    intermediates = write_intermediates(tmp_path / "intermediates")
    expectations = soynam_cran.CanonicalExpectations(samples=999)
    with pytest.raises(soynam_cran.BuildError, match="sample count"):
        build(tmp_path, intermediates=intermediates, expectations=expectations)
    assert not (tmp_path / "dataset").exists() or not list(
        (tmp_path / "dataset").iterdir()
    )


def test_family_sample_and_marker_ordering(tmp_path):
    samples = (("S0004", 3), ("S0001", 2), ("S0003", 3), ("S0002", 2))
    intermediates = write_intermediates(
        tmp_path / "intermediates", samples=tuple(sorted(samples))
    )
    build(tmp_path, intermediates=intermediates)
    output = tmp_path / "dataset"

    names = sorted(path.name for path in output.glob("*.tsv.gz"))
    assert names == [
        "NAM02_3_SNP_genotype_Wm82.a1.tsv.gz",
        "NAM02_phenotype_data.tsv.gz",
        "NAM03_3_SNP_genotype_Wm82.a1.tsv.gz",
        "NAM03_phenotype_data.tsv.gz",
    ]

    phenotype = read_gzip_text(output / "NAM02_phenotype_data.tsv.gz").splitlines()
    assert phenotype[0] == "Corrected Strain\tYld (kg/ha)"
    assert [line.split("\t")[0] for line in phenotype[1:]] == ["S0001", "S0002"]

    genotype = read_gzip_text(
        output / "NAM02_3_SNP_genotype_Wm82.a1.tsv.gz"
    ).splitlines()
    assert genotype[0].split("\t") == ["marker_id", "S0001", "S0002"]
    assert [line.split("\t")[0] for line in genotype[1:]] == MARKERS


def test_rebuild_is_byte_identical_and_manifest_content_hash_is_stable(tmp_path):
    intermediates = write_intermediates(tmp_path / "intermediates")
    first = build(tmp_path, name="first", intermediates=intermediates)
    second = build(tmp_path, name="second", intermediates=intermediates)

    for path in sorted((tmp_path / "first").glob("*.tsv.gz")):
        twin = tmp_path / "second" / path.name
        assert path.read_bytes() == twin.read_bytes(), path.name

    assert first["content_hash"] == second["content_hash"]
    without_time = {key: value for key, value in first.items() if key != "created_at"}
    assert without_time == {
        key: value for key, value in second.items() if key != "created_at"
    }


def test_gzip_header_carries_no_filename_or_timestamp(tmp_path):
    build(tmp_path)
    raw = (tmp_path / "dataset" / "NAM02_phenotype_data.tsv.gz").read_bytes()
    assert raw[4:8] == b"\x00\x00\x00\x00"  # mtime field
    assert raw[3] & 0x08 == 0  # FNAME flag


def test_manifest_records_provenance_and_output_checksums(tmp_path):
    manifest = build(tmp_path)
    written = json.loads(
        (tmp_path / "dataset" / soynam_cran.MANIFEST_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert written == manifest
    assert manifest["source"]["tarball_sha256"] == soynam_cran.SOURCE_SHA256
    assert manifest["source"]["package_license"] == "GPL-3"
    assert "unverified" in manifest["source"]["redistribution_terms"]
    assert manifest["phenotype"]["use_check"] is False
    assert manifest["phenotype"]["reml"] is True
    assert manifest["dataset"]["genotype_encoding"] == "cran-numeric-dosage"
    assert set(manifest["checksums"]["files"]) == {
        path.name for path in (tmp_path / "dataset").glob("*.tsv.gz")
    }
    assert set(manifest["checksums"]["builder_sources"]) == {
        "soynam_cran.py",
        "build_soynam_canonical.R",
    }
    serialised = json.dumps(manifest)
    assert str(tmp_path) not in serialised


def test_output_reloads_through_the_loader_as_numeric_dosage(tmp_path):
    build(tmp_path)
    dataset = soynam_data.load_soynam_dataset(tmp_path / "dataset")
    assert dataset.genotypes.shape == (4, len(MARKERS))
    assert set(dataset.marker_names) == set(MARKERS)
    finite = dataset.genotypes[~np.isnan(dataset.genotypes)]
    assert set(finite.tolist()) <= {-1.0, 0.0, 1.0}


def test_duplicate_sample_is_rejected(tmp_path):
    intermediates = write_intermediates(
        tmp_path / "intermediates",
        samples=(("S0001", 2), ("S0001", 2), ("S0003", 3)),
    )
    with pytest.raises(soynam_cran.BuildError, match="duplicate sample"):
        build(tmp_path, intermediates=intermediates)


def test_duplicate_marker_is_rejected(tmp_path):
    intermediates = write_intermediates(
        tmp_path / "intermediates", markers=["Gm01_100_A_C", "Gm01_100_A_C"]
    )
    with pytest.raises(soynam_cran.BuildError, match="duplicate marker"):
        build(tmp_path, intermediates=intermediates)


def test_strain_mapped_to_two_families_is_rejected(tmp_path):
    # The same strain listed under two families is reported as the family
    # conflict it is, not as a plain duplicate ID.
    intermediates = write_intermediates(
        tmp_path / "intermediates",
        samples=(("S0001", 2), ("S0001", 3), ("S0003", 3)),
    )
    with pytest.raises(soynam_cran.BuildError, match="more than one family"):
        build(tmp_path, intermediates=intermediates)


def test_sample_list_disagreement_between_intermediates_is_rejected(tmp_path):
    directory = write_intermediates(tmp_path / "intermediates")
    lines = (directory / "genotype.tsv").read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    header[-1] = "S9999"
    lines[0] = "\t".join(header)
    (directory / "genotype.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(soynam_cran.BuildError, match="disagree on the sample list"):
        build(tmp_path, intermediates=directory)


def test_unexpected_genotype_token_is_rejected(tmp_path):
    intermediates = write_intermediates(
        tmp_path / "intermediates",
        genotypes=[["0", "1", "2", "A"], ["0", "1", "2", "NA"], ["0", "0", "0", "0"]],
    )
    with pytest.raises(soynam_cran.BuildError, match="unexpected genotype values"):
        build(tmp_path, intermediates=intermediates)


def test_reference_hashes_are_enforced(tmp_path):
    expectations = soynam_cran.CanonicalExpectations(marker_list_sha256="0" * 64)
    with pytest.raises(soynam_cran.BuildError, match="marker ID list"):
        build(tmp_path, expectations=expectations)


def test_dataset_manifest_discovery(tmp_path):
    empty = tmp_path / "no-manifest"
    empty.mkdir()
    assert soynam_cran.describe_dataset_manifest(empty) is None

    build(tmp_path)
    described = soynam_cran.describe_dataset_manifest(tmp_path / "dataset")
    assert described["filename"] == soynam_cran.MANIFEST_FILENAME
    assert described["sha256"] == soynam_cran.sha256_file(
        tmp_path / "dataset" / soynam_cran.MANIFEST_FILENAME
    )

    duplicate = tmp_path / "dataset" / "soynam-cran-9.9.9-manifest.json"
    duplicate.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="multiple canonical dataset manifests"):
        soynam_cran.describe_dataset_manifest(tmp_path / "dataset")
