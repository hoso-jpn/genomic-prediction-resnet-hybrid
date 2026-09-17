"""Opt-in end-to-end build against the real CRAN release.

This test needs the published SoyNAM tarball and a pinned R environment, so
it is skipped unless both are pointed at explicitly:

    GPRH_SOYNAM_TARBALL=/path/to/SoyNAM_1.6.2.tar.gz \
    GPRH_RSCRIPT=/path/to/Rscript \
    uv run pytest -q tests/test_soynam_cran_integration.py

CI never sets those variables, so the default run neither downloads from CRAN
nor requires R. A skipped run is not evidence that the builder works.
"""

import os
import shutil

import pytest

import soynam_cran
import soynam_data

TARBALL = os.environ.get("GPRH_SOYNAM_TARBALL")
RSCRIPT = os.environ.get("GPRH_RSCRIPT", "Rscript")

pytestmark = pytest.mark.skipif(
    not TARBALL or shutil.which(RSCRIPT) is None,
    reason="set GPRH_SOYNAM_TARBALL and GPRH_RSCRIPT to run the CRAN build",
)


def test_canonical_build_matches_the_audited_reference(tmp_path):
    manifest = soynam_cran.build(
        output_dir=tmp_path / "dataset",
        source_tarball=TARBALL,
        rscript=RSCRIPT,
    )

    dataset = manifest["dataset"]
    assert dataset["samples"] == soynam_cran.CRAN_EXPECTATIONS.samples
    assert dataset["families"] == soynam_cran.CRAN_EXPECTATIONS.families
    assert dataset["markers"] == soynam_cran.CRAN_EXPECTATIONS.markers
    assert dataset["genotype_encoding"] == soynam_cran.GENOTYPE_ENCODING_NAME

    checksums = manifest["checksums"]
    assert (
        checksums["marker_id_list_sha256"]
        == soynam_cran.CRAN_EXPECTATIONS.marker_list_sha256
    )
    assert (
        checksums["sample_id_list_sha256"]
        == soynam_cran.CRAN_EXPECTATIONS.sample_list_sha256
    )
    assert (
        checksums["joined_phenotype_sha256"]
        == soynam_cran.CRAN_EXPECTATIONS.joined_phenotype_sha256
    )

    loaded = soynam_data.load_soynam_dataset(tmp_path / "dataset")
    assert loaded.genotypes.shape == (dataset["samples"], dataset["markers"])
    assert len(set(loaded.family_ids.tolist())) == dataset["families"]
