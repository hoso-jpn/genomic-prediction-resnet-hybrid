import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import soynam_data


class SoynamDataTest(unittest.TestCase):
    def test_raw_loader_excludes_parent_and_preserves_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            family_id = "Parent-A_NAM01"
            phenotype = pd.DataFrame(
                {
                    "Corrected Strain": ["Parent-A", "RIL-1", "RIL-2"],
                    "Yld (kg/ha)": [400.0, 500.0, 600.0],
                }
            )
            genotype = pd.DataFrame(
                {
                    "marker": ["m1", "m2"],
                    "Parent-A": ["A", "B"],
                    "RIL-1": ["-", "A/B"],
                    "RIL-2": ["B", "A"],
                }
            )
            phenotype.to_csv(
                data_dir / f"{family_id}{soynam_data.PHENOTYPE_SUFFIX}",
                sep="\t",
                index=False,
                compression="gzip",
            )
            genotype.to_csv(
                data_dir / f"{family_id}_4312{soynam_data.GENOTYPE_SUFFIX}",
                sep="\t",
                index=False,
                compression="gzip",
            )

            dataset = soynam_data.load_soynam_dataset(data_dir)

        np.testing.assert_array_equal(
            dataset.sample_names, np.array(["RIL-1", "RIL-2"])
        )
        np.testing.assert_allclose(dataset.phenotypes, [500.0, 600.0])
        self.assertTrue(np.isnan(dataset.genotypes[0, 0]))
        np.testing.assert_allclose(dataset.genotypes[0, 1], 0.0)
        np.testing.assert_allclose(dataset.genotypes[1], [1.0, -1.0])

    def test_pairing_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            pd.DataFrame(
                {
                    "Corrected Strain": ["RIL-1"],
                    "Yld (kg/ha)": [500.0],
                }
            ).to_csv(
                data_dir / f"A_NAM01{soynam_data.PHENOTYPE_SUFFIX}",
                sep="\t",
                index=False,
                compression="gzip",
            )
            with self.assertRaisesRegex(ValueError, "pairing is invalid"):
                soynam_data.load_soynam_dataset(data_dir)


if __name__ == "__main__":
    unittest.main()
