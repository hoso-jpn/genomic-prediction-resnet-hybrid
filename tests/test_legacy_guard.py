"""Tests for the legacy opt-in gate and the shared W&B mode rules."""

import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import external_logging
import legacy_guard

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_SCRIPTS = ("preprocess.py", "main.py", "train_gnn.py")


def _run_script(
    script_name: str, arguments: list[str], *, cwd: Path
) -> subprocess.CompletedProcess[str]:
    """Run a legacy script from a scratch directory with no W&B credentials."""
    environment = os.environ.copy()
    environment.pop("WANDB_API_KEY", None)
    # An ambient online setting must not weaken the gate or turn logging on.
    environment["WANDB_MODE"] = "online"
    return subprocess.run(
        [sys.executable, "-u", str(REPO_ROOT / script_name), *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )


class LegacyGuardUnitTest(unittest.TestCase):
    def test_missing_flag_exits_before_doing_anything(self) -> None:
        for script_name in LEGACY_SCRIPTS:
            with self.subTest(script=script_name):
                with self.assertRaises(SystemExit) as raised:
                    legacy_guard.require_opt_in(script_name, "description", [])
                self.assertEqual(raised.exception.code, 2)

    def test_flag_allows_the_run_and_leaves_logging_disabled(self) -> None:
        args = legacy_guard.require_opt_in("main.py", "description", ["--allow-legacy"])

        self.assertTrue(args.allow_legacy)
        self.assertEqual(args.wandb_mode, "disabled")

    def test_legacy_permission_and_wandb_permission_are_separate(self) -> None:
        args = legacy_guard.require_opt_in(
            "main.py", "description", ["--allow-legacy", "--wandb-mode", "offline"]
        )

        self.assertEqual(args.wandb_mode, "offline")

    def test_sweep_hyperparameter_arguments_are_passed_through(self) -> None:
        # A W&B sweep agent appends its own arguments; they must neither be
        # rejected nor satisfy the gate.
        args = legacy_guard.require_opt_in(
            "main.py",
            "description",
            ["--allow-legacy", "--lr=0.001", "--hidden_dim=128"],
        )
        self.assertTrue(args.allow_legacy)

        with self.assertRaises(SystemExit):
            legacy_guard.require_opt_in(
                "main.py", "description", ["--lr=0.001", "--hidden_dim=128"]
            )


class ExternalLoggingTest(unittest.TestCase):
    @staticmethod
    def _fake_wandb() -> types.ModuleType:
        module = types.ModuleType("wandb")
        module.calls = []
        module.config = {"from": "wandb"}
        module.init = lambda **kwargs: module.calls.append(("init", kwargs))
        module.log = lambda payload: module.calls.append(("log", payload))
        module.finish = lambda: module.calls.append(("finish", None))
        return module

    def test_disabled_logger_uses_the_static_config(self) -> None:
        fake = self._fake_wandb()
        with mock.patch.dict(sys.modules, {"wandb": fake}):
            logger = external_logging.create_run_logger(
                "disabled", project="p", config={"lr": 0.1}
            )
            config = logger.run_config({"lr": 0.1, "seed": 42})

        self.assertEqual(fake.calls, [])
        self.assertEqual(config.lr, 0.1)
        self.assertEqual(config.get("seed"), 42)
        self.assertEqual(config.get("missing", "fallback"), "fallback")
        self.assertRaises(AttributeError, lambda: config.missing)

    def test_offline_logger_reads_the_wandb_config(self) -> None:
        fake = self._fake_wandb()
        with (
            mock.patch.dict(sys.modules, {"wandb": fake}),
            mock.patch.dict(os.environ, {"WANDB_MODE": "online"}, clear=False),
        ):
            logger = external_logging.create_run_logger(
                "offline", project="p", config={"lr": 0.1}
            )

            self.assertEqual(os.environ["WANDB_MODE"], "offline")
            self.assertEqual(logger.run_config({"lr": 0.1}), {"from": "wandb"})

        self.assertEqual(fake.calls[0][1]["mode"], "offline")
        self.assertEqual(fake.calls[0][1]["project"], "p")


class LegacyScriptEntryPointTest(unittest.TestCase):
    def test_scripts_stop_without_the_flag_and_create_nothing(self) -> None:
        for script_name in LEGACY_SCRIPTS:
            with (
                self.subTest(script=script_name),
                tempfile.TemporaryDirectory() as directory,
            ):
                workdir = Path(directory)
                result = _run_script(script_name, [], cwd=workdir)

                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("--allow-legacy", result.stderr)
                self.assertIn("gblup_baseline.py", result.stderr)
                self.assertIn("resnet_baseline.py", result.stderr)
                # Nothing was read, written, or logged before the exit.
                self.assertEqual(list(workdir.iterdir()), [])

    def test_preprocess_proceeds_past_the_gate_when_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory)
            (workdir / "data").mkdir()
            result = _run_script("preprocess.py", ["--allow-legacy"], cwd=workdir)

            # The gate is passed (banner printed, no refusal message), and the
            # run then fails on its own missing input rather than on the flag.
            self.assertIn("[EXPERIMENTAL]", result.stdout)
            self.assertNotIn("Re-run it with --allow-legacy", result.stderr)
            self.assertNotEqual(result.returncode, 2)
            self.assertFalse((workdir / "wandb").exists())


class LegacyFamilyIdTest(unittest.TestCase):
    def test_missing_family_id_fails_instead_of_falling_back(self) -> None:
        import pandas as pd

        import main as legacy_main

        frame = pd.DataFrame({"Yld (kg/ha)": [1.0, 2.0]})
        with self.assertRaisesRegex(RuntimeError, "family_id"):
            legacy_main.require_family_ids(frame)

    def test_family_ids_are_returned_when_present(self) -> None:
        import pandas as pd

        import main as legacy_main

        frame = pd.DataFrame({"Yld (kg/ha)": [1.0, 2.0], "family_id": ["A", "B"]})
        self.assertEqual(list(legacy_main.require_family_ids(frame)), ["A", "B"])


if __name__ == "__main__":
    unittest.main()
