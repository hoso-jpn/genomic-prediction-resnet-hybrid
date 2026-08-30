"""Shared Weights & Biases mode handling for every entry point.

External logging is opt-in: the CLI selection is the only thing that
decides the mode, so no ambient environment variable can turn a run that
asked for ``disabled`` or ``offline`` into one that transmits data. This
module holds the single implementation of that rule; verified baselines
and legacy scripts both build their logger through it instead of calling
``wandb.init`` directly.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

WANDB_MODES = ("disabled", "offline", "online")
DEFAULT_WANDB_MODE = "disabled"
WANDB_MODE_HELP = (
    "Weights & Biases mode. 'disabled' (default) never initializes W&B, "
    "'offline' writes local W&B files only, and 'online' is the only "
    "setting that sends data to the service"
)


class StaticConfig(dict):
    """Run configuration used when no W&B run supplies one.

    Mirrors the small part of ``wandb.config`` the scripts rely on
    (attribute access plus ``get``) so a disabled run reads exactly the
    same configuration keys as an offline or online one.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


class NullRunLogger:
    """Logger used when external experiment logging is disabled.

    Instantiating this never imports or initializes W&B, so a disabled run
    performs no external logging setup at all rather than initializing a
    client and then suppressing its output.
    """

    mode = "disabled"

    def run_config(self, defaults: dict[str, Any]) -> StaticConfig:
        return StaticConfig(defaults)

    def log(self, payload: dict[str, Any]) -> None:
        return None

    def finish(self) -> None:
        return None


class WandbRunLogger:
    """Logger backed by an already-initialized W&B run."""

    def __init__(self, module: Any, mode: str) -> None:
        self._module = module
        self.mode = mode

    def run_config(self, defaults: dict[str, Any]) -> Any:
        """Return W&B's own config, which a sweep agent may have overridden."""
        return self._module.config

    def log(self, payload: dict[str, Any]) -> None:
        self._module.log(payload)

    def finish(self) -> None:
        self._module.finish()


def add_wandb_mode_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--wandb-mode`` option to a CLI parser."""
    parser.add_argument(
        "--wandb-mode",
        choices=WANDB_MODES,
        default=DEFAULT_WANDB_MODE,
        help=WANDB_MODE_HELP,
    )


def create_run_logger(
    mode: str,
    *,
    project: str,
    job_type: str | None = None,
    name: str | None = None,
    config: dict[str, Any] | None = None,
) -> NullRunLogger | WandbRunLogger:
    """Create the run logger for ``mode``, importing W&B only when used.

    The caller's selection is authoritative: ``WANDB_MODE`` is overwritten
    with the resolved mode before ``wandb.init``, and the same value is
    passed to ``wandb.init`` explicitly, so an ambient environment variable
    can neither escalate ``offline`` to ``online`` nor re-enable a run the
    caller asked to disable. ``offline`` keeps W&B's local run directory
    but sends nothing to the service.
    """
    if mode not in WANDB_MODES:
        raise ValueError(f"unknown W&B mode: {mode!r}")
    if mode == "disabled":
        return NullRunLogger()

    import wandb

    os.environ["WANDB_MODE"] = mode
    keyword_arguments: dict[str, Any] = {"project": project, "mode": mode}
    if job_type is not None:
        keyword_arguments["job_type"] = job_type
    if name is not None:
        keyword_arguments["name"] = name
    if config is not None:
        keyword_arguments["config"] = config
    wandb.init(**keyword_arguments)
    return WandbRunLogger(wandb, mode)
