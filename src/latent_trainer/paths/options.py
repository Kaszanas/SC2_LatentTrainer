"""Shared Click options for all path-charting CLI commands."""

from pathlib import Path

import click

PATH_CHARTING_CLI_COMMON_OPTIONS = [
    click.option(
        "--model_path",
        help="Filename of the trained model.",
        type=click.Path(
            exists=True,
            dir_okay=False,
            path_type=Path,
            resolve_path=True,
        ),
    ),
    click.option(
        "--dataset_filename",
        default="cached_dataset_rich.pt",
        show_default=True,
        help="Filename of the cached dataset placed in the DATA_DIR (set in settings.py).",
        type=str,
    ),
    click.option(
        "--sample_idx",
        type=int,
        default=None,
        help="Index of the game to analyse. If omitted, a game is chosen at random.",
    ),
    click.option(
        "--n_steps",
        type=int,
        default=20,
        show_default=True,
        help="Number of waypoints along the path.",
    ),
    click.option(
        "--top_k",
        type=int,
        default=10,
        show_default=True,
        help="Number of features requiring improvement to display.",
    ),
]


def global_options(fn):
    """Decorator that attaches all common path-charting options to a command."""
    for option in reversed(PATH_CHARTING_CLI_COMMON_OPTIONS):
        fn = option(fn)
    return fn
