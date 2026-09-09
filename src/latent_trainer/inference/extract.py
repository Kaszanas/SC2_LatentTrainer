"""Extract one .SC2Replay file into an SC2ReplayData via dockerized SC2InfoExtractorGo.

Usage (once the image is pulled):

    docker pull kaszanas/sc2infoextractorgo:dev

Container interface, verified against ``docker run kaszanas/sc2infoextractorgo:dev -help``:
mount an input directory (containing .SC2Replay files), an output
directory, a logs directory, and (for repeated calls) a dependency
directory for cached map/replay dependencies. Pass ``-single_json_output``
to get one JSON file containing an array of processed replays -- for a
single input replay, that array has exactly one element -- rather than the
tool's default zip-package-of-many-replays output.

If this module runs inside its own container (e.g. as part of the
`serving/` frontend), mount the host's Docker socket
(`-v /var/run/docker.sock:/var/run/docker.sock`) so the `docker run` calls
below can still spin up the extractor as a sibling container.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from sc2_datasets.replay_data.sc2_replay_data import SC2ReplayData

logger = logging.getLogger(__name__)

DEFAULT_DOCKER_IMAGE = "kaszanas/sc2infoextractorgo:dev"
DEFAULT_DEPENDENCY_CACHE_DIR = Path.home() / ".cache" / "latent_trainer_serve" / "sc2_extractor_dependencies"
# Where per-run input/output/logs get copied on failure (or always, if
# keep_run_dir=True), since the ephemeral tempfile.TemporaryDirectory is
# gone by the time an exception is inspected -- full visibility into what
# the container actually did, not a black box.
DEFAULT_DEBUG_DIR = Path.home() / ".cache" / "latent_trainer_serve" / "last_extraction_run"


def _read_failure_reasons(*search_dirs: Path) -> list[str]:
    """Parse SC2InfoExtractorGo's processed_failed_*.log files for real reasons.

    These land under the logs directory, not the output directory -- search
    both since that placement isn't documented anywhere.
    """
    reasons = []
    seen: set[Path] = set()
    for search_dir in search_dirs:
        for log_path in sorted(search_dir.rglob("processed_failed_*.log")):
            if log_path in seen:
                continue
            seen.add(log_path)
            try:
                data = json.loads(log_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            for entry in data.get("failedToProcess", []):
                reasons.append(f"{entry.get('fileName')}: {entry.get('reason')}")
    return reasons


def _stream_subprocess(cmd: list[str], timeout: int) -> tuple[int, list[str]]:
    """Run *cmd*, printing each line of its combined output live as it arrives.

    Returns (returncode, all_lines) -- nothing is hidden until the end the
    way subprocess.run(capture_output=True) does.
    """
    lines: list[str] = []
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            print(f"[SC2InfoExtractorGo] {line}", flush=True)
            lines.append(line)
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise
    return returncode, lines


class ReplayExtractionError(RuntimeError):
    """Raised when SC2InfoExtractorGo fails to produce a parseable replay JSON."""


def extract_replay(
    replay_path: Path,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    extractor_binary: Path | None = None,
    perform_integrity_checks: bool = False,
    perform_validity_checks: bool = False,
    dependency_cache_dir: Path | None = None,
    skip_dependency_download: bool = False,
    timeout: int = 300,
    debug_dir: Path | None = None,
    keep_run_dir: bool = False,
) -> SC2ReplayData:
    """Run SC2InfoExtractorGo on one replay and parse the result.

    Copies *replay_path* into a fresh temporary input directory, runs the
    extractor against it with ``-single_json_output``, then loads the
    resulting single-element JSON array into an :class:`SC2ReplayData`.

    Two ways to run the extractor:

    - ``extractor_binary`` set -- run that local binary directly (no Docker
      involved at all). This is what a containerized server should use: bake
      the extractor's binary into the same image at build time (multi-stage
      ``COPY --from=kaszanas/sc2infoextractorgo:dev /app/SC2InfoExtractorGo``)
      instead of needing Docker-out-of-Docker (a docker CLI + the host socket
      mounted in) just to spin up a sibling container per replay.
    - ``extractor_binary`` is ``None`` (default) -- fall back to ``docker
      run <docker_image> ...``, for local/dev use without building a custom
      image.

    All container output is printed live (prefixed ``[SC2InfoExtractorGo]``)
    as it happens -- this is not a black box. On any failure (or always, if
    ``keep_run_dir=True``), the run's input/output/logs are copied to
    *debug_dir* (default ``~/.cache/latent_trainer_serve/last_extraction_run``)
    before the temp directory is cleaned up, so you can inspect exactly what
    the container saw and wrote, including its own ``main_log.log``.

    Parameters
    ----------
    dependency_cache_dir:
        Persisted across calls (unlike input/output/logs, which are
        per-call temp dirs) so repeated extractions don't re-download the
        same map dependencies. Defaults to
        ``~/.cache/latent_trainer_serve/sc2_extractor_dependencies``.
    skip_dependency_download:
        Pass True once dependencies are already cached there, to avoid
        needing network access on every call.
    keep_run_dir:
        Copy the run directory to *debug_dir* even on success.

    Raises
    ------
    FileNotFoundError
        If *replay_path* doesn't exist.
    ReplayExtractionError
        If the container fails, times out, or produces no JSON output --
        the message includes SC2InfoExtractorGo's own reported reason
        (parsed from its processed_failed_*.log) when available, and the
        path the run's artifacts were copied to for further inspection.
    """
    replay_path = Path(replay_path).resolve()
    if not replay_path.is_file():
        raise FileNotFoundError(f"Replay file not found: {replay_path}")

    dependency_cache_dir = (
        Path(dependency_cache_dir)
        if dependency_cache_dir is not None
        else DEFAULT_DEPENDENCY_CACHE_DIR
    )
    dependency_cache_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(debug_dir) if debug_dir is not None else DEFAULT_DEBUG_DIR

    with tempfile.TemporaryDirectory(prefix="sc2_extract_") as tmp:
        tmp_path = Path(tmp)
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        logs_dir = tmp_path / "logs"
        for d in (input_dir, output_dir, logs_dir):
            d.mkdir(parents=True, exist_ok=True)

        def preserve_run_dir() -> Path:
            if debug_dir.exists():
                shutil.rmtree(debug_dir)
            shutil.copytree(tmp_path, debug_dir)
            return debug_dir

        # SC2InfoExtractorGo's -input/-output/-log_dir/-dependency_directory
        # matching is case-sensitive on the .SC2Replay extension, and its
        # log path is built via naive string concatenation (logPath +
        # "main_log.log", no separator) -- always normalize the extension
        # casing and pass directory flags with a trailing slash to avoid
        # both, confirmed against the tool's own source.
        dest_name = replay_path.stem + ".SC2Replay"
        shutil.copy2(replay_path, input_dir / dest_name)

        if extractor_binary is not None:
            # Direct invocation -- paths on the host filesystem, no mounts.
            cmd = [
                str(extractor_binary),
                "-input", f"{input_dir}/",
                "-output", f"{output_dir}/",
                "-log_dir", f"{logs_dir}/",
                "-dependency_directory", f"{dependency_cache_dir}/",
                "-single_json_output",
                "-number_of_packages", "0",
                "-log_level", "6",
            ]
        else:
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{input_dir}:/app/replays/input",
                "-v", f"{output_dir}:/app/replays/output",
                "-v", f"{logs_dir}:/app/logs",
                "-v", f"{dependency_cache_dir}:/app/dependencies",
                docker_image,
                "-input", "/app/replays/input/",
                "-output", "/app/replays/output/",
                "-log_dir", "/app/logs/",
                "-dependency_directory", "/app/dependencies/",
                "-single_json_output",
                "-number_of_packages", "0",
                "-log_level", "6",
            ]
        if perform_integrity_checks:
            cmd.append("-perform_integrity_checks")
        if perform_validity_checks:
            cmd.append("-perform_validity_checks")
        if skip_dependency_download:
            cmd.append("-skip_dependency_download")

        logger.info("Running SC2InfoExtractorGo on %s: %s", replay_path.name, " ".join(cmd))
        try:
            returncode, output_lines = _stream_subprocess(cmd, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            saved_to = preserve_run_dir()
            raise ReplayExtractionError(
                f"SC2InfoExtractorGo timed out after {timeout}s on "
                f"{replay_path.name}. Run artifacts saved to {saved_to}"
            ) from e
        except FileNotFoundError as e:
            missing = str(extractor_binary) if extractor_binary is not None else "docker"
            raise ReplayExtractionError(
                f"Could not run {missing!r} -- "
                + (
                    "check the extractor binary was baked into the image at that path."
                    if extractor_binary is not None
                    else "is Docker installed and on PATH "
                    "(or the Docker socket mounted, if running in a container)?"
                )
            ) from e

        failure_reasons = _read_failure_reasons(output_dir, logs_dir)

        if returncode != 0:
            saved_to = preserve_run_dir()
            reason_str = "; ".join(failure_reasons) or "(no processed_failed_*.log written)"
            raise ReplayExtractionError(
                f"SC2InfoExtractorGo failed (exit {returncode}) on "
                f"{replay_path.name}. Reason: {reason_str}. "
                f"Run artifacts (including main_log.log) saved to {saved_to}"
            )

        json_files = list(output_dir.rglob("*.json"))
        if not json_files:
            saved_to = preserve_run_dir()
            raise ReplayExtractionError(
                f"SC2InfoExtractorGo produced no JSON output for {replay_path.name}. "
                f"Run artifacts saved to {saved_to}"
            )
        if len(json_files) > 1:
            logger.warning(
                "Expected exactly one output JSON with -single_json_output, "
                "found %d; using %s",
                len(json_files),
                json_files[0],
            )

        with open(json_files[0], "r", encoding="utf-8") as f:
            loaded_data = json.load(f)

        # -single_json_output produces a JSON array of processed replays;
        # exactly one element, since we only fed it one replay file.
        if isinstance(loaded_data, list):
            if not loaded_data:
                saved_to = preserve_run_dir()
                reason_str = "; ".join(failure_reasons) or "(no processed_failed_*.log written)"
                raise ReplayExtractionError(
                    f"SC2InfoExtractorGo's output JSON was an empty array for "
                    f"{replay_path.name}. Reason: {reason_str}. "
                    f"Run artifacts (including main_log.log) saved to {saved_to}"
                )
            loaded_data = loaded_data[0]

        if keep_run_dir:
            saved_to = preserve_run_dir()
            logger.info("Run artifacts saved to %s", saved_to)

        return SC2ReplayData.from_dict(
            loaded_data=loaded_data,
            replay_filepath=str(replay_path),
        )
