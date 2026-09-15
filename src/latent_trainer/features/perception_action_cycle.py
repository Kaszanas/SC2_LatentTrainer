"""Perception-Action Cycle (PAC) extraction from StarCraft II game events.

A PAC is one *fixation* of the player's screen together with the actions issued
while it was held.  The idea comes from the expertise literature: a player
moves the camera to look somewhere (perception), acts on what they see
(action), then moves on.  How often that cycle turns, how quickly action
follows perception, and how much gets done per look are all measures of skill
that raw APM cannot express.

Four per-player statistics follow the conventions established by PACAnalyzer_:

``ppm``
    Perception-action cycles per minute.
``pal``
    PAC action latency -- seconds from establishing a fixation to the first
    action taken in it.  Low is sharp.
``app``
    Actions per PAC -- how much is accomplished per look.
``gap``
    Seconds from the last action of one fixation to the establishment of the
    next.  High means dithering between looks.

.. _PACAnalyzer: https://github.com/Reithan/PACAnalyzer

This module is deliberately free of third-party imports and of any dependency
on the surrounding project.  It identifies event kinds by class name, so it
works on any object exposing ``loop``, ``userid`` and (for camera events) a
``target`` with ``x``/``y``.  That keeps it portable enough to be lifted into
``sc2_datasets`` as-is.

Note on units: PACAnalyzer works in sc2reader frames at 16/s.  StarCraft II
replays in SC2EGSet are recorded at the "Faster" speed of 22.4 game loops per
second, so every duration here is converted through ``loops_per_second``
rather than assuming either constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# SC2 replays are recorded at "Faster": 22.4 game loops per real second.
GAME_LOOPS_PER_SECOND = 22.4

# Defaults from PACAnalyzer's ``PACInfo(6, 0.2)``.
DEFAULT_DISPLACEMENT_THRESHOLD = 6.0
DEFAULT_DURATION_THRESHOLD_SECONDS = 0.2

# Event classes that count as a player *action* inside a fixation.  Camera
# movement is perception and is handled separately; chat, options and leave
# events are not actions at all.
ACTION_EVENT_NAMES: frozenset[str] = frozenset(
    {
        "Cmd",
        "CmdUpdateTargetPoint",
        "CmdUpdateTargetUnit",
        "ControlGroupUpdate",
        "SelectionDelta",
    }
)
CAMERA_EVENT_NAME = "CameraUpdate"


@dataclass
class PerceptionActionCycle:
    """One camera fixation and the actions issued while it was held."""

    start_loop: int
    """Game loop at which the fixation was established."""

    x: float
    """Anchor x of the fixation (the camera position that opened it)."""

    y: float
    """Anchor y of the fixation."""

    camera_loops: list[int] = field(default_factory=list)
    """Loops of every camera update that fell within this fixation."""

    action_loops: list[int] = field(default_factory=list)
    """Loops of every action issued while the fixation was held."""

    end_loop: int | None = None
    """Loop at which the next fixation began; ``None`` for the last cycle.

    Dwell time is measured to the *next* fixation rather than to the last
    camera update inside this one, because a fixation held perfectly still
    produces no further camera events -- measuring to the last one would score
    the steadiest looks as zero-length.
    """

    @property
    def duration_loops(self) -> int:
        """How long the camera dwelt here, in game loops."""
        if self.end_loop is None:
            last = max(
                self.camera_loops + self.action_loops,
                default=self.start_loop,
            )
            return last - self.start_loop
        return self.end_loop - self.start_loop

    @property
    def n_actions(self) -> int:
        return len(self.action_loops)

    @property
    def latency_loops(self) -> int | None:
        """Loops from establishing the fixation to the first action in it."""
        if not self.action_loops:
            return None
        return self.action_loops[0] - self.start_loop


@dataclass
class PerceptionActionCycleStats:
    """Per-player summary of a list of :class:`PerceptionActionCycle`."""

    n_cycles: int
    ppm: float
    """Perception-action cycles per minute."""
    pal: float
    """Mean seconds from fixation to first action."""
    app: float
    """Mean actions per cycle."""
    gap: float
    """Mean seconds from a cycle's last action to the next fixation."""


def _displacement(x_a: float, y_a: float, x_b: float, y_b: float) -> float:
    """Combined x and y shift, matching PACAnalyzer's dispersion measure.

    This is the L1 (Manhattan) distance rather than Euclidean, which is what
    "max combined x & y shift" describes and what the reference implementation
    thresholds against.
    """
    return abs(x_a - x_b) + abs(y_a - y_b)


def extract_perception_action_cycles(
    game_events: list,
    user_id: int,
    displacement_threshold: float = DEFAULT_DISPLACEMENT_THRESHOLD,
    duration_threshold_seconds: float = DEFAULT_DURATION_THRESHOLD_SECONDS,
    loops_per_second: float = GAME_LOOPS_PER_SECOND,
) -> list[PerceptionActionCycle]:
    """Segment one player's game events into perception-action cycles.

    Parameters
    ----------
    game_events:
        The replay's game events.  Events belonging to other users are ignored,
        so the full list can be passed once per player.
    user_id:
        The ``userid`` to extract for.  Note this is the *user* id, not the
        ``playerID``; the two differ and are related through the replay's
        ``toonPlayerDescMap``.
    displacement_threshold:
        Maximum combined x+y camera shift that still counts as the same
        fixation.  Beyond it, a new cycle starts.
    duration_threshold_seconds:
        Minimum dwell for a fixation to stand on its own.  Shorter ones are
        treated as camera noise -- a flick across the map on the way somewhere
        else -- and are merged into the fixation that preceded them rather than
        discarded, so their actions are not lost.
    loops_per_second:
        Game loops per real second, used to interpret
        ``duration_threshold_seconds``.

    Returns
    -------
    list[PerceptionActionCycle]
        Cycles in chronological order.  Empty if the player produced no camera
        events at all.
    """
    # Stable sort: events sharing a loop keep their original relative order,
    # which preserves the true causal order of a camera move and the action
    # that immediately followed it.
    relevant = sorted(
        (event for event in game_events if getattr(event, "userid", None) == user_id),
        key=lambda event: event.loop,
    )

    cycles: list[PerceptionActionCycle] = []

    for event in relevant:
        name = type(event).__name__

        if name == CAMERA_EVENT_NAME:
            target = getattr(event, "target", None)
            if target is None:
                continue

            if (
                cycles
                and _displacement(target.x, target.y, cycles[-1].x, cycles[-1].y)
                <= displacement_threshold
            ):
                # Still looking at the same place.
                cycles[-1].camera_loops.append(event.loop)
                continue

            if cycles:
                cycles[-1].end_loop = event.loop
            cycles.append(
                PerceptionActionCycle(
                    start_loop=event.loop,
                    x=target.x,
                    y=target.y,
                    camera_loops=[event.loop],
                )
            )

        elif name in ACTION_EVENT_NAMES and cycles:
            # Actions before the first camera event have no fixation to belong
            # to and are dropped; in practice this is the opening few loops.
            cycles[-1].action_loops.append(event.loop)

    return _merge_short_fixations(
        cycles=cycles,
        minimum_duration_loops=duration_threshold_seconds * loops_per_second,
    )


def _merge_short_fixations(
    cycles: list[PerceptionActionCycle],
    minimum_duration_loops: float,
) -> list[PerceptionActionCycle]:
    """Fold fixations shorter than the threshold into the preceding one."""
    if minimum_duration_loops <= 0:
        return cycles

    merged: list[PerceptionActionCycle] = []
    for cycle in cycles:
        too_brief = cycle.duration_loops < minimum_duration_loops
        if too_brief and merged:
            previous = merged[-1]
            previous.camera_loops.extend(cycle.camera_loops)
            previous.action_loops.extend(cycle.action_loops)
            previous.end_loop = cycle.end_loop
            continue
        merged.append(cycle)

    return merged


def summarize_perception_action_cycles(
    cycles: list[PerceptionActionCycle],
    total_loops: int,
    loops_per_second: float = GAME_LOOPS_PER_SECOND,
) -> PerceptionActionCycleStats:
    """Reduce a player's cycles to the four PAC statistics.

    Cycles containing no actions are excluded from ``pal`` and from ``gap``
    (they have no first or last action to measure from) but still count toward
    ``ppm`` and ``app`` -- looking somewhere and doing nothing is itself part
    of the behaviour being measured.

    Returns zeroed statistics rather than raising when there is nothing to
    summarise, so a player who never moved the camera does not abort a batch.
    """
    n_cycles = len(cycles)
    if n_cycles == 0 or total_loops <= 0:
        return PerceptionActionCycleStats(
            n_cycles=0, ppm=0.0, pal=0.0, app=0.0, gap=0.0
        )

    minutes = total_loops / loops_per_second / 60.0
    ppm = n_cycles / minutes if minutes > 0 else 0.0

    latencies = [
        cycle.latency_loops for cycle in cycles if cycle.latency_loops is not None
    ]
    pal = (sum(latencies) / len(latencies) / loops_per_second) if latencies else 0.0

    app = sum(cycle.n_actions for cycle in cycles) / n_cycles

    gaps = [
        cycles[index + 1].start_loop - cycles[index].action_loops[-1]
        for index in range(n_cycles - 1)
        if cycles[index].action_loops
    ]
    gap = (sum(gaps) / len(gaps) / loops_per_second) if gaps else 0.0

    return PerceptionActionCycleStats(
        n_cycles=n_cycles, ppm=ppm, pal=pal, app=app, gap=gap
    )


class PACAlgorithm:
    """Convenience wrapper binding a threshold configuration to the functions.

    The module-level functions are the interface; this exists so a
    configuration can be set once and reused across a dataset, and to fill the
    placeholder of the same name in ``sc2_datasets``.
    """

    def __init__(
        self,
        displacement_threshold: float = DEFAULT_DISPLACEMENT_THRESHOLD,
        duration_threshold_seconds: float = DEFAULT_DURATION_THRESHOLD_SECONDS,
        loops_per_second: float = GAME_LOOPS_PER_SECOND,
    ) -> None:
        self.displacement_threshold = displacement_threshold
        self.duration_threshold_seconds = duration_threshold_seconds
        self.loops_per_second = loops_per_second

    def extract(self, game_events: list, user_id: int) -> list[PerceptionActionCycle]:
        """Segment one player's events into cycles using this configuration."""
        return extract_perception_action_cycles(
            game_events=game_events,
            user_id=user_id,
            displacement_threshold=self.displacement_threshold,
            duration_threshold_seconds=self.duration_threshold_seconds,
            loops_per_second=self.loops_per_second,
        )

    def summarize(
        self, cycles: list[PerceptionActionCycle], total_loops: int
    ) -> PerceptionActionCycleStats:
        """Reduce cycles to the four PAC statistics using this configuration."""
        return summarize_perception_action_cycles(
            cycles=cycles,
            total_loops=total_loops,
            loops_per_second=self.loops_per_second,
        )
