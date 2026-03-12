"""Find touches and passes from the ball path.

Touch: a frame where the ball's velocity changes sharply (a kick, a trap, a
deflection) with a player close to the ball. The change is measured between
the average velocity just before and just after the frame, so it catches both
the ball speeding up (kick) and slowing down / turning (control, deflection).

Possession: consecutive touches by the same player (control then pass, or a
dribble) are one possession.

Pass: the ball going from one possession to the next. Its first frame (the
last touch of the passer) is the moment offside is judged; the next possession
is who received it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pitchlens.offside.ball_path import BallPath


@dataclass
class FramePlayers:
    positions: np.ndarray    # (N, 2) pitch meters (NaN = unknown)
    teams: np.ndarray        # (N,)
    track_ids: np.ndarray    # (N,)


@dataclass
class Touch:
    frame: int
    track_id: int | None     # None: no player close enough (e.g. off-screen or a bounce)
    team: int | None
    player_index: int | None  # index into that frame's players
    ball: np.ndarray
    velocity_change: float   # m/s


@dataclass
class Possession:
    touches: list[Touch] = field(default_factory=list)

    @property
    def track_id(self) -> int | None:
        return self.touches[0].track_id

    @property
    def team(self) -> int | None:
        return self.touches[0].team


@dataclass
class Pass:
    kick: Touch              # passer's last touch: the moment offside is judged
    receive: Touch | None    # first touch of the next possession (None if the clip ends)

    @property
    def completed(self) -> bool:
        return (self.receive is not None and self.kick.team is not None
                and self.receive.team == self.kick.team)


def _velocity_change(vel: np.ndarray, window: int) -> np.ndarray:
    n = len(vel)
    change = np.full(n, np.nan)
    for t in range(window, n - window):
        before = vel[t - window:t]
        after = vel[t + 1:t + 1 + window]
        if np.isfinite(before).all() and np.isfinite(after).all():
            change[t] = np.linalg.norm(after.mean(axis=0) - before.mean(axis=0))
    return change


def _nearest_player(players: list[FramePlayers], ball: np.ndarray, t: int, search: int,
                    max_distance: float) -> tuple[int, int] | None:
    """(frame, player index) of the closest player to the ball within ±search frames."""
    best, best_d = None, max_distance
    for f in range(max(0, t - search), min(len(players), t + search + 1)):
        fp = players[f]
        if len(fp.positions) == 0 or not np.isfinite(ball[f]).all():
            continue
        d = np.linalg.norm(fp.positions - ball[f], axis=1)
        d[~np.isfinite(d)] = np.inf
        i = int(np.argmin(d))
        if d[i] <= best_d:
            best, best_d = (f, i), d[i]
    return best


def detect_touches(path: BallPath, players: list[FramePlayers], window: int = 3,
                   min_change: float = 4.0, noise_factor: float = 2.5, max_distance: float = 2.0,
                   min_gap_s: float = 0.3, unattributed_factor: float = 2.0) -> list[Touch]:
    """Touches in time order.

    min_change:   velocity change (m/s) that counts as a touch on clean data
    noise_factor: on noisy data the threshold rises to this many times the clip's
                  typical (median) velocity change, which is mostly detection jitter
    max_distance: how close (m) a player must be to the ball to be credited with it
    min_gap_s:    touches closer together than this are merged (the strongest is kept)
    unattributed_factor: a touch with no player near is more likely jitter than a real
                  touch (e.g. an off-screen receiver), so it must be this many times stronger
    """
    change = _velocity_change(path.velocity(), window)
    measured = change[np.isfinite(change)]
    threshold = min_change
    if len(measured):
        threshold = max(min_change, noise_factor * float(np.median(measured)))
    candidates = [t for t in np.flatnonzero(np.nan_to_num(change) >= threshold)]
    min_gap = max(1, int(round(min_gap_s * path.fps)))
    # non-maximum suppression: strongest change wins within min_gap
    chosen: list[int] = []
    for t in sorted(candidates, key=lambda t: -change[t]):
        if all(abs(t - c) >= min_gap for c in chosen):
            chosen.append(t)

    touches = []
    for t in sorted(chosen):
        hit = _nearest_player(players, path.positions, t, search=window, max_distance=max_distance)
        if hit is None:
            if change[t] >= unattributed_factor * threshold:
                touches.append(Touch(int(t), None, None, None, path.positions[t].copy(), float(change[t])))
            continue
        f, i = hit
        fp = players[f]
        # report the player index in the touch frame itself if they're tracked there
        idx = i
        if f != t:
            same = np.flatnonzero(players[t].track_ids == fp.track_ids[i]) if t < len(players) else []
            idx = int(same[0]) if len(same) else None
        touches.append(Touch(int(t), int(fp.track_ids[i]), int(fp.teams[i]), idx,
                             path.positions[t].copy(), float(change[t])))
    return touches


def group_possessions(touches: list[Touch]) -> list[Possession]:
    """Consecutive touches by the same player form one possession."""
    out: list[Possession] = []
    for touch in touches:
        if out and touch.track_id is not None and out[-1].track_id == touch.track_id:
            out[-1].touches.append(touch)
        else:
            out.append(Possession([touch]))
    return out


def find_passes(touches: list[Touch]) -> list[Pass]:
    """Passes between consecutive possessions; only those with a known passer."""
    poss = group_possessions(touches)
    out = []
    for i, p in enumerate(poss):
        if p.track_id is None:
            continue
        nxt = poss[i + 1].touches[0] if i + 1 < len(poss) else None
        out.append(Pass(p.touches[-1], nxt))
    return out
