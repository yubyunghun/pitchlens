"""Which teammates can the player on the ball pass to?

A pass is modelled as the ball travelling in a straight line from the ball to
the receiver at `pass_speed`. A defender cuts the lane if they can get to some
point of that line before the ball does, and before the receiver does (the
receiver knows the pass is coming and steps in to meet it, so a defender
marking from behind can close them down but not intercept):

    ball time      = distance along the line / pass_speed
    defender time  = reaction + max(0, distance to the point - reach) / run_speed
    receiver time  = max(0, distance to the point - reach) / run_speed
    margin         = defender time - ball time   (> 0: the ball gets there first)
                     at points the defender reaches before the receiver

The lane's margin is the tightest one over every defender and every point of
the line. It becomes an openness from 0 to 1 (0.5 = dead heat), so lanes can be
"open", "risky" or "blocked" rather than just yes/no.

This is geometry only: it doesn't know body orientation, passing skill or
lofted balls over a defender, so "open" means "probably open on the ground".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OPEN, RISKY, BLOCKED = "open", "risky", "blocked"


@dataclass(frozen=True)
class LaneModel:
    pass_speed: float = 15.0      # m/s, a firm ground pass
    run_speed: float = 7.0        # m/s, a defender sprinting
    reaction: float = 0.3         # s before the defender moves
    reach: float = 1.0            # m a defender can stretch to intercept
    softness: float = 0.15        # s: how quickly openness goes from 0 to 1 around a zero margin
    samples: int = 25             # points checked along each lane
    open_above: float = 0.7
    blocked_below: float = 0.3


@dataclass
class Lane:
    receiver_index: int
    length_m: float
    margin_s: float               # tightest (defender time - ball time) along the lane
    openness: float               # 0 (cut) .. 1 (clear)
    blocker_index: int | None     # defender who comes closest to cutting it
    status: str                   # OPEN / RISKY / BLOCKED
    forward_m: float = 0.0        # meters gained toward the opponents' goal (< 0: backwards)
    bypassed: int = 0             # opponents the pass goes past (between ball and receiver in depth)

    def value(self, bypass_worth_m: float = 8.0) -> float:
        """How useful the pass is if it arrives: ground gained, plus extra per opponent bypassed."""
        return self.forward_m + bypass_worth_m * self.bypassed


def lane_margin(ball: np.ndarray, receiver: np.ndarray, defenders: np.ndarray,
                model: LaneModel = LaneModel()) -> tuple[float, int | None]:
    """(tightest margin in seconds, index into `defenders` of who sets it)."""
    if len(defenders) == 0:
        return float("inf"), None
    s = np.linspace(0.0, 1.0, model.samples)
    points = ball[None, :] + s[:, None] * (receiver - ball)[None, :]          # (S, 2)
    t_ball = s * np.linalg.norm(receiver - ball) / model.pass_speed           # (S,)
    dist = np.linalg.norm(defenders[:, None, :] - points[None, :, :], axis=2)  # (D, S)
    t_def = model.reaction + np.maximum(0.0, dist - model.reach) / model.run_speed
    t_recv = np.maximum(0.0, np.linalg.norm(points - receiver, axis=1) - model.reach) / model.run_speed
    margin = np.where(t_def < t_recv[None, :], t_def - t_ball[None, :], np.inf)
    d, k = np.unravel_index(int(np.argmin(margin)), margin.shape)
    if not np.isfinite(margin[d, k]):
        return float("inf"), None                  # the receiver beats every defender everywhere
    return float(margin[d, k]), int(d)


def openness_from_margin(margin: float, model: LaneModel = LaneModel()) -> float:
    if not np.isfinite(margin):
        return 1.0
    return float(1.0 / (1.0 + np.exp(-margin / model.softness)))


def status_of(openness: float, model: LaneModel = LaneModel()) -> str:
    if openness >= model.open_above:
        return OPEN
    if openness < model.blocked_below:
        return BLOCKED
    return RISKY


def passing_lanes(positions: np.ndarray, teams: np.ndarray, carrier: int, ball: np.ndarray | None = None,
                  model: LaneModel = LaneModel()) -> list[Lane]:
    """Lanes from the ball to every visible teammate of `carrier` (pitch meters).

    ball: where the pass starts; defaults to the carrier's position.
    """
    pos = np.asarray(positions, dtype=float).reshape(-1, 2)
    teams = np.asarray(teams)
    known = np.isfinite(pos).all(axis=1)
    start = pos[carrier] if ball is None or not np.isfinite(ball).all() else np.asarray(ball, float)
    team = teams[carrier]
    others = {t for t in teams[known] if t != team and t >= 0}
    defender_idx = np.flatnonzero(known & np.isin(teams, list(others)))
    defenders = pos[defender_idx]
    lanes = []
    for i in np.flatnonzero(known & (teams == team)):
        if i == carrier:
            continue
        margin, d = lane_margin(start, pos[i], defenders, model)
        openness = openness_from_margin(margin, model)
        lanes.append(Lane(int(i), float(np.linalg.norm(pos[i] - start)), margin, openness,
                          None if d is None else int(defender_idx[d]), status_of(openness, model)))
    return lanes


def score_lanes(lanes: list[Lane], positions: np.ndarray, teams: np.ndarray, start: np.ndarray,
                attacks_right: bool) -> list[Lane]:
    """Fills in forward_m and bypassed for each lane (in place) and returns them.

    start: where the pass starts (ball, or the carrier). attacks_right: the passing team's direction.
    """
    pos = np.asarray(positions, dtype=float).reshape(-1, 2)
    teams = np.asarray(teams)
    sign = 1.0 if attacks_right else -1.0
    start_depth = sign * float(start[0])
    for lane in lanes:
        team = teams[lane.receiver_index]
        recv_depth = sign * pos[lane.receiver_index, 0]
        lane.forward_m = float(recv_depth - start_depth)
        opp = np.isfinite(pos).all(axis=1) & (teams != team) & (teams >= 0)
        opp_depth = sign * pos[opp, 0]
        lane.bypassed = int(((opp_depth > start_depth) & (opp_depth < recv_depth)).sum())
    return lanes


def best_option(lanes: list[Lane], bypass_worth_m: float = 8.0) -> Lane | None:
    """The most valuable OPEN lane, or None if nothing is open."""
    open_lanes = [ln for ln in lanes if ln.status == OPEN]
    if not open_lanes:
        return None
    return max(open_lanes, key=lambda ln: (ln.value(bypass_worth_m), ln.openness))


def evaluate_lanes(positions: np.ndarray, teams: np.ndarray, carrier: int, attacks_right: bool,
                   ball: np.ndarray | None = None, model: LaneModel = LaneModel()) -> list[Lane]:
    """Lanes from the ball to every teammate, with openness and value."""
    pos = np.asarray(positions, dtype=float).reshape(-1, 2)
    start = pos[carrier] if ball is None or not np.isfinite(ball).all() else np.asarray(ball, float)
    return score_lanes(passing_lanes(pos, teams, carrier, start, model), pos, teams, start, attacks_right)


def ball_carrier(positions: np.ndarray, teams: np.ndarray, ball: np.ndarray, team: int | None = None,
                 max_distance: float = 3.0) -> int | None:
    """Player nearest the ball (optionally only of `team`), if within max_distance meters."""
    pos = np.asarray(positions, dtype=float).reshape(-1, 2)
    if ball is None or not np.isfinite(ball).all() or len(pos) == 0:
        return None
    d = np.linalg.norm(pos - np.asarray(ball, float), axis=1)
    d[~np.isfinite(d)] = np.inf
    if team is not None:
        d[np.asarray(teams) != team] = np.inf
    i = int(np.argmin(d))
    return i if d[i] <= max_distance else None
