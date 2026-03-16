"""Offside decision for a pass.

Being in an offside position at the kick is only an offence if the player then
becomes involved in active play. Of the ways to be involved (Law 11: playing
or touching the ball, interfering with an opponent, gaining an advantage),
this decides the clearest one: the player in an offside position is the one
who receives the ball.

Verdicts:
  OFFSIDE       receiver was in an offside position at the kick
  TOO_CLOSE     receiver was within the measurement tolerance of the line
  ONSIDE        receiver was onside at the kick
  NO_OFFENCE    the ball went to an opponent (no offside from an opponent's
                possession; teammates in an offside position were not involved)
  NO_DECISION   can't be judged from the footage (line or receiver unknown,
                or the ball wasn't received in the clip)

Not covered yet: interfering with an opponent without touching the ball, and
restarts that are exempt from offside (throw-ins, goal kicks, corners), which
can't be detected from tracking alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pitchlens.offside.freeze import FrozenFrame
from pitchlens.offside.passes import Pass
from pitchlens.offside.position import OFFSIDE as POS_OFFSIDE
from pitchlens.offside.position import ONSIDE as POS_ONSIDE
from pitchlens.offside.position import TOO_CLOSE as POS_TOO_CLOSE
from pitchlens.offside.position import PositionCheck

OFFSIDE = "offside"
TOO_CLOSE = "too_close"
ONSIDE = "onside"
NO_OFFENCE = "no_offence"
NO_DECISION = "no_decision"


@dataclass
class OffsideDecision:
    pass_: Pass
    frozen: FrozenFrame
    verdict: str
    reason: str
    receiver_check: PositionCheck | None = None
    uninvolved_offside: list[int] = field(default_factory=list)   # in offside position, not receiving

    @property
    def is_offence(self) -> bool:
        return self.verdict == OFFSIDE


def decide(p: Pass, frozen: FrozenFrame) -> OffsideDecision:
    receiver = p.receive
    offside_ids = frozen.offside_track_ids()

    def result(verdict, reason, check=None):
        rid = None if receiver is None else receiver.track_id
        return OffsideDecision(p, frozen, verdict, reason, check, [t for t in offside_ids if t != rid])

    if frozen.line is None:
        return result(NO_DECISION, "offside line unknown at the kick (too few defenders visible)")
    if receiver is None:
        return result(NO_DECISION, "ball not received before the clip ends")
    if receiver.track_id is None or receiver.team is None:
        return result(NO_DECISION, "receiver not identified (off-screen or not tracked)")
    if receiver.team != frozen.attacking_team:
        return result(NO_OFFENCE, "ball went to an opponent")
    check = frozen.checks.get(receiver.track_id)
    if check is None:
        return result(NO_DECISION, "receiver's position unknown at the kick")
    if check.status == POS_OFFSIDE:
        return result(OFFSIDE, f"receiver {check.margin_m:+.2f} m beyond the line at the kick", check)
    if check.status == POS_TOO_CLOSE:
        return result(TOO_CLOSE, f"receiver {check.margin_m:+.2f} m from the line: within tolerance", check)
    assert check.status == POS_ONSIDE
    return result(ONSIDE, f"receiver {abs(check.margin_m):.2f} m behind the line at the kick", check)


def decide_all(passes: list[Pass], frozen: list[FrozenFrame]) -> list[OffsideDecision]:
    return [decide(p, f) for p, f in zip(passes, frozen)]
