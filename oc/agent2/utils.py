from dataclasses import dataclass
from enum import Enum
import numpy as np
from typing import Any


class Speaker(Enum):
    YOU = 1
    THEM = 2

class Action(Enum):
    START = 1
    FOLLOWUP = 2
    SELECT = 3

@dataclass
class Plan:
    dots: np.ndarray
    plan_idxs: np.ndarray
    turn: int
    id: int

@dataclass
class OurPlan(Plan):
    info_gain: float | None
    # do we give them a confirmation
    confirmation: bool | None

@dataclass
class StartPlan(OurPlan):
    pass

@dataclass
class FollowupPlan(OurPlan):
    newdots: np.ndarray
    olddots: np.ndarray
    reference_id: int
    pass

@dataclass
class SelectPlan(OurPlan):
    reference_id: int
    pass

@dataclass
class ConfirmedPlan:
    plan: Plan
    confirmed: bool | None


@dataclass
class State:
    belief_dists: list[np.ndarray]
    plans_confirmations: list[PlanConfirmation]
    our_plans: list[Plan]
    their_plans: list[PartnerPlan]
    turn: int
    past: Any
