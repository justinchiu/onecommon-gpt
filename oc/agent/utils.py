from dataclasses import dataclass
from enum import Enum
import numpy as np


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
    newdots: np.ndarray
    olddots: np.ndarray
    plan_idxs: np.ndarray
    should_select: bool
    confirmation: bool | None
    info_gain: float | None
    turn: int

@dataclass
class PlanConfirmation:
    dots: np.ndarray
    confirmed: bool | None
    selection: bool
    speaker: Speaker
    config_idx: int
    turn: int

@dataclass
class PartnerPlan:
    preds: np.ndarray

@dataclass
class State:
    belief_dists: list[np.ndarray]
    plans_confirmations: list[PlanConfirmation]
    our_plans: list[Plan]
    their_plans: list[PartnerPlan]
    turn: int
