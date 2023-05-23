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
    config_idx: int
    feats: Any
    plan_idxs: np.ndarray
    # did the turn have a confirmation
    confirmation: bool | None
    confirmed: bool

@dataclass
class OurPlan(Plan):
    info_gain: float | None

@dataclass
class StartPlan(OurPlan):
    pass

@dataclass
class FollowupPlan(OurPlan):
    newdots: np.ndarray
    olddots: np.ndarray
    reference_turn: int
    pass

@dataclass
class SelectPlan(Plan):
    reference_turn: int
    pass

@dataclass
class State:
    belief_dist: np.ndarray
    plan: Plan | None
    speaker: Speaker | None
    turn: int
    past: Any
    write_extra: Any = None
    read_extra: Any = None
