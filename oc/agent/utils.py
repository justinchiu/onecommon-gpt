from dataclasses import dataclass
from enum import Enum
import numpy as np

class Qtypes(Enum):
    START = "New question."
    FOLD = "Follow up question, no new dots."
    FNEW = "Follow up question, new dots."
    SELECT = "Select a dot."
    NOOP = "No op."

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

@dataclass
class PlanConfirmation:
    dots: np.ndarray
    confirmed: bool | None
    selection: bool
    speaker: Speaker
    config_idx: int

@dataclass
class PartnerPlan:
    preds: np.ndarray

