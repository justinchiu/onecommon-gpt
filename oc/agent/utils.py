from dataclasses import dataclass
import numpy as np

@dataclass
class Plan:
    dots: np.ndarray
    newdots: np.ndarray
    olddots: np.ndarray
    plan_idxs: np.ndarray
    should_select: bool
