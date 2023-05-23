from dataclasses import dataclass
from collections.abc import Iterable, Mapping

@dataclass
class UnderstandShortOutput:
    code: str
    constraints: Iterable[Mapping[str, str]]
    dots: str
    select: str
    state: str
    speaker: str
    text: str


@dataclass
class UnderstandShortInput2:
    olddots: list[str]
    reference_turn: int | None
    speaker: str
    text: str

@dataclass
class UnderstandShortOutput2:
    olddots: list[str]
    newdots: list[str]
    constraints: Iterable[Mapping[str, str]]
    select: str
    speaker: str
    text: str
