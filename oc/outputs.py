from dataclasses import dataclass
from collections.abc import Iterable, Mapping

@dataclass
class UnderstandShortOutput:
    code: str
    constraints: Iterable[Mapping[str, str]]
    dots: str
    selection: str
    speaker: str
    text: str
