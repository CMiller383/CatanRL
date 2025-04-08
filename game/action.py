from dataclasses import dataclass
from typing import Any
from game.enums import ActionType


@dataclass(frozen=True)
class Action:
    type: ActionType
    payload: Any = None
