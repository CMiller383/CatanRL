# game/action.py

from dataclasses import dataclass
from enum import Enum
from typing import Any

class ActionType(Enum):
    ROLL_DICE = "roll_dice"
    END_TURN = "end_turn"
    BUILD_SETTLEMENT = "build_settlement"
    UPGRADE_TO_CITY = "upgrade_city"
    PLACE_ROAD = "place_road"
    PLACE_FREE_ROAD = "place_free_road"
    BUY_DEV_CARD = "buy_dev_card"
    PLAY_KNIGHT_CARD = "play_knight_card"
    PLAY_ROAD_BUILDING_CARD = "play_road_building_card"
    PLAY_YEAR_OF_PLENTY_CARD = "play_year_of_plenty_card"
    PLAY_MONOPOLY_CARD = "play_monopoly_card"
    MOVE_ROBBER = "move_robber"
    STEAL = "steal_resource"
    SELECT_YEAR_OF_PLENTY_RESOURCE = "select_year_of_plenty_resource"
    SELECT_MONOPOLY_RESOURCE = "select_monopoly_resource"

@dataclass(frozen=True)
class Action:
    type: ActionType
    payload: Any = None
