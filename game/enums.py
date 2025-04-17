from enum import Enum

class GamePhase(Enum):
    SETUP_PHASE_1 = 0  # First settlement + road for each player
    SETUP_PHASE_2 = 1  # Second settlement + road for each player (reverse order)
    REGULAR_PLAY = 2   # Regular gameplay

class SettlementType(Enum):
    NONE = 0
    SETTLEMENT = 1
    CITY = 2

class Resource(Enum):
    WOOD = "wood"
    BRICK = "brick"
    WHEAT = "wheat"
    SHEEP = "sheep"
    ORE = "ore"
    DESERT = "desert"  # optional

    from enum import Enum

class DevCardType(Enum):
    KNIGHT = "knight"
    VICTORY_POINT = "victory_point"
    ROAD_BUILDING = "road_building"
    YEAR_OF_PLENTY = "year_of_plenty"
    MONOPOLY = "monopoly"

class ActionType(Enum):
    ROLL_DICE = "roll_dice"
    END_TURN = "end_turn"
    BUILD_SETTLEMENT = "build_settlement"
    UPGRADE_TO_CITY = "upgrade_city"
    BUILD_ROAD = "build_road"
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
    TRADE_RESOURCES = "trade_resources"