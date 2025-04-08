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
