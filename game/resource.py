from enum import Enum

class Resource(Enum):
    WOOD = "wood"
    BRICK = "brick"
    WHEAT = "wheat"
    SHEEP = "sheep"
    ORE = "ore"
    DESERT = "desert"  # optional

    def __str__(self):
        return self.value
