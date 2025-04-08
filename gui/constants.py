from game.enums import Resource

SCREEN_PROPORTION = .7

RESOURCE_COLORS = {
    Resource.WOOD: (60, 179, 113),    # Medium Sea Green
    Resource.BRICK: (205, 92, 92),    # Indian Red
    Resource.WHEAT: (255, 215, 0),     # Gold
    Resource.SHEEP: (152, 251, 152),   # Pale Green
    Resource.ORE: (169, 169, 169),     # Dark Gray
    Resource.DESERT: (244, 164, 96)    # Sandy Brown
}

PLAYER_COLOR_RGBS = {
    0: (255, 99, 71),    # Tomato Red
    1: (65, 105, 225),   # Royal Blue
    2: (255, 255, 255),  # White
    3: (138, 43, 226)    # Blue Violet
}
PLAYER_COLOR_NAMES = {
    0: 'red',
    1: 'blue',
    2: 'white',
    3: 'violet'
}

ROAD_COLOR = (160, 82, 45)             # Sienna
ROAD_HIGHLIGHT_COLOR = (255, 69, 0)    # Orange Red
ROAD_INVALID_COLOR = (140, 140, 140)   # Dark Gray
ROAD_VALID_COLOR = (200, 95, 55) 

SPOT_COLOR = (255, 255, 255)           # White

TEXT_COLOR = (0, 0, 0)                 # Black
BACKGROUND_COLOR = (176, 224, 230)     # Powder Blue
END_BUTTON_ENABLED_COLOR = (255, 0, 0)
END_BUTTON_DISABLED_COLOR = (140, 0, 0)

DEV_CARD_COLOR = (245, 245, 220)       # Beige
DEV_CARD_ENABLED_COLOR = (220, 220, 150)  # Light yellow
DEV_CARD_DISABLED_COLOR = (190, 190, 190)  # Gray
DEV_CARD_SELECTED_COLOR = (255, 223, 0)  # Gold