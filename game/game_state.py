from .development_card import DevelopmentCardDeck
from .enums import Resource, GamePhase


class GameState:
    def __init__(self, board):
        self.board = board
        self.current_phase = GamePhase.SETUP_PHASE_1
        self.current_player_idx = 0
        self.players = []
        
        self.setup_phase_settlement_placed = False  # Flag for tracking if settlement is placed in setup
        self.waiting_for_human_input = False  # Flag to track if we're waiting for human input
        self.last_settlement_placed = None  # Track last settlement for road placement
        self.dice1_roll = None
        self.dice2_roll = None
        self.rolled_dice = False
        self.possible_actions = set()

        self.turn_number = 0

        self.dev_card_deck = DevelopmentCardDeck()

        # state for dev cards
        self.dev_card_played_this_turn = False
        self.awaiting_robber_placement = False
        self.awaiting_steal_selection = False
        self.robber_hex_id = None  # Current location of the robber
        self.awaiting_resource_selection = False
        self.awaiting_resource_selection_count = 0  # For Year of Plenty
        self.awaiting_monopoly_selection = False
        self.road_building_roads_placed = 0
        self.awaiting_road_builder_placements = False

        for hex_id, hex_obj in self.board.hexes.items():
            if hex_obj.resource == Resource.DESERT:
                self.robber_hex_id = hex_id
                break
        
        # army
        self.largest_army_player = None
        self.largest_army_size = 2 #at least 3 knights 
        
        # road
        self.longest_road_player = None
        self.longest_road_length = 4  # at least 5 roads 

        self.winner = None
    
    def get_current_player(self): ##please dont remove thiis i know its not necessary but i tried to get rid of it and it broke everything
        return self.players[self.current_player_idx]
    
def check_game_over(state):
    """Check if the game is over."""
    if state.winner is not None:
        return True

    # Check if any player has 10 victory points
    for player in state.players:
        if player.victory_points >= 10:
            state.winner = player.player_idx
            return True

    return False
    