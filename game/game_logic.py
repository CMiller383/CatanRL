# game/game_logic.py
from enum import Enum
from .board import Board
from .spot import SettlementType
from .resource import Resource
from .player import Player

class GamePhase(Enum):
    SETUP_PHASE_1 = 0  # First settlement + road for each player
    SETUP_PHASE_2 = 1  # Second settlement + road for each player (reverse order)
    REGULAR_PLAY = 2   # Regular gameplay

class GameLogic:
    def __init__(self, board):
        self.board = board
        self.current_phase = GamePhase.SETUP_PHASE_1
        self.current_player_idx = 0
        self.num_players = 4  # Fixed at 4 players
        self.players = []
        self.setup_turn_order = []  # For tracking setup phase
        self.setup_phase_settlement_placed = False  # Flag for tracking if settlement is placed in setup
        self.last_turn_of_setup = False  # Flag for tracking if we're in the last turn of setup
        
        # Initialize players
        self._init_players()
    
    def _init_players(self):
        # Create 4 players and add them to the board
        colors = ["Red", "Blue", "White", "Orange"]
        for i in range(self.num_players):
            player = Player(i+1, f"Player {i+1}")
            player.color = colors[i]
            self.players.append(player)
            self.board.add_player(i+1, f"Player {i+1}")
        
        # Setup turn order for initial placement
        self.setup_turn_order = list(range(self.num_players))
    
    def get_current_player(self):
        """Returns the current player object"""
        return self.players[self.current_player_idx]
    
    def is_valid_initial_settlement(self, spot_id):
        """
        Check if a spot is valid for initial settlement placement
        """
        spot = self.board.get_spot(spot_id)
        
        # Make sure the spot exists
        if spot is None:
            return False
        
        # Make sure spot is not already occupied
        if spot.player is not None:
            return False
        
        # In initial setup, we don't need to check for connectivity to other settlements
        # but we do need to check for the distance rule
        
        # Check distance rule: no settlement can be on an adjacent spot
        for adjacent_road in self.board.roads.values():
            if spot_id == adjacent_road.spot1_id:
                adjacent_spot = self.board.spots.get(adjacent_road.spot2_id)
                if adjacent_spot.player is not None:
                    return False
            elif spot_id == adjacent_road.spot2_id:
                adjacent_spot = self.board.spots.get(adjacent_road.spot1_id)
                if adjacent_spot.player is not None:
                    return False
        
        return True
    
    def place_initial_settlement(self, spot_id):
        """
        Place an initial settlement during setup phase
        Returns True if successful, False otherwise
        """
        if not self.is_valid_initial_settlement(spot_id):
            return False
        
        player = self.get_current_player()
        spot = self.board.get_spot(spot_id)
        
        # Place settlement
        spot.build_settlement(player.player_id, SettlementType.SETTLEMENT)
        player.add_settlement(spot_id)
        
        # Update game state
        self.setup_phase_settlement_placed = True
        
        # If in second setup phase, give resources for adjacent hexes
        if self.current_phase == GamePhase.SETUP_PHASE_2:
            self._give_initial_resources(spot_id, player)
        
        return True
    
    def _give_initial_resources(self, spot_id, player):
        """Give resources for hexes adjacent to the second settlement"""
        spot = self.board.get_spot(spot_id)
        for hex_id in spot.adjacent_hex_ids:
            hex_obj = self.board.get_hex(hex_id)
            # Don't give resources for desert
            if hex_obj.resource != Resource.DESERT:
                player.add_resource(hex_obj.resource, 1)
    
    def is_valid_initial_road(self, road_id, last_settlement_id):
        """
        Check if a road placement is valid in setup phase
        The road must be connected to the last settlement placed
        """
        road = self.board.get_road(road_id)
        
        # Make sure the road exists
        if road is None:
            return False
        
        # Make sure road is not already claimed
        if road.owner is not None:
            return False
        
        # Check if the road is connected to the last settlement
        if road.spot1_id != last_settlement_id and road.spot2_id != last_settlement_id:
            return False
        
        return True
    
    def place_initial_road(self, road_id, last_settlement_id):
        """
        Place an initial road during setup phase
        Returns True if successful, False otherwise
        """
        if not self.is_valid_initial_road(road_id, last_settlement_id):
            return False
        
        player = self.get_current_player()
        road = self.board.get_road(road_id)
        
        # Place road
        road.build_road(player.player_id)
        player.add_road(road_id)
        
        # Advance to next player or phase
        self._advance_setup_phase()
        
        return True
    
    def _advance_setup_phase(self):
        """Advance to the next player or phase in setup"""
        # Reset the settlement placement flag
        self.setup_phase_settlement_placed = False
        
        if self.current_phase == GamePhase.SETUP_PHASE_1:
            # Move to the next player
            self.current_player_idx = (self.current_player_idx + 1) % self.num_players
            
            # If we've gone through all players, switch to phase 2 (reverse order)
            if self.current_player_idx == 0:
                self.current_phase = GamePhase.SETUP_PHASE_2
                self.current_player_idx = self.num_players - 1  # Start with the last player
        
        elif self.current_phase == GamePhase.SETUP_PHASE_2:
            # Move to the previous player (reverse order)
            self.current_player_idx -= 1
            
            # If we've gone through all players in reverse order
            if self.current_player_idx < 0:
                self.current_player_idx = 0  # Reset to first player for regular play
                self.current_phase = GamePhase.REGULAR_PLAY
    
    def get_setup_instructions(self):
        """Get instructions for the current setup phase"""
        player = self.get_current_player()
        
        if not self.setup_phase_settlement_placed:
            return f"{player.name} ({player.color}): Place your {'second' if self.current_phase == GamePhase.SETUP_PHASE_2 else 'first'} settlement"
        else:
            return f"{player.name} ({player.color}): Place a road connected to your settlement"
    
    def is_setup_complete(self):
        """Check if the setup phase is complete"""
        return self.current_phase == GamePhase.REGULAR_PLAY