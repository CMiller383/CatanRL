# game/game_logic.py
from enum import Enum
from .board import Board
from .spot import SettlementType
from .resource import Resource
from .player import Player
import random

class GamePhase(Enum):
    SETUP_PHASE_1 = 0  # First settlement + road for each player
    SETUP_PHASE_2 = 1  # Second settlement + road for each player (reverse order)
    REGULAR_PLAY = 2   # Regular gameplay

from agent.base import AgentType, create_agent

class GameLogic:
    def __init__(self, board, num_human_players=1, agent_types=None):
        self.board = board
        self.current_phase = GamePhase.SETUP_PHASE_1
        self.current_player_idx = 0
        self.num_players = 4  # Fixed at 4 players
        self.players = []
        self.agents = []
        self.setup_turn_order = []  # For tracking setup phase
        self.setup_phase_settlement_placed = False  # Flag for tracking if settlement is placed in setup
        self.waiting_for_human_input = False  # Flag to track if we're waiting for human input
        self.last_settlement_placed = None  # Track last settlement for road placement
        self.last_dice1_roll = None
        self.last_dice2_roll = None
        self.rolled_dice = False
        
        # Default all agents to random if not specified
        if agent_types is None:
            agent_types = [AgentType.RANDOM] * (self.num_players - num_human_players)
        
        # Initialize players and agents
        self._init_players(num_human_players, agent_types)
    
    def _init_players(self, num_human_players, agent_types):
        # Create 4 players and add them to the board        
        # Create human players first
        for i in range(min(num_human_players, self.num_players)):
            player = Player(i+1, f"Player {i+1} (Human)")
            player.is_human = True
            self.players.append(player)
            self.board.add_player(i+1, f"Player {i+1}")
            self.agents.append(create_agent(i+1, AgentType.HUMAN))
        
        # Create AI players for remaining slots
        for i in range(num_human_players, self.num_players):
            agent_type = agent_types[i - num_human_players]
            player = Player(i+1, f"Player {i+1} ({agent_type.name})")
            player.is_human = False
            self.players.append(player)
            self.board.add_player(i+1, f"Player {i+1}")
            self.agents.append(create_agent(i+1, agent_type))
        
        # Setup turn order for initial placement
        self.setup_turn_order = list(range(self.num_players))
    
    def get_current_player(self):
        """Returns the current player object"""
        return self.players[self.current_player_idx]
    
    def get_current_agent(self):
        """Returns the current agent object"""
        return self.agents[self.current_player_idx]
    
    def is_current_player_human(self):
        """Check if current player is human"""
        return self.players[self.current_player_idx].is_human
    
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
    
    def roll_dice(self):
        if self.current_phase != GamePhase.REGULAR_PLAY or self.rolled_dice:
            print(self.rolled_dice)
            return False
        
        self.last_dice1_roll = random.randint(1, 6)
        self.last_dice2_roll = random.randint(1, 6)
        self.rolled_dice = True
        self.distribute_resources(self.last_dice1_roll + self.last_dice2_roll)
        return True

    def distribute_resources(self, dice_result):
        for spot_id in self.board.spots:
            spot = self.board.get_spot(spot_id)
            if spot.player != None:
                for hex_id in spot.adjacent_hex_ids:
                    hex = self.board.get_hex(hex_id)
                    if hex.number == dice_result:
                        amount = 1
                        if spot.settlement_type == SettlementType.CITY:
                            amount = 2
                        player = self.players[spot.player - 1]
                        player.add_resource(hex.resource, amount)
    
    def user_can_end_turn(self):
        if not (self.rolled_dice and self.current_phase == GamePhase.REGULAR_PLAY):
            return False
        
        if not self.is_current_player_human():
            return False
        
        return True
        
    def end_turn(self):
        if not (self.rolled_dice and self.current_phase == GamePhase.REGULAR_PLAY):
            return
        
        self.rolled_dice = False
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players

    def upgrade_spot(self, spot_id):
        curr_player = self.get_current_player()
        spot = self.board.get_spot(spot_id)
        if not self.rolled_dice:
            print("roll dice before upgrading spot")
        
        # cant upgrade a spot with a city on it
        if spot.settlement_type == SettlementType.CITY:
            return False
        elif spot.settlement_type == SettlementType.SETTLEMENT:
            if spot.player != self.current_player_idx + 1:
                print('not curr player')
                return False
            # check if we have the resources
            if curr_player.has_city_resources():
                curr_player.buy_city()
                spot.build_settlement(curr_player.player_id, SettlementType.CITY)
                return True
            else:
                print('missing resources')
                return False
        else:
            print("trying to build settlement")

            # check that player has road to this spot
            has_road_to_spot = False
            for road_id in curr_player.roads:
                road = self.board.get_road(road_id)
                if spot_id in (road.spot1_id, road.spot2_id):
                    has_road_to_spot = True
            
            if not has_road_to_spot:
                print("Doesnt have adjascent road")
                return False
            
            #check that there isnt an adjascent settlement
            if not self.is_valid_initial_settlement(spot_id):
                print("too close to another settlement")
                return False

            # check that player has enough resources
            if not curr_player.has_settlement_resources():
                print("doesnt have resources")
                return False
            
            curr_player.buy_settlement()
            spot.build_settlement(curr_player.player_id, SettlementType.SETTLEMENT)
            return True
    


    def place_road(self, new_road_id):
        new_road = self.board.get_road(new_road_id)
        curr_player = self.get_current_player()

        if not self.rolled_dice:
            print("Roll dice before placing road")
            return False
        
        # Make sure the road exists
        if new_road is None or new_road.owner is not None:
            print('Road doesnt exist or has owner')
            return False
        
        if not curr_player.has_road_resources():
            print("Player does not have enough resources")
            return False
        
        touching_settlement = False
        for settlement_spot_id in curr_player.settlements:
            if settlement_spot_id in (new_road.spot1_id, new_road.spot2_id):
                touching_settlement = True
        
        touching_road = False
        for road_id in self.get_current_player().roads:
            road = self.board.get_road(road_id)
            if (new_road.spot1_id in (road.spot1_id, road.spot2_id) or 
                new_road.spot2_id in (road.spot1_id, road.spot2_id)):
                touching_road = True
        
        if not (touching_road or touching_settlement):
            print("Player does not have touching road or settlement")
            return False
        
        self.get_current_player().buy_road()
        new_road.build_road(curr_player.player_id)
        curr_player.add_road(new_road_id)
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
            print(f"Giving resources to {player.name} for second settlement")
            for resource, count in player.resources.items():
                if count > 0:
                    print(f"  - {resource.name}: {count}")
        
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
        print("advancing")
        self._advance_setup_phase()
        
        return True
    
    def _advance_setup_phase(self):
        """Advance to the next player or phase in setup"""
        # Reset the settlement placement flag
        self.setup_phase_settlement_placed = False
        self.rolled_dice = False
        
        if self.current_phase == GamePhase.SETUP_PHASE_1:
            # If we've gone through all players, switch to phase 2 (reverse order)
            if self.current_player_idx == self.num_players - 1:
                self.current_phase = GamePhase.SETUP_PHASE_2
            else:
                print("incrementing")
                self.current_player_idx += 1   

        elif self.current_phase == GamePhase.SETUP_PHASE_2:
            # If we've gone through all players in reverse order
            if self.current_player_idx == 0:
                self.current_phase = GamePhase.REGULAR_PLAY
            else: 
                self.current_player_idx -= 1
            
    
    def get_setup_instructions(self):
        """Get instructions for the current setup phase"""
        player = self.get_current_player()
        
        if not self.setup_phase_settlement_placed:
            return f"{player.name}: Place your {'second' if self.current_phase == GamePhase.SETUP_PHASE_2 else 'first'} settlement"
        else:
            return f"{player.name}: Place a road connected to your settlement"
    
    def is_setup_complete(self):
        """Check if the setup phase is complete"""
        return self.current_phase == GamePhase.REGULAR_PLAY
    
    def process_ai_turn(self):
        """Process a turn for an AI player"""
        if self.is_current_player_human():
            # Not an AI player, do nothing
            self.waiting_for_human_input = True
            return False
        
        # Get the current agent
        agent = self.get_current_agent()
        
        # Handle the setup phase
        if not self.is_setup_complete():
            if not self.setup_phase_settlement_placed:
                # AI needs to place a settlement
                from agent.random_agent import RandomAgent
                if isinstance(agent, RandomAgent):
                    spot_id = agent.get_initial_settlement(self)
                    if spot_id and self.place_initial_settlement(spot_id):
                        self.last_settlement_placed = spot_id
                        print(f"AI {self.get_current_player().name} placed settlement at spot {spot_id}")
                        return True  # Successfully processed part of the turn
            else:
                # AI needs to place a road
                from agent.random_agent import RandomAgent
                if isinstance(agent, RandomAgent):
                    road_id = agent.get_initial_road(self, self.last_settlement_placed)
                    if road_id and self.place_initial_road(road_id, self.last_settlement_placed):
                        print(f"AI {self.get_current_player().name} placed road at {road_id}")
                        self.last_settlement_placed = None
                        return True  # Successfully processed the turn
        
        # Handle regular play phase (to be implemented)
        # we will need to handle knights later before this
        self.roll_dice()
        # we will then need to chose moves
        self.end_turn()

        return False  # Turn not processed