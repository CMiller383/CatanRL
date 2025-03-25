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
        self.setup_phase_settlement_placed = False  # Flag for tracking if settlement is placed in setup
        self.waiting_for_human_input = False  # Flag to track if we're waiting for human input
        self.last_settlement_placed = None  # Track last settlement for road placement
        self.last_dice1_roll = None
        self.last_dice2_roll = None
        self.rolled_dice = False
        self.possible_moves = set()
        
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
        """Check if a spot is valid for initial settlement placement"""
        spot = self.board.get_spot(spot_id)
        
        # Make sure the spot exists and is free
        if spot is None or spot.player is not None:
            return False

        # Check distance rule
        return self.is_two_spots_away_from_settlement(spot_id)
    
    def roll_dice(self):
        if self.current_phase != GamePhase.REGULAR_PLAY:
            return False
        
        if "roll_dice" not in self.possible_moves:
            return False
        
        self.last_dice1_roll = random.randint(1, 6)
        self.last_dice2_roll = random.randint(1, 6)
        self.rolled_dice = True
        self.distribute_resources(self.last_dice1_roll + self.last_dice2_roll)

        self.possible_moves = self.get_possible_moves()

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
        
        if "end_turn" not in self.possible_moves:
            return
        
        self.rolled_dice = False
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players

        self.possible_moves = self.get_possible_moves()


    def get_possible_moves(self):
        moves = set()

        if self.current_phase != GamePhase.REGULAR_PLAY:
            # For now, we only consider regular play moves
            return moves
        
        curr_player = self.get_current_player()

        if not self.rolled_dice:
            moves.add("roll_dice")
            moves.add("play_knight")
            
            return moves
        else:
            moves.add("end_turn")
        
        # Add all possible settlement builds
        if curr_player.has_settlement_resources():
            for spot_id, spot in self.board.spots.items():
                if spot.player is not None:
                    continue 

                has_adjascent_road = self.has_adjascent_road(spot_id)
                is_two_spots_away = self.is_two_spots_away_from_settlement(spot_id)

                if has_adjascent_road and is_two_spots_away:
                    moves.add(("upgrade", spot_id))
        
        # Add all possible city builds
        if curr_player.has_city_resources():
            for spot_id, spot in self.board.spots.items():
                if spot.player == curr_player.player_id and spot.settlement_type == SettlementType.SETTLEMENT:
                    moves.add(("upgrade", spot_id))
        
        # Add all possible road builds
        if curr_player.has_road_resources():
            for road_id, road in self.board.roads.items():
                # Check if road is already claimed
                if road.owner is not None:
                    continue

                # Check connectivity: the road must touch a settlement or road of the current player.
                touching_settlement = any(spot_id in (road.spot1_id, road.spot2_id) for spot_id in curr_player.settlements)
                touching_road = False
                for r_id in curr_player.roads:
                    existing_road = self.board.get_road(r_id)
                    if existing_road:
                        if road.spot1_id in (existing_road.spot1_id, existing_road.spot2_id) or \
                        road.spot2_id in (existing_road.spot1_id, existing_road.spot2_id):
                            touching_road = True
                            break

                if touching_settlement or touching_road:
                    moves.add(("road", road_id))
        
        return moves

    # checks that our spot is touching a road we built
    def has_adjascent_road(self, spot_id):
        curr_player = self.get_current_player()

        has_adjacent_road = False
        for r_id in curr_player.roads:
            road = self.board.get_road(r_id)
            if road and spot_id in (road.spot1_id, road.spot2_id):
                has_adjacent_road = True
                break
        
        return has_adjacent_road

    # distance rule
    # checks that a spot is 2 away
    def is_two_spots_away_from_settlement(self, spot_id):
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


    def upgrade_spot(self, spot_id):        
        if ("upgrade", spot_id) not in self.possible_moves:
            return False
        
        curr_player = self.get_current_player()
        
        spot = self.board.get_spot(spot_id)

        if spot.settlement_type == SettlementType.SETTLEMENT:
            spot.build_settlement(curr_player.player_id, SettlementType.CITY)
            curr_player.buy_city()
        else:
            spot.build_settlement(curr_player.player_id, SettlementType.SETTLEMENT)
            curr_player.buy_settlement()
        
        self.possible_moves = self.get_possible_moves()

        return True


    def place_road(self, road_id):
        if ("road", road_id) not in self.possible_moves:
            return False
        
        new_road = self.board.get_road(road_id)
        curr_player = self.get_current_player()

        curr_player.buy_road()
        new_road.build_road(curr_player.player_id)
        curr_player.add_road(road_id)

        self.possible_moves = self.get_possible_moves()

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
                self.current_player_idx += 1   

        elif self.current_phase == GamePhase.SETUP_PHASE_2:
            # If we've gone through all players in reverse order
            if self.current_player_idx == 0:
                self.current_phase = GamePhase.REGULAR_PLAY
                self.possible_moves = self.get_possible_moves()
            else: 
                self.current_player_idx -= 1
            
    def is_setup_complete(self):
        """Check if the setup phase is complete"""
        return self.current_phase == GamePhase.REGULAR_PLAY
    
    def do_move(self, move):
        if move not in self.possible_moves:
            return False
        
        if move[0] == "upgrade":
            self.upgrade_spot(move[1])
        elif move[0] == "road":
            self.place_road(move[1])
        elif move == "end_turn":
            self.end_turn()
        elif move == "roll_dice":
            self.roll_dice()
        
        
        self.possible_moves = self.get_possible_moves()
        return True
    

    # WE SHOULD EVENTUALLY CHANGE THIS TO PASS IN SOME DATA TO THE AGENT
    # AND GET A MOVE BACK
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
        random_move = random.choice(tuple(self.possible_moves))
        while random_move != "end_turn":
            self.do_move(random_move)
            random_move = random.choice(tuple(self.possible_moves))
        
        self.do_move(random_move)