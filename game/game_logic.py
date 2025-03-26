# game/game_logic.py
from enum import Enum
from .board import Board
from .spot import SettlementType
from .resource import Resource
from .player import Player
import random
from agent.random_agent import RandomAgent
from agent.simple_heuristic_agent import SimpleHeuristicAgent
from .development_card import DevelopmentCard, DevelopmentCardDeck, DevCardType

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

        self.dev_card_deck = DevelopmentCardDeck()

        # state for dev cards
        self.dev_card_played_this_turn = False
        self.awaiting_robber_placement = False
        self.robber_hex_id = None  # Current location of the robber
        self.awaiting_resource_selection = False
        self.awaiting_resource_selection_count = 0  # For Year of Plenty
        self.awaiting_monopoly_selection = False
        self.road_building_roads_placed = 0

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
        
        # Default all agents to random if not specified
        if agent_types is None:
            agent_types = [AgentType.HEURISTIC] * (self.num_players - num_human_players)
        
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
        dice_sum = self.last_dice1_roll + self.last_dice2_roll
        self.rolled_dice = True
        
        # Check for "7" - activate robber
        if dice_sum == 7:
            self._handle_robber_roll()
        else:
            # Distribute resources normally
            self.distribute_resources(dice_sum)

        self.possible_moves = self.get_possible_moves()
        return True

    def _handle_robber_roll(self):
        """Handle the effects of rolling a 7 (robber activation)"""
        # First, all players with more than 7 cards discard half, rounded down
        for player in self.players:
            total_cards = sum(player.resources.values())
            if total_cards > 7:
                discard_count = total_cards // 2
                # For AI players, automatically discard random cards
                if not player.is_human:
                    self._auto_discard_resources(player, discard_count)
                else:
                    # For human players, this will need UI support
                    # We'll handle this in the UI layer
                    pass
        
        # Set flag to await robber placement
        self.awaiting_robber_placement = True

    def _auto_discard_resources(self, player, discard_count):
        """Automatically discard resources for AI players when a 7 is rolled"""
        resources_list = []
        for resource, count in player.resources.items():
            resources_list.extend([resource] * count)
        
        random.shuffle(resources_list)
        for i in range(discard_count):
            if resources_list:
                resource = resources_list.pop()
                player.resources[resource] -= 1

    def distribute_resources(self, dice_result):
        """Distribute resources based on dice roll"""
        for hex_id, hex_obj in self.board.hexes.items():
            # Skip hexes where the robber is located
            if hex_obj.number == dice_result and hex_id != self.robber_hex_id:
                for spot_id in self.board.spots:
                    spot = self.board.get_spot(spot_id)
                    if spot.player is not None and hex_id in spot.adjacent_hex_ids:
                        amount = 1
                        if spot.settlement_type == SettlementType.CITY:
                            amount = 2
                        player = self.players[spot.player - 1]
                        player.add_resource(hex_obj.resource, amount)
    
    def user_can_end_turn(self):
        if not (self.rolled_dice and self.current_phase == GamePhase.REGULAR_PLAY):
            return False
        
        if not self.is_current_player_human():
            return False
        
        # Can't end turn if awaiting robber placement or other dev card actions
        if self.awaiting_robber_placement or self.awaiting_resource_selection or self.awaiting_monopoly_selection:
            return False
            
        # Can't end turn during road building if not placed both roads yet
        if self.road_building_roads_placed > 0 and self.road_building_roads_placed < 2:
            return False
        
        return True
        
    def end_turn(self):
        if not (self.rolled_dice and self.current_phase == GamePhase.REGULAR_PLAY):
            return False

        if "end_turn" not in self.possible_moves:
            return False
        
        # Reset turn flags
        self.rolled_dice = False
        self.dev_card_played_this_turn = False
        self.road_building_roads_placed = 0
        
        # Reset the development card purchase flag
        self.get_current_player().reset_dev_card_purchase_flag()
        
        # Move to next player
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players
        self.possible_moves = self.get_possible_moves()
        return True


    def get_possible_moves(self):
        moves = set()

        if self.current_phase != GamePhase.REGULAR_PLAY:
            # Only considering regular play moves for now
            return moves
            
        # Special states that limit available moves
        if self.awaiting_robber_placement:
            # Only allow robber placement
            for hex_id in self.board.hexes.keys():
                if hex_id != self.robber_hex_id:
                    moves.add(("move_robber", hex_id))
            return moves
        
        if self.awaiting_resource_selection:
            # Only allow resource selection for Year of Plenty
            for resource in [Resource.WOOD, Resource.BRICK, Resource.WHEAT, Resource.SHEEP, Resource.ORE]:
                moves.add(("select_resource", resource))
            return moves
            
        if self.awaiting_monopoly_selection:
            # Only allow resource selection for Monopoly
            for resource in [Resource.WOOD, Resource.BRICK, Resource.WHEAT, Resource.SHEEP, Resource.ORE]:
                moves.add(("select_monopoly", resource))
            return moves
            
        # If in road building mode and still have roads to place
        if 0 < self.road_building_roads_placed < 2:
            # Only allow road placement
            for road_id, road in self.board.roads.items():
                if road.owner is None and self._is_road_connected(road_id):
                    moves.add(("free_road", road_id))
            return moves
        
        curr_player = self.get_current_player()

        if not self.rolled_dice:
            moves.add("roll_dice")
            
            # Can play development cards before rolling
            if not self.dev_card_played_this_turn:
                # Check for knight cards that can be played before rolling
                knight_indices = [i for i, card in enumerate(curr_player.dev_cards) 
                                if card.card_type == DevCardType.KNIGHT and
                                (not curr_player.just_purchased_dev_card or i < len(curr_player.dev_cards) - 1)]
                
                if knight_indices:
                    moves.add("play_knight")
            
            return moves
        else:
            # Can always end turn if dice have been rolled
            moves.add("end_turn")
        
        # Buy a development card
        if not self.dev_card_deck.is_empty() and curr_player.has_dev_card_resources():
            moves.add("buy_dev_card")
        
        # Play development cards (if not already played one this turn)
        if not self.dev_card_played_this_turn:
            # Check for playable cards (excluding just purchased)
            check_cards = curr_player.dev_cards[:-1] if curr_player.just_purchased_dev_card else curr_player.dev_cards
            
            has_knight = any(card.card_type == DevCardType.KNIGHT for card in check_cards)
            has_road_building = any(card.card_type == DevCardType.ROAD_BUILDING for card in check_cards)
            has_year_of_plenty = any(card.card_type == DevCardType.YEAR_OF_PLENTY for card in check_cards)
            has_monopoly = any(card.card_type == DevCardType.MONOPOLY for card in check_cards)
            
            # Add corresponding actions if cards are available
            if has_knight:
                moves.add("play_knight")
            if has_road_building and len(curr_player.roads) < curr_player.MAX_ROADS - 1:  # Need space for 2 roads
                moves.add("play_road_building")
            if has_year_of_plenty:
                moves.add("play_year_of_plenty")
            if has_monopoly:
                moves.add("play_monopoly")
        
        # Building actions
        
        # Build settlements (if under limit)
        if curr_player.has_settlement_resources() and len(curr_player.settlements) < curr_player.MAX_SETTLEMENTS:
            for spot_id, spot in self.board.spots.items():
                if spot.player is None:  # Spot is unoccupied
                    if self._has_adjacent_road(spot_id, curr_player.player_id) and self.is_two_spots_away_from_settlement(spot_id):
                        moves.add(("build_settlement", spot_id))
        
        # Upgrade to cities (if under limit)
        if curr_player.has_city_resources() and hasattr(curr_player, 'cities') and len(curr_player.cities) < curr_player.MAX_CITIES:
            for spot_id in curr_player.settlements:
                spot = self.board.get_spot(spot_id)
                if spot and spot.settlement_type == SettlementType.SETTLEMENT:
                    moves.add(("upgrade_city", spot_id))
        
        # Build roads (if under limit)
        if curr_player.has_road_resources() and len(curr_player.roads) < curr_player.MAX_ROADS:
            for road_id, road in self.board.roads.items():
                if road.owner is None and self._is_road_connected(road_id):
                    moves.add(("road", road_id))
        
        # Make sure end_turn is always available during regular play after rolling dice
        if self.rolled_dice:
            moves.add("end_turn")
        
        return moves
    
    def _is_road_connected(self, road_id):
        curr_player = self.get_current_player()
        road = self.board.get_road(road_id)
        
        if road is None:
            return False
        
        # Check if road connects to a settlement/city owned by the player
        spot1 = self.board.get_spot(road.spot1_id)
        spot2 = self.board.get_spot(road.spot2_id)
        
        if (spot1 and spot1.player == curr_player.player_id) or (spot2 and spot2.player == curr_player.player_id):
            return True
        
        # Check if road connects to an existing road
        for r_id in curr_player.roads:
            r = self.board.get_road(r_id)
            if r and (r.spot1_id == road.spot1_id or r.spot1_id == road.spot2_id or
                    r.spot2_id == road.spot1_id or r.spot2_id == road.spot2_id):
                return True
        
        return False

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
    def _has_adjacent_road(self, spot_id, player_id):
        """Check if the spot is adjacent to a road owned by the player"""
        for road_id, road in self.board.roads.items():
            if road.owner == player_id:
                if spot_id in (road.spot1_id, road.spot2_id):
                    return True
        return False
    def build_settlement(self, spot_id):
        """Build a settlement at a spot"""
        if ("build_settlement", spot_id) not in self.possible_moves:
            return False
        
        spot = self.board.get_spot(spot_id)
        player = self.get_current_player()
        
        spot.build_settlement(player.player_id)
        player.buy_settlement()
        player.add_settlement(spot_id)
        
        self.possible_moves = self.get_possible_moves()
        return True
    def upgrade_to_city(self, spot_id):
        """Upgrade a settlement to a city"""
        if ("upgrade_city", spot_id) not in self.possible_moves:
            return False
        
        spot = self.board.get_spot(spot_id)
        player = self.get_current_player()
        
        spot.build_settlement(player.player_id, SettlementType.CITY)
        player.buy_city()
        player.add_city(spot_id)
        
        self.possible_moves = self.get_possible_moves()
        return True
    
    def is_setup_complete(self):
        """Check if the setup phase is complete"""
        return self.current_phase == GamePhase.REGULAR_PLAY
    def do_move(self, move):
        if move not in self.possible_moves:
            return False
        """Execute a game move"""
        if not isinstance(move, tuple):
            # Handle string moves
            if move == "roll_dice":
                return self.roll_dice()
            elif move == "end_turn":
                return self.end_turn()
            elif move == "buy_dev_card":
                return self.buy_development_card()
            elif move == "play_knight":
                return self.play_knight_card()
            elif move == "play_road_building":
                return self.play_road_building_card()
            elif move == "play_year_of_plenty":
                return self.play_year_of_plenty_card()
            elif move == "play_monopoly":
                return self.play_monopoly_card()
        else:
            # Handle tuple moves
            action, data = move
            if action == "build_settlement":
                return self.build_settlement(data)
            elif action == "upgrade_city":
                return self.upgrade_to_city(data)
            elif action == "road":
                return self.place_road(data)
            elif action == "free_road":
                return self.place_free_road(data)
            elif action == "select_resource":
                return self.select_year_of_plenty_resource(data)
            elif action == "select_monopoly":
                return self.select_monopoly_resource(data)
            elif action == "move_robber":
                return self.move_robber(data)
        
        return False
    

    def process_ai_turn(self):
        """Process a turn for an AI player"""

        print('here')

        if self.is_current_player_human():
            # Not an AI player, do nothing
            self.waiting_for_human_input = True
            return False
        
        # Get the current agent
        agent = self.get_current_agent()
        
        # Handle the setup phase
        if not self.is_setup_complete():
            if not self.setup_phase_settlement_placed:
                print('initial settle')
                spot_id = agent.get_initial_settlement(self)
                self.last_settlement_placed = spot_id
                return self.place_initial_settlement(spot_id)
            else:
                road_id = agent.get_initial_road(self, self.last_settlement_placed)
                self.place_initial_road(road_id, self.last_settlement_placed)
                self.last_settlement_placed = None
                return True
        
        move = agent.get_move(self)
        while move != "end_turn":
            self.do_move(move)
            move = agent.get_move(self)

        self.end_turn()

    def buy_development_card(self):
        """Buy a development card from the deck"""
        if "buy_dev_card" not in self.possible_moves:
            return False
            
        curr_player = self.get_current_player()
        
        # Check if deck is empty
        if self.dev_card_deck.is_empty():
            return False
            
        # Draw a card and give it to the player
        card = self.dev_card_deck.draw_card()
        success = curr_player.buy_dev_card(card)
        
        if success:
            self.possible_moves = self.get_possible_moves()
            return True
        
        return False

    def play_knight_card(self):
        """Play a knight development card"""
        if "play_knight" not in self.possible_moves:
            return False
            
        curr_player = self.get_current_player()
        
        # Find a knight card in the player's hand (not just purchased)
        knight_indices = [i for i, card in enumerate(curr_player.dev_cards) 
                        if card.card_type == DevCardType.KNIGHT and 
                        (not curr_player.just_purchased_dev_card or i < len(curr_player.dev_cards) - 1)]
        
        if not knight_indices:
            return False
            
        # Play the knight card
        card = curr_player.play_dev_card(knight_indices[0])
        if not card:
            return False
            
        # Set flags and activate robber
        self.dev_card_played_this_turn = True
        self.awaiting_robber_placement = True
        
        # Check for largest army
        if curr_player.knights_played >= 3 and (self.largest_army_player is None or 
                                            curr_player.knights_played > self.largest_army_size):
            self.largest_army_player = curr_player.player_id
            self.largest_army_size = curr_player.knights_played
        
        self.possible_moves = self.get_possible_moves()
        return True

    def play_road_building_card(self):
        """Play a road building development card"""
        if "play_road_building" not in self.possible_moves:
            return False
            
        curr_player = self.get_current_player()
        
        # Find a road building card in the player's hand (not just purchased)
        road_indices = [i for i, card in enumerate(curr_player.dev_cards) 
                    if card.card_type == DevCardType.ROAD_BUILDING and 
                    (not curr_player.just_purchased_dev_card or i < len(curr_player.dev_cards) - 1)]
        
        if not road_indices:
            return False
            
        # Play the road building card
        card = curr_player.play_dev_card(road_indices[0])
        if not card:
            return False
            
        # Set flags and wait for road placement
        self.dev_card_played_this_turn = True
        self.road_building_roads_placed = 0
        
        self.possible_moves = self.get_possible_moves()
        return True

    def play_year_of_plenty_card(self):
        """Play a year of plenty development card"""
        if "play_year_of_plenty" not in self.possible_moves:
            return False
            
        curr_player = self.get_current_player()
        
        # Find a year of plenty card in the player's hand (not just purchased)
        yop_indices = [i for i, card in enumerate(curr_player.dev_cards) 
                    if card.card_type == DevCardType.YEAR_OF_PLENTY and 
                    (not curr_player.just_purchased_dev_card or i < len(curr_player.dev_cards) - 1)]
        
        if not yop_indices:
            return False
            
        # Play the year of plenty card
        card = curr_player.play_dev_card(yop_indices[0])
        if not card:
            return False
            
        # Set flags and wait for resource selection
        self.dev_card_played_this_turn = True
        self.awaiting_resource_selection = True
        self.awaiting_resource_selection_count = 2  # Select 2 resources
        
        self.possible_moves = self.get_possible_moves()
        return True

    def play_monopoly_card(self):
        """Play a monopoly development card"""
        if "play_monopoly" not in self.possible_moves:
            return False
            
        curr_player = self.get_current_player()
        
        # Find a monopoly card in the player's hand (not just purchased)
        monopoly_indices = [i for i, card in enumerate(curr_player.dev_cards) 
                        if card.card_type == DevCardType.MONOPOLY and 
                        (not curr_player.just_purchased_dev_card or i < len(curr_player.dev_cards) - 1)]
        
        if not monopoly_indices:
            return False
            
        # Play the monopoly card
        card = curr_player.play_dev_card(monopoly_indices[0])
        if not card:
            return False
            
        # Set flags and wait for resource selection
        self.dev_card_played_this_turn = True
        self.awaiting_monopoly_selection = True
        
        self.possible_moves = self.get_possible_moves()
        return True

    def select_year_of_plenty_resource(self, resource):
        """Select a resource for Year of Plenty"""
        if not self.awaiting_resource_selection:
            return False
            
        curr_player = self.get_current_player()
        curr_player.add_resource(resource, 1)
        
        self.awaiting_resource_selection_count -= 1
        if self.awaiting_resource_selection_count <= 0:
            self.awaiting_resource_selection = False
            
        self.possible_moves = self.get_possible_moves()
        return True

    def select_monopoly_resource(self, resource):
        """Select a resource for Monopoly and steal from other players"""
        if not self.awaiting_monopoly_selection:
            return False
            
        curr_player = self.get_current_player()
        
        # Steal the selected resource from all other players
        for player in self.players:
            if player.player_id != curr_player.player_id:
                amount = player.resources[resource]
                player.resources[resource] = 0
                curr_player.add_resource(resource, amount)
        
        self.awaiting_monopoly_selection = False
        self.possible_moves = self.get_possible_moves()
        return True

    def place_free_road(self, road_id):
        """Place a free road during road building"""
        curr_player = self.get_current_player()
        
        if self.road_building_roads_placed >= 2:
            return False
            
        road = self.board.get_road(road_id)
        
        # Check if road is already claimed
        if not road or road.owner is not None:
            return False
            
        # Check connectivity: the road must touch a settlement or another road owned by the player
        touching_settlement = False
        for spot_id in (road.spot1_id, road.spot2_id):
            spot = self.board.get_spot(spot_id)
            if spot and spot.player == curr_player.player_id:
                touching_settlement = True
                break
                
        touching_road = False
        for r_id in curr_player.roads:
            existing_road = self.board.get_road(r_id)
            if existing_road:
                if road.spot1_id in (existing_road.spot1_id, existing_road.spot2_id) or \
                road.spot2_id in (existing_road.spot1_id, existing_road.spot2_id):
                    touching_road = True
                    break
                    
        if not (touching_settlement or touching_road):
            return False
            
        # Place road without cost
        road.build_road(curr_player.player_id)
        curr_player.add_road(road_id)
        
        self.road_building_roads_placed += 1
        self.possible_moves = self.get_possible_moves()
        
        return True
    
    def move_robber(self, hex_id):
        """Move the robber to a new hex and prepare for stealing"""
        if not self.awaiting_robber_placement:
            return False
        
        if hex_id == self.robber_hex_id:
            return False  # Can't place robber on same hex
        
        hex_obj = self.board.get_hex(hex_id)
        if not hex_obj:
            return False
        
        self.robber_hex_id = hex_id
        self.awaiting_robber_placement = False
        
        # Find potential victims (players with settlements adjacent to this hex)
        current_player = self.get_current_player()
        potential_victims = []
        
        for spot_id, spot in self.board.spots.items():
            if hex_id in spot.adjacent_hex_ids and spot.player is not None:
                # Don't steal from yourself and don't include duplicates
                if spot.player != current_player.player_id and spot.player not in potential_victims:
                    # Only include players who have resources
                    victim = self.players[spot.player - 1]
                    if sum(victim.resources.values()) > 0:
                        potential_victims.append(spot.player)
        
        # If AI player, steal immediately
        if not self.is_current_player_human():
            if potential_victims:
                self.steal_resource_from_player(random.choice(potential_victims))
        else:
            # For human player, set up UI selection
            self.awaiting_steal_selection = True
            self.potential_victims = potential_victims
        
        self.possible_moves = self.get_possible_moves()
        return True

    def steal_resource_from_player(self, victim_id):
        """Steal a random resource from the specified player"""
        if victim_id not in range(1, len(self.players) + 1):
            return False
            
        current_player = self.get_current_player()
        victim = self.players[victim_id - 1]
        
        # Create a list of resources the victim has
        available_resources = []
        for resource, count in victim.resources.items():
            available_resources.extend([resource] * count)
        
        # Steal a random resource
        if available_resources:
            stolen_resource = random.choice(available_resources)
            victim.resources[stolen_resource] -= 1
            current_player.add_resource(stolen_resource, 1)
            
            print(f"Player {current_player.player_id} stole {stolen_resource.name} from Player {victim_id}")
            
        self.awaiting_steal_selection = False
        return True
    
    def create_agent(player_id, agent_type):
        if agent_type == AgentType.HUMAN:
            return None
        elif agent_type == AgentType.RANDOM:
            from agent.random_agent import RandomAgent
            return RandomAgent(player_id)
        elif agent_type == AgentType.HEURISTIC:
            from agent.simple_heuristic_agent import SimpleHeuristicAgent
            return SimpleHeuristicAgent(player_id)
        else:
            print("Unsupported agent type")