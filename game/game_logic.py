# game/game_logic.py
from .enums import GamePhase
from game.game_state import GameState
from .board import Board
from .spot import SettlementType
from .enums import Resource
from .player import Player
import random
from agent.random_agent import RandomAgent
from agent.simple_heuristic_agent import SimpleHeuristicAgent
from .development_card import DevelopmentCard, DevelopmentCardDeck, DevCardType


from agent.base import AgentType, create_agent

class GameLogic:
    def __init__(self, board, agent_types=None):
        self.state = GameState(board)
        self.agents = []
        
        # Default all agents to human if not specified
        if agent_types is None:
            agent_types = [AgentType.HUMAN] * 4
        

        for i in range(4):
            agent_type = agent_types[i]
            player = Player(i, f"Player {i} {agent_type}")
            player.is_human = agent_type == AgentType.HUMAN
            self.state.players.append(player)
            self.agents.append(create_agent(i, agent_type))

    
    def get_current_agent(self):
        """Returns the current agent object"""
        return self.agents[self.state.current_player_idx]
    
    def is_current_player_human(self):
        """Check if current player is human"""
        return self.get_current_agent().is_human()
    
    def is_valid_initial_settlement(self, spot_id):
        """Check if a spot is valid for initial settlement placement"""
        spot = self.state.board.get_spot(spot_id)
        
        # Make sure the spot exists and is free
        if spot is None or spot.player_idx is not None:
            return False

        # Check distance rule
        return self.is_two_spots_away_from_settlement(spot_id)
    
    def roll_dice(self):
        state = self.state

        if state.current_phase != GamePhase.REGULAR_PLAY:
            return False
        
        if "roll_dice" not in state.possible_actions:
            return False
        
        state.dice1_roll = random.randint(1, 6)
        state.dice2_roll = random.randint(1, 6)

        dice_sum = state.dice1_roll + state.dice2_roll
        state.rolled_dice = True
        
        # Check for "7" - activate robber
        if dice_sum == 7:
            self._handle_robber_roll()
        else:
            # Distribute resources normally
            self.distribute_resources(dice_sum)

        state.possible_actions = self.get_possible_actions()
        return True

    def _handle_robber_roll(self):
        """Handle the effects of rolling a 7 (robber activation)"""

        state = self.state

        # First, all players with more than 7 cards discard half, rounded down
        for player in state.players:
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
        state.awaiting_robber_placement = True

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

        state = self.state
        for hex_id, hex_obj in state.board.hexes.items():

            # Skip hexes where the robber is located
            if hex_obj.number == dice_result and hex_id != state.robber_hex_id:
                for spot_id in state.board.spots:
                    spot = state.board.get_spot(spot_id)
                    if spot.player_idx is not None and hex_id in spot.adjacent_hex_ids:
                        amount = 1
                        if spot.settlement_type == SettlementType.CITY:
                            amount = 2

                        player = self.state.players[spot.player_idx]
                        player.add_resource(hex_obj.resource, amount)
    
    def user_can_end_turn(self):
        state = self.state
        if not (state.rolled_dice and state.current_phase == GamePhase.REGULAR_PLAY):
            return False
        
        if not self.is_current_player_human():
            return False
        
        # Can't end turn if awaiting robber placement or other dev card actions
        if state.awaiting_robber_placement or state.awaiting_resource_selection or state.awaiting_monopoly_selection:
            return False
            
        # Can't end turn during road building if not placed both roads yet
        if state.road_building_roads_placed > 0 and state.road_building_roads_placed < 2:
            return False
        
        return True
        
    def end_turn(self):
        state = self.state
        if not (state.rolled_dice and state.current_phase == GamePhase.REGULAR_PLAY):
            return False

        if "end_turn" not in state.possible_actions:
            return False
        
        # Reset turn flags
        state.rolled_dice = False
        state.dev_card_played_this_turn = False
        state.road_building_roads_placed = 0
        
        # Reset the development card purchase flag
        state.get_current_player().reset_dev_card_purchase_flag()
        
        # Move to next player
        state.current_player_idx = (state.current_player_idx + 1) % 4
        return True


    def get_possible_actions(self):
        state = self.state

        actions = set()

        if state.current_phase != GamePhase.REGULAR_PLAY:
            # Only considering regular play moves for now
            return actions
        
        # Special states that limit available moves
        if state.awaiting_robber_placement:
            # Only allow robber placement
            for hex_id in state.board.hexes.keys():
                if hex_id != state.robber_hex_id:
                    actions.add(("move_robber", hex_id))
            return actions
        
        if state.awaiting_resource_selection:
            # Only allow resource selection for Year of Plenty
            for resource in [Resource.WOOD, Resource.BRICK, Resource.WHEAT, Resource.SHEEP, Resource.ORE]:
                actions.add(("select_resource", resource))
            return actions
            
        if state.awaiting_monopoly_selection:
            # Only allow resource selection for Monopoly
            for resource in [Resource.WOOD, Resource.BRICK, Resource.WHEAT, Resource.SHEEP, Resource.ORE]:
                actions.add(("select_monopoly", resource))
            return actions
            
        # If in road building mode and still have roads to place
        if 0 < state.road_building_roads_placed < 2:
            # Only allow road placement
            for road_id, road in state.board.roads.items():
                if road.owner is None and self._is_road_connected(road_id):
                    actions.add(("free_road", road_id))
            return actions
        
        curr_player = state.get_current_player()


        if not state.rolled_dice:
            actions.add("roll_dice")
            
            # Can play development cards before rolling
            if not state.dev_card_played_this_turn:
                # Check for knight cards that can be played before rolling
                knight_indices = [i for i, card in enumerate(curr_player.dev_cards) 
                                if card.card_type == DevCardType.KNIGHT and
                                (not curr_player.just_purchased_dev_card or i < len(curr_player.dev_cards) - 1)]
                
                if knight_indices:
                    actions.add("play_knight")
            
            return actions
        else:
            # Can always end turn if dice have been rolled
            actions.add("end_turn")
        
        # Buy a development card
        if not state.dev_card_deck.is_empty() and curr_player.has_dev_card_resources():
            actions.add("buy_dev_card")
        
        # Play development cards (if not already played one this turn)
        if not state.dev_card_played_this_turn:
            # Check for playable cards (excluding just purchased)
            check_cards = curr_player.dev_cards[:-1] if curr_player.just_purchased_dev_card else curr_player.dev_cards
            
            has_knight = any(card.card_type == DevCardType.KNIGHT for card in check_cards)
            has_road_building = any(card.card_type == DevCardType.ROAD_BUILDING for card in check_cards)
            has_year_of_plenty = any(card.card_type == DevCardType.YEAR_OF_PLENTY for card in check_cards)
            has_monopoly = any(card.card_type == DevCardType.MONOPOLY for card in check_cards)
            
            # Add corresponding actions if cards are available
            if has_knight:
                actions.add("play_knight")
            if has_road_building and len(curr_player.roads) < curr_player.MAX_ROADS - 1:  # Need space for 2 roads
                actions.add("play_road_building")
            if has_year_of_plenty:
                actions.add("play_year_of_plenty")
            if has_monopoly:
                actions.add("play_monopoly")
        
        # Building actions
        
        # Build settlements (if under limit)
        if curr_player.has_settlement_resources() and len(curr_player.settlements) < curr_player.MAX_SETTLEMENTS:
            for spot_id, spot in state.board.spots.items():
                if spot.player_idx is None:  # Spot is unoccupied
                    if self._has_adjacent_road(spot_id, curr_player.player_idx) and self.is_two_spots_away_from_settlement(spot_id):
                        actions.add(("build_settlement", spot_id))
        
        # Upgrade to cities (if under limit)
        if curr_player.has_city_resources() and hasattr(curr_player, 'cities') and len(curr_player.cities) < curr_player.MAX_CITIES:
            for spot_id in curr_player.settlements:
                spot = state.board.get_spot(spot_id)
                if spot and spot.settlement_type == SettlementType.SETTLEMENT:
                    actions.add(("upgrade_city", spot_id))
        
        # Build roads (if under limit)
        if curr_player.has_road_resources() and len(curr_player.roads) < curr_player.MAX_ROADS:
            for road_id, road in state.board.roads.items():
                if road.owner is None and self._is_road_connected(road_id):
                    actions.add(("road", road_id))
        
        # Make sure end_turn is always available during regular play after rolling dice
        if state.rolled_dice:
            actions.add("end_turn")
        
        return actions
    
    def _is_road_connected(self, road_id):
        state = self.state
        board = state.board

        curr_player = state.get_current_player()
        road = board.get_road(road_id)
        
        if road is None:
            return False
        
        # Check if road connects to a settlement/city owned by the player
        spot1 = board.get_spot(road.spot1_id)
        spot2 = board.get_spot(road.spot2_id)
        
        if (spot1 and spot1.player_idx == curr_player.player_idx) or (spot2 and spot2.player_idx == curr_player.player_idx):
            return True
        
        # Check if road connects to an existing road
        for r_id in curr_player.roads:
            r = board.get_road(r_id)
            if r and (r.spot1_id == road.spot1_id or r.spot1_id == road.spot2_id or
                    r.spot2_id == road.spot1_id or r.spot2_id == road.spot2_id):
                return True
        
        return False

    # checks that our spot is touching a road we built
    def has_adjascent_road(self, spot_id):
        state = self.state

        curr_player = state.get_current_player()

        has_adjacent_road = False
        for r_id in curr_player.roads:
            road = state.board.get_road(r_id)
            if road and spot_id in (road.spot1_id, road.spot2_id):
                has_adjacent_road = True
                break
        
        return has_adjacent_road

    # distance rule
    # checks that a spot is 2 away
    def is_two_spots_away_from_settlement(self, spot_id):
        state = self.state

        for adjacent_road in state.board.roads.values():
            if spot_id == adjacent_road.spot1_id:
                adjacent_spot = state.board.spots.get(adjacent_road.spot2_id)
                if adjacent_spot.player_idx is not None:
                    return False
            elif spot_id == adjacent_road.spot2_id:
                adjacent_spot = state.board.spots.get(adjacent_road.spot1_id)
                if adjacent_spot.player_idx is not None:
                    return False
        return True


    def place_road(self, road_id):
        state = self.state

        if ("road", road_id) not in state.possible_actions:
            return False
        
        new_road = state.board.get_road(road_id)
        curr_player = state.get_current_player()

        curr_player.buy_road()
        new_road.build_road(curr_player.player_idx)
        curr_player.add_road(road_id)

        return True
        

    def place_initial_settlement(self, spot_id):
        """
        Place an initial settlement during setup phase
        Returns True if successful, False otherwise
        """
        state = self.state

        if not self.is_valid_initial_settlement(spot_id):
            return False
        
        player = state.get_current_player()
        spot = state.board.get_spot(spot_id)
        
        # Place settlement
        spot.build_settlement(player.player_idx, SettlementType.SETTLEMENT)
        player.add_settlement(spot_id)
        
        # Update game state
        state.setup_phase_settlement_placed = True
        
        # If in second setup phase, give resources for adjacent hexes
        if state.current_phase == GamePhase.SETUP_PHASE_2:
            self._give_initial_resources(spot_id, player)
            print(f"Giving resources to {player.name} for second settlement")
            for resource, count in player.resources.items():
                if count > 0:
                    print(f"  - {resource.name}: {count}")
        
        return True
    
    def _give_initial_resources(self, spot_id, player):
        """Give resources for hexes adjacent to the second settlement"""
        state = self.state

        spot = state.board.get_spot(spot_id)
        for hex_id in spot.adjacent_hex_ids:
            hex_obj = state.board.get_hex(hex_id)
            # Don't give resources for desert
            if hex_obj.resource != Resource.DESERT:
                player.add_resource(hex_obj.resource, 1)
    
    def is_valid_initial_road(self, road_id, last_settlement_id):
        """
        Check if a road placement is valid in setup phase
        The road must be connected to the last settlement placed
        """
        state = self.state

        road = state.board.get_road(road_id)
        
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
        state = self.state

        if not self.is_valid_initial_road(road_id, last_settlement_id):
            return False
        
        player = state.get_current_player()
        road = state.board.get_road(road_id)
        
        # Place road
        road.build_road(player.player_idx)
        player.add_road(road_id)
        
        # Advance to next player or phase
        self._advance_setup_phase()
        
        return True
    
    def _advance_setup_phase(self):
        """Advance to the next player or phase in setup"""
        state = self.state

        # Reset the settlement placement flag
        state.setup_phase_settlement_placed = False
        state.rolled_dice = False
        
        if state.current_phase == GamePhase.SETUP_PHASE_1:
            # If we've gone through all players, switch to phase 2 (reverse order)
            if state.current_player_idx == 3:
                state.current_phase = GamePhase.SETUP_PHASE_2
            else:
                state.current_player_idx += 1   

        elif state.current_phase == GamePhase.SETUP_PHASE_2:
            # If we've gone through all players in reverse order
            if state.current_player_idx == 0:
                state.current_phase = GamePhase.REGULAR_PLAY
                state.possible_actions = self.get_possible_actions()
            else: 
                state.current_player_idx -= 1

    def _has_adjacent_road(self, spot_id, player_idx):
        """Check if the spot is adjacent to a road owned by the player"""
        state = self.state

        for road_id, road in state.board.roads.items():
            if road.owner == player_idx:
                if spot_id in (road.spot1_id, road.spot2_id):
                    return True
        return False
    
    def build_settlement(self, spot_id):
        """Build a settlement at a spot"""
        state = self.state

        if ("build_settlement", spot_id) not in state.possible_actions:
            return False
        
        spot = state.board.get_spot(spot_id)
        player = state.get_current_player()
        
        spot.build_settlement(player.player_idx)
        player.buy_settlement()
        player.add_settlement(spot_id)
        
        return True
    
    def upgrade_to_city(self, spot_id):
        """Upgrade a settlement to a city"""
        state = self.state

        if ("upgrade_city", spot_id) not in state.possible_actions:
            return False
        
        spot = state.board.get_spot(spot_id)
        player = state.get_current_player()
        
        spot.build_settlement(player.player_idx, SettlementType.CITY)
        player.buy_city()
        player.add_city(spot_id)
        
        return True
    
    def is_setup_complete(self):
        """Check if the setup phase is complete"""
        return self.state.current_phase == GamePhase.REGULAR_PLAY
    
    def do_action(self, move):
        """Execute a game move"""
        state = self.state

        if move not in state.possible_actions:
            return False
        
        
        if not isinstance(move, tuple):
            # Handle string moves
            if move == "roll_dice":
                success =  self.roll_dice()
            elif move == "end_turn":
                success = self.end_turn()
            elif move == "buy_dev_card":
                success =  self.buy_development_card()
            elif move == "play_knight":
                success = self.play_knight_card()
            elif move == "play_road_building":
                success = self.play_road_building_card()
            elif move == "play_year_of_plenty":
                success = self.play_year_of_plenty_card()
            elif move == "play_monopoly":
                success = self.play_monopoly_card()
        else:
            # Handle tuple moves
            action, data = move
            if action == "build_settlement":
                success = self.build_settlement(data)
            elif action == "upgrade_city":
                success = self.upgrade_to_city(data)
            elif action == "road":
                success = self.place_road(data)
            elif action == "free_road":
                success = self.place_free_road(data)
            elif action == "select_resource":
                success = self.select_year_of_plenty_resource(data)
            elif action == "select_monopoly":
                success = self.select_monopoly_resource(data)
            elif action == "move_robber":
                success = self.move_robber(data)
        
        
        if success:
            self.state.possible_actions = self.get_possible_actions()

        return success
    

    def process_ai_turn(self):
        """Process a turn for an AI player"""
        state = self.state

        if self.is_current_player_human():
            # Not an AI player, do nothing
            state.waiting_for_human_input = True
            return False
        
        # Get the current agent
        agent = self.get_current_agent()
        
        # Handle the setup phase
        if not self.is_setup_complete():
            if not state.setup_phase_settlement_placed:
                spot_id = agent.get_initial_settlement(self)
                state.last_settlement_placed = spot_id
                return self.place_initial_settlement(spot_id)
            else:
                road_id = agent.get_initial_road(self, state.last_settlement_placed)
                self.place_initial_road(road_id, state.last_settlement_placed)
                state.last_settlement_placed = None
                return True
        
        action = agent.get_action(self)
        while action != "end_turn":
            self.do_action(action)
            action = agent.get_action(self)

        self.do_action("end_turn")

    def buy_development_card(self):
        """Buy a development card from the deck"""
        state = self.state

        if "buy_dev_card" not in state.possible_actions:
            return False
            
        curr_player = state.get_current_player()
        
        # Check if deck is empty
        if state.dev_card_deck.is_empty():
            return False
            
        # Draw a card and give it to the player
        card = state.dev_card_deck.draw_card()
        success = curr_player.buy_dev_card(card)
        
        return success


    def play_knight_card(self):
        """Play a knight development card"""
        state = self.state

        if "play_knight" not in state.possible_actions:
            return False
            
        curr_player = state.get_current_player()
        
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
        state.dev_card_played_this_turn = True
        state.awaiting_robber_placement = True
        
        # Check for largest army
        if curr_player.knights_played >= 3 and (state.largest_army_player is None or 
                                            curr_player.knights_played > state.largest_army_size):
            state.largest_army_player = curr_player.player_idx
            state.largest_army_size = curr_player.knights_played
        
        state.possible_actions = state.get_possible_actions()
        return True

    def play_road_building_card(self):
        """Play a road building development card"""
        state = self.state

        if "play_road_building" not in state.possible_actions:
            return False
            
        curr_player = state.get_current_player()
        
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
        state.dev_card_played_this_turn = True
        state.road_building_roads_placed = 0
        
        return True

    def play_year_of_plenty_card(self):
        """Play a year of plenty development card"""
        state = self.state

        if "play_year_of_plenty" not in state.possible_actions:
            return False
            
        curr_player = state.get_current_player()
        
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
        state.dev_card_played_this_turn = True
        state.awaiting_resource_selection = True
        state.awaiting_resource_selection_count = 2  # Select 2 resources
        
        return True

    def play_monopoly_card(self):
        """Play a monopoly development card"""
        state = self.state

        if "play_monopoly" not in state.possible_actions:
            return False
        
        state = self.state
            
        curr_player = state.get_current_player()
        
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
        state.dev_card_played_this_turn = True
        state.awaiting_monopoly_selection = True
        
        return True

    def select_year_of_plenty_resource(self, resource):
        """Select a resource for Year of Plenty"""
        state = self.state

        if not state.awaiting_resource_selection:
            return False
            
        curr_player = state.get_current_player()
        curr_player.add_resource(resource, 1)
        
        state.awaiting_resource_selection_count -= 1
        if state.awaiting_resource_selection_count <= 0:
            state.awaiting_resource_selection = False
            
        return True

    def select_monopoly_resource(self, resource):
        """Select a resource for Monopoly and steal from other players"""
        state = self.state

        if not state.awaiting_monopoly_selection:
            return False
            
        curr_player = state.get_current_player()
        
        # Steal the selected resource from all other players
        for player in state.players:
            if player.player_idx != curr_player.player_idx:
                amount = player.resources[resource]
                player.resources[resource] = 0
                curr_player.add_resource(resource, amount)
        
        state.awaiting_monopoly_selection = False
        return True

    def place_free_road(self, road_id):
        """Place a free road during road building"""
        state = self.state

        curr_player = state.get_current_player()
        
        if state.road_building_roads_placed >= 2:
            return False
            
        road = state.board.get_road(road_id)
        
        # Check if road is already claimed
        if not road or road.owner is not None:
            return False
            
        # Check connectivity: the road must touch a settlement or another road owned by the player
        touching_settlement = False
        for spot_id in (road.spot1_id, road.spot2_id):
            spot = state.board.get_spot(spot_id)
            if spot and spot.player_idx == curr_player.player_idx:
                touching_settlement = True
                break
                
        touching_road = False
        for r_id in curr_player.roads:
            existing_road = state.board.get_road(r_id)
            if existing_road:
                if road.spot1_id in (existing_road.spot1_id, existing_road.spot2_id) or \
                road.spot2_id in (existing_road.spot1_id, existing_road.spot2_id):
                    touching_road = True
                    break
                    
        if not (touching_settlement or touching_road):
            return False
            
        # Place road without cost
        road.build_road(curr_player.player_idx)
        curr_player.add_road(road_id)
        
        state.road_building_roads_placed += 1
        
        return True
    
    def move_robber(self, hex_id):
        """Move the robber to a new hex and prepare for stealing"""
        state = self.state

        if not state.awaiting_robber_placement:
            return False
        
        if hex_id == state.robber_hex_id:
            return False  # Can't place robber on same hex
        
        hex_obj = state.board.get_hex(hex_id)
        if not hex_obj:
            return False
        
        state.robber_hex_id = hex_id
        state.awaiting_robber_placement = False
        
        # Find potential victims (players with settlements adjacent to this hex)
        current_player = state.get_current_player()
        potential_victims = []
        
        for spot_id, spot in state.board.spots.items():
            if hex_id in spot.adjacent_hex_ids and spot.player_idx is not None:
                # Don't steal from yourself and don't include duplicates
                if spot.player_idx != current_player.player_idx and spot.player_idx not in potential_victims:
                    # Only include players who have resources
                    victim = state.players[spot.player_idx - 1]
                    if sum(victim.resources.values()) > 0:
                        potential_victims.append(spot.player_idx)
        
        # If AI player, steal immediately
        if not self.is_current_player_human():
            if potential_victims:
                self.steal_resource_from_player(random.choice(potential_victims))
        else:
            # For human player, set up UI selection
            self.awaiting_steal_selection = True
            self.state.potential_victims = potential_victims
        
        return True

    def steal_resource_from_player(self, victim_idx):
        """Steal a random resource from the specified player"""
        state = self.state

        if victim_idx not in range(4):
            return False
            
        current_player = state.get_current_player()
        victim = state.players[victim_idx]
        
        # Create a list of resources the victim has
        available_resources = []
        for resource, count in victim.resources.items():
            available_resources.extend([resource] * count)
        
        # Steal a random resource
        if available_resources:
            stolen_resource = random.choice(available_resources)
            victim.resources[stolen_resource] -= 1
            current_player.add_resource(stolen_resource, 1)
            
            print(f"Player {current_player.player_idx} stole {stolen_resource.name} from Player {victim_idx}")
            
        state.awaiting_steal_selection = False
        return True
    
    def create_agent(player_idx, agent_type):
        if agent_type == AgentType.HUMAN:
            return None
        elif agent_type == AgentType.RANDOM:
            from agent.random_agent import RandomAgent
            return RandomAgent(player_idx)
        elif agent_type == AgentType.HEURISTIC:
            from agent.simple_heuristic_agent import SimpleHeuristicAgent
            return SimpleHeuristicAgent(player_idx)
        else:
            print("Unsupported agent type")