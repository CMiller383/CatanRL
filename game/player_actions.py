import random
from game.enums import DevCardType, GamePhase, SettlementType
from game.resource_manager import distribute_resources, handle_robber_roll
from game.road_utils import update_longest_road
from game.game_state import check_game_over

def roll_dice(state):
    if state.current_phase != GamePhase.REGULAR_PLAY:
        return False
    
    state.dice1_roll = random.randint(1, 6)
    state.dice2_roll = random.randint(1, 6)

    dice_sum = state.dice1_roll + state.dice2_roll
    state.rolled_dice = True
    
    # Check for "7" - activate robber
    if dice_sum == 7:
        handle_robber_roll(state)
    else:
        # Distribute resources normally
        distribute_resources(state, dice_sum)
    
    return True

def end_turn(state):
    if not (state.rolled_dice and state.current_phase == GamePhase.REGULAR_PLAY):
        return False
    
    # Reset turn flags
    state.rolled_dice = False
    state.dev_card_played_this_turn = False
    state.road_building_roads_placed = 0
    
    # Move to next player
    state.current_player_idx = (state.current_player_idx + 1) % 4
    state.turn_number += 1
    return True

def place_road(state, road_id):
    new_road = state.board.get_road(road_id)
    curr_player = state.get_current_player()

    curr_player.buy_road()
    new_road.build_road(curr_player.player_idx)
    curr_player.add_road(road_id)

    update_longest_road(state)

    check_game_over(state)

    return True

def build_settlement(state, spot_id):
    """Build a settlement at a spot"""
    spot = state.board.get_spot(spot_id)
    player = state.get_current_player()
    
    spot.build_settlement(player.player_idx)
    player.buy_settlement()
    player.add_settlement(spot_id)
    
    check_game_over(state)
    return True

def upgrade_to_city(state, spot_id):
    """Upgrade a settlement to a city"""
    spot = state.board.get_spot(spot_id)
    player = state.get_current_player()
    
    spot.build_settlement(player.player_idx, SettlementType.CITY)
    player.buy_city()
    player.add_city(spot_id)
    check_game_over(state)
    return True

def buy_development_card(state):
    """Buy a development card from the deck"""
        
    curr_player = state.get_current_player()
    
    # Check if deck is empty
    if state.dev_card_deck.is_empty():
        return False
        
    # Draw a card and give it to the player
    card = state.dev_card_deck.draw_card()
    card.turn_bought = state.turn_number
    success = curr_player.buy_dev_card(card)
    
    return success


def play_knight_card(state):
    """Play a knight development card"""
        
    curr_player = state.get_current_player()
    
    # Find a knight card in the player's hand (not just purchased)
    knight_indices = [i for i, card in enumerate(curr_player.dev_cards) 
                    if card.card_type == DevCardType.KNIGHT and 
                    card.turn_bought < state.turn_number]
    
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
        if state.largest_army_player:
            state.players[state.largest_army_player].victory_points -= 2
        curr_player.victory_points += 2
        state.largest_army_player = curr_player.player_idx
        state.largest_army_size = curr_player.knights_played

    return True

def play_road_building_card(state):
    """Play a road building development card"""
    curr_player = state.get_current_player()
    
    # Find a road building card in the player's hand (not just purchased)
    road_indices = [i for i, card in enumerate(curr_player.dev_cards) 
                if card.card_type == DevCardType.ROAD_BUILDING and 
                card.turn_bought < state.turn_number]
    
    if not road_indices:
        return False
        
    # Play the road building card
    card = curr_player.play_dev_card(road_indices[0])
    if not card:
        return False
        
    # Set flags and wait for road placement
    state.dev_card_played_this_turn = True
    state.road_building_roads_placed = 0
    state.awaiting_road_builder_placements = True
    
    return True

def play_year_of_plenty_card(state):
    """Play a year of plenty development card"""
        
    curr_player = state.get_current_player()
    
    # Find a year of plenty card in the player's hand (not just purchased)
    yop_indices = [i for i, card in enumerate(curr_player.dev_cards) 
                if card.card_type == DevCardType.YEAR_OF_PLENTY and 
                card.turn_bought < state.turn_number]
    
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

def play_monopoly_card(state):
    """Play a monopoly development card"""

    curr_player = state.get_current_player()
    
    # Find a monopoly card in the player's hand (not just purchased)
    monopoly_indices = [i for i, card in enumerate(curr_player.dev_cards) 
                    if card.card_type == DevCardType.MONOPOLY and 
                    card.turn_bought < state.turn_number]
    
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

def select_year_of_plenty_resource(state, resource):
    """Select a resource for Year of Plenty"""

    if not state.awaiting_resource_selection:
        return False
        
    curr_player = state.get_current_player()
    curr_player.add_resource(resource, 1)
    
    state.awaiting_resource_selection_count -= 1
    if state.awaiting_resource_selection_count <= 0:
        state.awaiting_resource_selection = False
        
    return True

def select_monopoly_resource(state, resource):
    """Select a resource for Monopoly and steal from other players"""

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

def place_free_road(state, road_id):
    """Place a free road during road building"""

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

    update_longest_road(state)
    check_game_over(state)
    
    state.road_building_roads_placed += 1
    if state.road_building_roads_placed == 2:
        state.awaiting_road_builder_placements = False
    
    return True

def move_robber(state, hex_id):
    """Move the robber to a new hex and prepare for stealing"""

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
    
    
    state.awaiting_steal_selection = True
    state.potential_victims = potential_victims
    
    return True

def steal_resource_from_player(state, victim_idx):
    """Steal a random resource from the specified player"""

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
        
        # print(f"Player {current_player.player_idx} stole {stolen_resource.name} from Player {victim_idx}")
        
    state.awaiting_steal_selection = False
    return True

def trade_resources(state, trade_payload):
    """
    Execute a 4:1 trade
    
    Args:
        state: The game state
        trade_payload: Tuple of (resource_to_give, resource_to_get)
    
    Returns:
        success: Whether the trade was successful
    """
    resource_to_give, resource_to_get = trade_payload
    player = state.get_current_player()
    
    # Check if player has enough of the resource to give
    if player.resources[resource_to_give] < 4:
        return False
    
    # Execute the trade
    player.resources[resource_to_give] -= 4
    player.resources[resource_to_get] += 1
    
    return True