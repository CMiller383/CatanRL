from game.enums import GamePhase, SettlementType
from game.possible_action_generator import get_possible_actions
from game.resource_manager import give_initial_resources
from game.rules import is_two_spots_away_from_settlement, is_valid_initial_road


def place_initial_settlement(state, spot_id):
    """
    Place an initial settlement during setup phase
    Returns True if successful, False otherwise
    """

    if not is_valid_initial_settlement(state, spot_id):
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
        give_initial_resources(state, spot_id, player)
        print(f"Giving resources to {player.name} for second settlement")
        for resource, count in player.resources.items():
            if count > 0:
                print(f"  - {resource.name}: {count}")
    
    return True


def place_initial_road(state, road_id, last_settlement_id):
    """
    Place an initial road during setup phase
    Returns True if successful, False otherwise
    """

    if not is_valid_initial_road(state, road_id, last_settlement_id):
        return False
    
    player = state.get_current_player()
    road = state.board.get_road(road_id)
    
    # Place road
    road.build_road(player.player_idx)
    player.add_road(road_id)
    
    # Advance to next player or phase
    advance_setup_phase(state)
    
    return True


def advance_setup_phase(state):
    """Advance to the next player or phase in setup"""
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
            state.possible_actions = get_possible_actions(state)
        else: 
            state.current_player_idx -= 1

def is_valid_initial_settlement(state, spot_id):
    """Check if a spot is valid for initial settlement placement"""
    spot = state.board.get_spot(spot_id)
    
    # Make sure the spot exists and is free
    if spot is None or spot.player_idx is not None:
        return False

    # Check distance rule
    return is_two_spots_away_from_settlement(state, spot_id)