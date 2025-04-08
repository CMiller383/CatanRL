from game.enums import GamePhase


def is_valid_initial_road(state, road_id, last_settlement_id):
    """
    Check if a road placement is valid in setup phase
    The road must be connected to the last settlement placed
    """
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

def has_adjacent_road(state, spot_id, player_idx):
    """Check if the spot is adjacent to a road owned by the player"""

    for road_id, road in state.board.roads.items():
        if road.owner == player_idx:
            if spot_id in (road.spot1_id, road.spot2_id):
                return True
    return False

def is_two_spots_away_from_settlement(state, spot_id):
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

def has_adjascent_road(state, spot_id):
    curr_player = state.get_current_player()

    has_adjacent_road = False
    for r_id in curr_player.roads:
        road = state.board.get_road(r_id)
        if road and spot_id in (road.spot1_id, road.spot2_id):
            has_adjacent_road = True
            break
    
    return has_adjacent_road

def is_road_connected(state, road_id):
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

