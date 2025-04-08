from game.enums import DevCardType, GamePhase, Resource, SettlementType
from game.rules import has_adjacent_road, is_road_connected, is_two_spots_away_from_settlement


def get_possible_actions(state):
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
            if road.owner is None and is_road_connected(state, road_id):
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
                if has_adjacent_road(state, spot_id, curr_player.player_idx) and is_two_spots_away_from_settlement(state, spot_id):
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
            if road.owner is None and is_road_connected(state, road_id):
                actions.add(("road", road_id))
    
    # Make sure end_turn is always available during regular play after rolling dice
    if state.rolled_dice:
        actions.add("end_turn")
    
    return actions