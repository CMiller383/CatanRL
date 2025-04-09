from game.action import Action
from game.enums import ActionType, DevCardType, GamePhase, Resource, SettlementType
from game.rules import has_adjacent_road, is_road_connected, is_two_spots_away_from_settlement


def get_possible_actions(state):
    actions = set()

    if state.current_phase != GamePhase.REGULAR_PLAY:
        return actions

    # Special one-at-a-time interactions
    if state.awaiting_robber_placement:
        for hex_id in state.board.hexes:
            if hex_id != state.robber_hex_id:
                actions.add(Action(ActionType.MOVE_ROBBER, payload=hex_id))
        return actions

    if state.awaiting_resource_selection:
        for resource in Resource:
            if resource != Resource.DESERT:
                actions.add(Action(ActionType.SELECT_YEAR_OF_PLENTY_RESOURCE, payload=resource))
        return actions

    if state.awaiting_monopoly_selection:
        for resource in Resource:
            if resource != Resource.DESERT:
                actions.add(Action(ActionType.SELECT_MONOPOLY_RESOURCE, payload=resource))
        return actions

    if 0 < state.road_building_roads_placed < 2:
        for road_id, road in state.board.roads.items():
            if road.owner is None and is_road_connected(state, road_id):
                actions.add(Action(ActionType.PLACE_FREE_ROAD, payload=road_id))
        return actions

    curr_player = state.get_current_player()

    # Dice roll not yet done
    if not state.rolled_dice:
        actions.add(Action(ActionType.ROLL_DICE))

        # Dev cards that can be played before rolling
        if not state.dev_card_played_this_turn:
            knight_indices = [
                i for i, card in enumerate(curr_player.dev_cards)
                if card.card_type == DevCardType.KNIGHT and
                   (not curr_player.just_purchased_dev_card or i < len(curr_player.dev_cards) - 1)
            ]
            if knight_indices:
                actions.add(Action(ActionType.PLAY_KNIGHT_CARD))
        return actions

    # Dice rolled
    actions.add(Action(ActionType.END_TURN))

    if not state.dev_card_deck.is_empty() and curr_player.has_dev_card_resources():
        actions.add(Action(ActionType.BUY_DEV_CARD))

    if not state.dev_card_played_this_turn:
        check_cards = curr_player.dev_cards[:-1] if curr_player.just_purchased_dev_card else curr_player.dev_cards

        if any(card.card_type == DevCardType.KNIGHT for card in check_cards):
            actions.add(Action(ActionType.PLAY_KNIGHT_CARD))

        if any(card.card_type == DevCardType.ROAD_BUILDING for card in check_cards):
            if len(curr_player.roads) < curr_player.MAX_ROADS - 1:
                actions.add(Action(ActionType.PLAY_ROAD_BUILDING_CARD))

        if any(card.card_type == DevCardType.YEAR_OF_PLENTY for card in check_cards):
            actions.add(Action(ActionType.PLAY_YEAR_OF_PLENTY_CARD))

        if any(card.card_type == DevCardType.MONOPOLY for card in check_cards):
            actions.add(Action(ActionType.PLAY_MONOPOLY_CARD))

    # Build settlement
    if curr_player.has_settlement_resources() and len(curr_player.settlements) < curr_player.MAX_SETTLEMENTS:
        for spot_id, spot in state.board.spots.items():
            if spot.player_idx is None:
                if has_adjacent_road(state, spot_id, curr_player.player_idx) and is_two_spots_away_from_settlement(state, spot_id):
                    actions.add(Action(ActionType.BUILD_SETTLEMENT, payload=spot_id))

    # Upgrade to city
    if curr_player.has_city_resources() and hasattr(curr_player, 'cities') and len(curr_player.cities) < curr_player.MAX_CITIES:
        for spot_id in curr_player.settlements:
            spot = state.board.get_spot(spot_id)
            if spot and spot.settlement_type == SettlementType.SETTLEMENT:
                actions.add(Action(ActionType.UPGRADE_TO_CITY, payload=spot_id))

    # Build road
    if curr_player.has_road_resources() and len(curr_player.roads) < curr_player.MAX_ROADS:
        for road_id, road in state.board.roads.items():
            if road.owner is None and is_road_connected(state, road_id):
                actions.add(Action(ActionType.BUILD_ROAD, payload=road_id))

    return actions
