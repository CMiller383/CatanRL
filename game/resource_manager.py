import random
from game.enums import Resource, SettlementType

def give_initial_resources(state, spot_id, player):
    """Give resources for hexes adjacent to the second settlement"""
    spot = state.board.get_spot(spot_id)
    for hex_id in spot.adjacent_hex_ids:
        hex_obj = state.board.get_hex(hex_id)
        # Don't give resources for desert
        if hex_obj.resource != Resource.DESERT:
            player.add_resource(hex_obj.resource, 1)

def distribute_resources(state, dice_result):
    """Distribute resources based on dice roll"""

    for hex_id, hex_obj in state.board.hexes.items():

        # Skip hexes where the robber is located
        if hex_obj.number == dice_result and hex_id != state.robber_hex_id:
            for spot_id in state.board.spots:
                spot = state.board.get_spot(spot_id)
                if spot.player_idx is not None and hex_id in spot.adjacent_hex_ids:
                    amount = 1
                    if spot.settlement_type == SettlementType.CITY:
                        amount = 2

                    player = state.players[spot.player_idx]
                    player.add_resource(hex_obj.resource, amount)

def handle_robber_roll(state):
    """Handle the effects of rolling a 7 (robber activation)"""

    # First, all players with more than 7 cards discard half, rounded down
    for player in state.players:
        total_cards = sum(player.resources.values())
        if total_cards > 7:
            discard_count = total_cards // 2
            
            # =================== NEED TO LET THEM CHOOSE TO DISCARD RESOURCES HERE ======================
            auto_discard_resources(player, discard_count)

    
    # Set flag to await robber placement
    state.awaiting_robber_placement = True

def auto_discard_resources(player, discard_count):
    """Automatically discard resources for AI players when a 7 is rolled"""
    resources_list = []
    for resource, count in player.resources.items():
        resources_list.extend([resource] * count)
    
    random.shuffle(resources_list)
    for i in range(discard_count):
        if resources_list:
            resource = resources_list.pop()
            player.resources[resource] -= 1