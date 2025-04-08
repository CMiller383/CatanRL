from .enums import SettlementType

class Spot:
    def __init__(self, spot_id: int, adjacent_hex_ids: list, position: tuple = None, has_port: bool = False):
        self.spot_id = spot_id
        self.adjacent_hex_ids = adjacent_hex_ids  # List of hex IDs adjacent to this vertex
        self.position = position                  # (x, y) coordinate of the spot
        self.player_idx = None                        # Which player (if any) occupies this spot
        self.settlement_type = SettlementType.NONE
        self.has_port = has_port

    def build_settlement(self, player_idx, settlement_type: SettlementType = SettlementType.SETTLEMENT):
        self.player_idx = player_idx
        self.settlement_type = settlement_type

    def __repr__(self):
        return (f"Spot(id={self.spot_id}, hexes={self.adjacent_hex_ids}, position={self.position}, "
                f"player={self.player_idx}, type={self.settlement_type.name}, port={self.has_port})")
