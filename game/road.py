class Road:
    def __init__(self, spot1_id: int, spot2_id: int, adjacent_hex_ids: list):
        self.spot1_id = spot1_id
        self.spot2_id = spot2_id
        self.adjacent_hex_ids = adjacent_hex_ids
        self.owner = None  # The player who owns the road, if any

    def build_road(self, player):
        self.owner = player

    def __repr__(self):
        return (f"Road(spot1={self.spot1_id}, spot2={self.spot2_id}, "
                f"hexes={self.adjacent_hex_ids}, owner={self.owner})")
