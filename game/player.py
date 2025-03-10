from collections import defaultdict
from .resource import Resource

class Player:
    def __init__(self, player_id: int, name: str = ""):
        self.player_id = player_id
        self.name = name or f"Player {player_id}"
        self.settlements = []  # List of spot IDs where settlements are built
        self.roads = []        # List of road ids owned by the player
        self.resources = defaultdict(int)  # Maps Resource to count

    def add_settlement(self, spot_id: int):
        self.settlements.append(spot_id)

    def add_road(self, road):
        self.roads.append(road)

    def add_resource(self, resource: Resource, amount: int = 1):
        self.resources[resource] += amount

    def has_city_resources(self):
        if self.resources[Resource.WHEAT] >= 2 and self.resources[Resource.ORE] >= 3:
            return True
        else:
            return False
        
    def buy_city(self):
        if self.has_city_resources():
            self.resources[Resource.ORE] -= 3
            self.resources[Resource.WHEAT] -= 2

    def has_settlement_resources(self):
        if (self.resources[Resource.WHEAT] >= 1 and self.resources[Resource.SHEEP] >= 1 and
                self.resources[Resource.BRICK] >= 1 and self.resources[Resource.WOOD] >= 1):
            return True
        else:
            return False
        
    def buy_settlement(self):
        if self.has_settlement_resources():
            self.resources[Resource.BRICK] -= 1
            self.resources[Resource.WOOD] -= 1
            self.resources[Resource.WHEAT] -= 1
            self.resources[Resource.SHEEP] -= 1

    def has_road_resources(self):
        if self.resources[Resource.BRICK] >= 1 and self.resources[Resource.WOOD] >= 1:
            return True
        else:
            return False
        
    def buy_road(self):
        if self.has_road_resources():
            self.resources[Resource.BRICK] -= 1
            self.resources[Resource.WOOD] -= 1

    def __repr__(self):
        return f"Player(id={self.player_id}, name={self.name})"
