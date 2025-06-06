from collections import defaultdict
from .enums import Resource
from .development_card import DevCardType

class Player:
    def __init__(self, player_idx: int, name: str = ""):
        self.player_idx = player_idx
        self.name = name or f"Player {player_idx}"
        self.settlements = []  # List of spot IDs where settlements are built
        self.roads = []        # List of road ids owned by the player
        self.resources = defaultdict(int)  # Maps Resource to count
        self.cities = []       # List of spot IDs where cities are built
        self.knights_played = 0
        self.victory_points = 0

        #dev cards
        self.dev_cards = []
        self.played_dev_cards = []

    def add_settlement(self, spot_id: int):
        if len(self.settlements) < 5:
            self.settlements.append(spot_id)
            self.victory_points += 1
            return True
        return False

    def add_road(self, road_id):
        if len(self.roads) < 15:
            self.roads.append(road_id)
            return True
        return False
    
    def add_city(self, spot_id: int):
        if len(self.cities) < 4:
            if spot_id in self.settlements:
                self.settlements.remove(spot_id)
            self.cities.append(spot_id)
            self.victory_points += 1
            return True
        return False
    
    def add_resource(self, resource: Resource, amount: int = 1):
        self.resources[resource] += amount

    def has_city_resources(self):
        if self.resources[Resource.WHEAT] >= 2 and self.resources[Resource.ORE] >= 3 and len(self.cities) < 4:
            return True
        else:
            return False
        
    def buy_city(self):
        if self.has_city_resources():
            self.resources[Resource.ORE] -= 3
            self.resources[Resource.WHEAT] -= 2

    def has_settlement_resources(self):
        if (self.resources[Resource.WHEAT] >= 1 and self.resources[Resource.SHEEP] >= 1 and
                self.resources[Resource.BRICK] >= 1 and self.resources[Resource.WOOD] >= 1) and len(self.settlements) < 5:
            return True
        else:
            return False
        
    def buy_settlement(self):
        if self.has_settlement_resources() and len(self.settlements) < 5:
            self.resources[Resource.BRICK] -= 1
            self.resources[Resource.WOOD] -= 1
            self.resources[Resource.WHEAT] -= 1
            self.resources[Resource.SHEEP] -= 1
            return True
        return False

    def has_road_resources(self):
        if self.resources[Resource.BRICK] >= 1 and self.resources[Resource.WOOD] >= 1 and len(self.roads) < 15:
            return True
        else:
            return False
        
    def buy_road(self):
        if self.has_road_resources() and len(self.roads) < 15:
            self.resources[Resource.BRICK] -= 1
            self.resources[Resource.WOOD] -= 1
            return True
        return False
    
    def has_dev_card_resources(self):
        return (self.resources[Resource.ORE] >= 1 and 
                self.resources[Resource.WHEAT] >= 1 and 
                self.resources[Resource.SHEEP] >= 1)
    
    def buy_dev_card(self, card):
        if self.has_dev_card_resources():
            self.resources[Resource.ORE] -= 1
            self.resources[Resource.WHEAT] -= 1
            self.resources[Resource.SHEEP] -= 1
            self.dev_cards.append(card)

            if card.card_type == DevCardType.VICTORY_POINT:
                self.victory_points += 1

            return True
        return False
    
    def play_dev_card(self, card_idx):
        if 0 <= card_idx < len(self.dev_cards):
            card = self.dev_cards.pop(card_idx)
            self.played_dev_cards.append(card)
            
            if card.card_type == DevCardType.KNIGHT:
                self.knights_played += 1
                
            return card
        return None

    def __repr__(self):
        return f"Player(id={self.player_id}, name={self.name})"