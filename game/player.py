from collections import defaultdict
from .resource import Resource
from .development_card import DevCardType

class Player:
    def __init__(self, player_id: int, name: str = ""):
        self.player_id = player_id
        self.name = name or f"Player {player_id}"
        self.settlements = []  # List of spot IDs where settlements are built
        self.roads = []        # List of road ids owned by the player
        self.resources = defaultdict(int)  # Maps Resource to count
        self.cities = []       # List of spot IDs where cities are built
        self.knights_played = 0

        #maxes
        self.MAX_SETTLEMENTS = 5
        self.MAX_CITIES = 4
        self.MAX_ROADS = 15

        #dev cards
        self.dev_cards = []
        self.played_dev_cards = []
        self.just_purchased_dev_card = False

    def add_settlement(self, spot_id: int):
        if len(self.settlements) < 15:
            self.settlements.append(spot_id)
            return True
        return False

    def add_road(self, road_id):
        if len(self.roads) < 15:
            self.roads.append(road_id)
            return True
        return False
    
    def add_city(self, spot_id: int):
        if len(self.cities) < self.MAX_CITIES:
            if spot_id in self.settlements:
                self.settlements.remove(spot_id)
            self.cities.append(spot_id)
            return True
        return False
    
    def add_resource(self, resource: Resource, amount: int = 1):
        self.resources[resource] += amount

    def has_city_resources(self):
        if self.resources[Resource.WHEAT] >= 2 and self.resources[Resource.ORE] >= 3 and len(self.cities) < self.MAX_CITIES:
            return True
        else:
            return False
        
    def buy_city(self):
        if self.has_city_resources():
            self.resources[Resource.ORE] -= 3
            self.resources[Resource.WHEAT] -= 2

    def has_settlement_resources(self):
        if (self.resources[Resource.WHEAT] >= 1 and self.resources[Resource.SHEEP] >= 1 and
                self.resources[Resource.BRICK] >= 1 and self.resources[Resource.WOOD] >= 1) and len(self.settlements) < self.MAX_SETTLEMENTS:
            return True
        else:
            return False
        
    def buy_settlement(self):
        if self.has_settlement_resources() and len(self.settlements) < self.MAX_SETTLEMENTS:
            self.resources[Resource.BRICK] -= 1
            self.resources[Resource.WOOD] -= 1
            self.resources[Resource.WHEAT] -= 1
            self.resources[Resource.SHEEP] -= 1
            return True
        return False

    def has_road_resources(self):
        if self.resources[Resource.BRICK] >= 1 and self.resources[Resource.WOOD] >= 1 and len(self.roads) < self.MAX_ROADS:
            return True
        else:
            return False
        
    def buy_road(self):
        if self.has_road_resources() and len(self.roads) < self.MAX_ROADS:
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
            self.just_purchased_dev_card = True
            return True
        return False
    
    def play_dev_card(self, card_idx):
        if 0 <= card_idx < len(self.dev_cards):
            if self.just_purchased_dev_card and card_idx == len(self.dev_cards) - 1:
                return False
                
            card = self.dev_cards.pop(card_idx)
            self.played_dev_cards.append(card)
            
            if card.card_type == DevCardType.KNIGHT:
                self.knights_played += 1
                
            return card
        return None
    
    def reset_dev_card_purchase_flag(self):
        self.just_purchased_dev_card = False
    
    def get_victory_points(self):
        """Calculate total victory points"""
        # 1 point per settlement, 2 per city
        settlement_points = len(self.settlements)
        city_points = len(self.cities) * 2
        
        # Count victory point cards
        vp_card_points = sum(1 for card in self.dev_cards + self.played_dev_cards 
                          if card.card_type == DevCardType.VICTORY_POINT)
        
        # Add other victory points (longest road, largest army) handled by the game logic
        return settlement_points + city_points + vp_card_points

    def __repr__(self):
        return f"Player(id={self.player_id}, name={self.name})"