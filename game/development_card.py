import random

from game.enums import DevCardType


class DevelopmentCard:
    def __init__(self, card_type: DevCardType, name: str):
        self.card_type = card_type
        self.name = name
        self.played = False
        
    def __repr__(self):
        return f"DevelopmentCard({self.name}, {self.card_type})"

class DevelopmentCardDeck:
    def __init__(self):
        # Initialize the development card deck according to standard Catan rules
        self.cards = []
        
        # 14 Knight cards
        for i in range(14):
            self.cards.append(DevelopmentCard(DevCardType.KNIGHT, "Knight"))
        
        # 5 Victory Point cards
        self.cards.append(DevelopmentCard(DevCardType.VICTORY_POINT, "Great Hall"))
        self.cards.append(DevelopmentCard(DevCardType.VICTORY_POINT, "Market"))
        self.cards.append(DevelopmentCard(DevCardType.VICTORY_POINT, "Chapel"))
        self.cards.append(DevelopmentCard(DevCardType.VICTORY_POINT, "University"))
        self.cards.append(DevelopmentCard(DevCardType.VICTORY_POINT, "Library"))
        
        # 2 Road Building cards
        self.cards.append(DevelopmentCard(DevCardType.ROAD_BUILDING, "Road Building"))
        self.cards.append(DevelopmentCard(DevCardType.ROAD_BUILDING, "Road Building"))
        
        # 2 Year of Plenty cards
        self.cards.append(DevelopmentCard(DevCardType.YEAR_OF_PLENTY, "Year of Plenty"))
        self.cards.append(DevelopmentCard(DevCardType.YEAR_OF_PLENTY, "Year of Plenty"))
        
        # 2 Monopoly cards
        self.cards.append(DevelopmentCard(DevCardType.MONOPOLY, "Monopoly"))
        self.cards.append(DevelopmentCard(DevCardType.MONOPOLY, "Monopoly"))
        
        # Shuffle the deck
        random.shuffle(self.cards)
    
    def draw_card(self):
        """Draw a card from the deck if available"""
        if not self.cards:
            return None
        return self.cards.pop()
    
    def is_empty(self):
        """Check if the deck is empty"""
        return len(self.cards) == 0
    
    def __len__(self):
        return len(self.cards)