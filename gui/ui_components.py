"""
UI components for the Catan game GUI.
"""
import pygame
from game.action import Action
from gui.constants import *
from game.development_card import DevCardType
from game.enums import ActionType, GamePhase, Resource

class DevCardHandler:
    def __init__(self, game_logic, screen, window_width, window_height, fonts, dev_card_icons=None):
        self.game_logic = game_logic
        self.screen = screen
        self.window_width = window_width
        self.window_height = window_height
        self.font = fonts['font']
        self.info_font = fonts['info_font']
        self.card_font = fonts['card_font']
        self.card_desc_font = fonts['card_desc_font']
        
        self.dev_card_buttons = []
        self.dev_card_action_buttons = []
        self.selected_dev_card = None
        self.dev_card_icons = dev_card_icons

    def draw_development_cards(self):
        """Draw development cards panel and controls for the current player"""
        if self.game_logic.state.current_phase != GamePhase.REGULAR_PLAY or not self.game_logic.is_current_player_human():
            return  # Only show dev cards in regular play phase for human players
            
        curr_player = self.game_logic.state.get_current_player()
        
        # Clear previous buttons
        self.dev_card_buttons = []
        self.dev_card_action_buttons = []
        
        # Dev card panel area (bottom center of screen)
        panel_width = int(self.window_width * 0.5)
        panel_height = int(self.window_height * 0.2)
        panel_x = (self.window_width * 0.8 - panel_width) // 2  # Center it in the game area (excluding player panel)
        panel_y = self.window_height - panel_height - 10  # 10px margin from bottom
        
        # Only draw if player has dev cards or can buy them
        has_dev_cards = len(curr_player.dev_cards) > 0
        can_buy_dev_card = Action(ActionType.BUY_DEV_CARD) in self.game_logic.state.possible_actions

        if not (has_dev_cards or can_buy_dev_card):
            return
            
        # Draw semi-transparent panel
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((220, 220, 220, 180))  # Light gray with transparency
        self.screen.blit(panel_surface, (panel_x, panel_y))
        
        # Title for the panel
        title = self.info_font.render("Development Cards", True, TEXT_COLOR)
        title_rect = title.get_rect(center=(panel_x + panel_width // 2, panel_y + 15))
        self.screen.blit(title, title_rect)
        
        # Area for cards
        card_area_y = panel_y + 35
        card_area_height = panel_height - 60
        
        # Buy dev card button
        
        if can_buy_dev_card:
            buy_button_width = 120
            buy_button_height = 30
            buy_button_x = panel_x + panel_width - buy_button_width - 10
            buy_button_y = panel_y + 10
            
            buy_button = pygame.Rect(buy_button_x, buy_button_y, buy_button_width, buy_button_height)
            pygame.draw.rect(self.screen, (100, 200, 100), buy_button)  # Green
            pygame.draw.rect(self.screen, TEXT_COLOR, buy_button, 1)  # Border
            
            buy_text = self.info_font.render("Buy Card", True, TEXT_COLOR)
            buy_text_rect = buy_text.get_rect(center=buy_button.center)
            self.screen.blit(buy_text, buy_text_rect)
            
            # Store button for click detection
            self.dev_card_action_buttons.append(("buy_dev_card", buy_button))
        
        # Display player's dev cards
        if has_dev_cards:
            cards_per_row = min(4, len(curr_player.dev_cards))
            card_width = min(120, (panel_width - 20) // cards_per_row)
            card_height = min(150, card_area_height - 10)
            card_spacing = min(10, (panel_width - card_width * cards_per_row) // (cards_per_row + 1))
            
            for i, card in enumerate(curr_player.dev_cards):
                row = i // cards_per_row
                col = i % cards_per_row
                
                # Calculate card position
                card_x = panel_x + card_spacing + col * (card_width + card_spacing)
                card_y = card_area_y + row * (card_height + 10)
                
                card_color = DEV_CARD_COLOR if card.card_type != DevCardType.VICTORY_POINT else (255, 223, 0)  # Gold for VP cards
                
                # Draw card background
                card_rect = pygame.Rect(card_x, card_y, card_width, card_height)
                pygame.draw.rect(self.screen, card_color, card_rect)
                pygame.draw.rect(self.screen, TEXT_COLOR, card_rect, 1)  # Border
                
                # Card title
                card_title = self.card_font.render(card.name, True, TEXT_COLOR)
                card_title_rect = card_title.get_rect(center=(card_x + card_width // 2, card_y + 15))
                self.screen.blit(card_title, card_title_rect)
                
                # Card icon or type
                if self.dev_card_icons and card.card_type in self.dev_card_icons:
                    icon = self.dev_card_icons[card.card_type]
                    icon_rect = icon.get_rect(center=(card_x + card_width // 2, card_y + card_height // 2))
                    self.screen.blit(icon, icon_rect)
                else:
                    # Text description if no icon
                    card_type_text = self.card_desc_font.render(card.card_type.value.replace("_", " ").title(), True, TEXT_COLOR)
                    card_type_rect = card_type_text.get_rect(center=(card_x + card_width // 2, card_y + card_height // 2))
                    self.screen.blit(card_type_text, card_type_rect)
                
                # Store card button for click detection if playable
                self.dev_card_buttons.append((i, card_rect))
            
            # If a card is selected, show action buttons
            if self.selected_dev_card is not None:
                action_buttons_y = card_area_y + card_area_height - 40
                
                # Play card button
                selected_card = curr_player.dev_cards[self.selected_dev_card]
                if selected_card.card_type != DevCardType.VICTORY_POINT:
                    play_button_width = 100
                    play_button_x = panel_x + panel_width // 2 - play_button_width // 2
                    
                    play_button = pygame.Rect(play_button_x, action_buttons_y, play_button_width, 30)
                    pygame.draw.rect(self.screen, (100, 200, 100), play_button)  # Green
                    pygame.draw.rect(self.screen, TEXT_COLOR, play_button, 1)  # Border
                    
                    play_text = self.info_font.render("Play Card", True, TEXT_COLOR)
                    play_text_rect = play_text.get_rect(center=play_button.center)
                    self.screen.blit(play_text, play_text_rect)
                    
                    # Store for click detection
                    self.dev_card_action_buttons.append((f"play_{selected_card.card_type.value}", play_button))
                    
    def check_dev_card_click(self, mouse_pos):
        """Check if a development card was clicked"""
        for card_idx, card_rect in self.dev_card_buttons:
            if card_rect.collidepoint(mouse_pos):
                return card_idx
        return None

    def check_dev_card_action_click(self, mouse_pos):
        """Check if a development card action button was clicked"""
        for action, button_rect in self.dev_card_action_buttons:
            if button_rect.collidepoint(mouse_pos):
                return action
        return None


class ResourceSelectionHandler:
    def __init__(self, game_logic, screen, window_width, window_height, fonts):
        self.game_logic = game_logic
        self.screen = screen
        self.window_width = window_width
        self.window_height = window_height
        self.font = fonts['font']
        self.info_font = fonts['info_font']
        
        self.resource_selection_buttons = []
    
    def draw_resource_selection(self):
        """Draw resource selection for Year of Plenty and Monopoly actions"""
        if not self.game_logic.state.awaiting_resource_selection and not self.game_logic.state.awaiting_monopoly_selection:
            return
            
        # Clear previous buttons
        self.resource_selection_buttons = []
        
        # Panel in center of screen
        panel_width = 300
        panel_height = 300
        panel_x = (self.window_width * 0.8 - panel_width) // 2
        panel_y = (self.window_height - panel_height) // 2
        
        # Draw panel
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((240, 240, 240, 230))  # Light gray with higher opacity
        self.screen.blit(panel_surface, (panel_x, panel_y))
        
        # Title
        if self.game_logic.state.awaiting_resource_selection:
            title = "Select Resource (Year of Plenty)"
            count_text = f"Select {self.game_logic.state.awaiting_resource_selection_count} resource(s)"
        else:
            title = "Select Resource (Monopoly)"
            count_text = "Choose one resource to take from all players"
            
        title_text = self.info_font.render(title, True, TEXT_COLOR)
        count_text_surface = self.font.render(count_text, True, TEXT_COLOR)
        
        title_rect = title_text.get_rect(center=(panel_x + panel_width // 2, panel_y + 20))
        count_rect = count_text_surface.get_rect(center=(panel_x + panel_width // 2, panel_y + 45))
        
        self.screen.blit(title_text, title_rect)
        self.screen.blit(count_text_surface, count_rect)
        
        # Draw resource buttons
        button_width = 100
        button_height = 30
        button_margin = 10
        button_y = panel_y + 80
        
        resources = [Resource.WOOD, Resource.BRICK, Resource.WHEAT, Resource.SHEEP, Resource.ORE]
        
        for i, resource in enumerate(resources):
            button_x = panel_x + (panel_width - button_width) // 2
            button_y_pos = button_y + i * (button_height + button_margin)
            
            button = pygame.Rect(button_x, button_y_pos, button_width, button_height)
            pygame.draw.rect(self.screen, RESOURCE_COLORS[resource], button)
            pygame.draw.rect(self.screen, TEXT_COLOR, button, 1)  # Border
            
            text = self.font.render(resource.name.title(), True, TEXT_COLOR)
            text_rect = text.get_rect(center=button.center)
            self.screen.blit(text, text_rect)
            
            # Store button for click detection
            action_type = "monopoly" if self.game_logic.state.awaiting_monopoly_selection else "year_of_plenty"
            self.resource_selection_buttons.append((action_type, resource, button))
    
    def check_resource_selection_click(self, mouse_pos):
        """Check if a resource selection button was clicked"""
        for action_type, resource, button_rect in self.resource_selection_buttons:
            if button_rect.collidepoint(mouse_pos):
                return (action_type, resource)
        return None


class StealSelectionHandler:
    def __init__(self, game_logic, screen, window_width, window_height, fonts, players):
        self.game_logic = game_logic
        self.screen = screen
        self.window_width = window_width
        self.window_height = window_height
        self.info_font = fonts['info_font']
        self.players = players
        
        self.steal_buttons = []
        self.steal_panel = None
        self.potential_victims = []
    
    def setup_steal_from_player(self, potential_victims):
        """
        Set up UI for selecting a player to steal from after moving the robber
        potential_victims: list of player IDs that can be stolen from
        """
        # Store victims list
        self.potential_victims = potential_victims
        self.steal_buttons = []
        
        # If no victims available, just return
        if not potential_victims:
            return
        
        # Create a panel in the center of the screen
        panel_width = 300
        panel_height = 40 + len(potential_victims) * 50
        panel_x = (self.window_width * 0.8 - panel_width) // 2
        panel_y = (self.window_height - panel_height) // 2
        
        self.steal_panel = {
            'rect': pygame.Rect(panel_x, panel_y, panel_width, panel_height),
            'buttons': []
        }
        
        # Add buttons for each potential victim
        title_height = 40
        button_height = 40
        button_margin = 10
        button_width = panel_width - 20
        
        # Store button data for each victim
        for i, player_idx in enumerate(potential_victims):
            button_y = panel_y + title_height + i * (button_height + button_margin)
            button_rect = pygame.Rect(panel_x + 10, button_y, button_width, button_height)
            self.steal_buttons.append((player_idx, button_rect))
    
    def draw_steal_selection(self):
        """Draw UI for selecting a player to steal from"""
        if not self.steal_buttons:
            return
        
        # Draw panel background
        panel_rect = self.steal_panel['rect']
        panel_surface = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
        panel_surface.fill((240, 240, 240, 230))  # Light gray with higher opacity
        self.screen.blit(panel_surface, panel_rect)
        
        # Draw title
        title_text = self.info_font.render("Select a player to steal from:", True, TEXT_COLOR)
        title_rect = title_text.get_rect(center=(panel_rect.centerx, panel_rect.y + 20))
        self.screen.blit(title_text, title_rect)
        
        # Draw player buttons
        for player_idx, button_rect in self.steal_buttons:
            player = self.players[player_idx]
            color = PLAYER_COLOR_RGBS[player_idx]
            
            pygame.draw.rect(self.screen, color, button_rect)
            pygame.draw.rect(self.screen, TEXT_COLOR, button_rect, 1)  # Border
            
            text = self.info_font.render(f"{player.name}", True, TEXT_COLOR)
            text_rect = text.get_rect(center=button_rect.center)
            self.screen.blit(text, text_rect)
    
    def check_steal_button_click(self, mouse_pos):
        """Check if a steal selection button was clicked"""
        for player_idx, button_rect in self.steal_buttons:
            if button_rect.collidepoint(mouse_pos):
                return player_idx
        return None
    
    def clear_steal_selection(self):
        """Clear the steal selection UI"""
        self.steal_buttons = []
        self.steal_panel = None
        self.potential_victims = []