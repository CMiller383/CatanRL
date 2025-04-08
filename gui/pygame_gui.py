import os
import pygame
import math
from game.board import Board

from game.spot import SettlementType
from game.game_logic import GameLogic
from game.enums import GamePhase

# Global screen proportion (how much of the screen to use)
SCREEN_PROPORTION = 0.75

from agent.base import AgentType

class CatanGame:
    def __init__(self, window_width=None, window_height=None, agent_types=None):
        pygame.init()
        
        # Auto-detect screen size if not provided
        if window_width is None or window_height is None:
            screen_info = pygame.display.Info()
            max_width, max_height = screen_info.current_w, screen_info.current_h
            
            # Apply the screen proportion
            window_width = int(max_width * SCREEN_PROPORTION)
            window_height = int(max_height * SCREEN_PROPORTION)
            
            # Ensure it's a square or landscape
            window_height = min(window_width, window_height)
        
        self.window_width = window_width
        self.window_height = window_height
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Settlers of Catan Board")
        self.clock = pygame.time.Clock()
        
        # Font sizes scaled based on resolution
        base_font_size = int(min(window_width, window_height) / 50)
        self.font = pygame.font.SysFont('Arial', base_font_size)
        self.number_font = pygame.font.SysFont('Arial', int(base_font_size * 1.5), bold=True)
        self.info_font = pygame.font.SysFont('Arial', int(base_font_size * 1.2))
        self.instruction_font = pygame.font.SysFont('Arial', int(base_font_size * 1.8), bold=True)
        self.card_font = pygame.font.SysFont('Arial', int(base_font_size * 1.1), bold=True)
        self.card_desc_font = pygame.font.SysFont('Arial', int(base_font_size * 0.9))
        
        self.board = Board()
        
        # Initialize game logic with player setup
        self.game_logic = GameLogic(self.board, agent_types)
        self.players = self.game_logic.state.players  # Reference to players for easier access
        self.ai_thinking_timer = 0  # Timer to create a slight delay between AI moves
        
        # Calculate scale and offset to fit the board in the window
        self.scale, self.offset = self.compute_transform()
        
        # Calculate positions for spots and roads
        self.spot_positions = {}
        self.calculate_spot_positions()
        
        self.road_positions = {}
        self.calculate_road_positions()
        
        # Hex centers and vertices for drawing
        self.hex_centers = {}
        self.hex_vertices = {}
        self.calculate_hex_positions()
        
        # Hit detection - scale these based on resolution
        self.spot_radius = int(min(window_width, window_height) / 80)
        self.road_width = int(min(window_width, window_height) / 100)
        self.number_circle_radius = int(min(window_width, window_height) / 40)
        self.settlement_size = int(self.spot_radius * 1.5)
        
        # Game state
        self.last_settlement_placed = None  # To track the most recent settlement for road placement
        
        # Calculate scale and offset to fit the board in the window
        self.scale, self.offset = self.compute_transform()
        
        # Calculate positions for spots and roads
        self.spot_positions = {}
        self.calculate_spot_positions()
        
        self.road_positions = {}
        self.calculate_road_positions()
        
        # Hex centers and vertices for drawing
        self.hex_centers = {}
        self.hex_vertices = {}
        self.calculate_hex_positions()
        
        # Hit detection - scale these based on resolution
        self.spot_radius = int(min(window_width, window_height) / 80)
        self.road_width = int(min(window_width, window_height) / 100)
        self.number_circle_radius = int(min(window_width, window_height) / 40)
        self.settlement_size = int(self.spot_radius * 1.5)
        
        # Game state
        self.last_settlement_placed = None  # To track the most recent settlement for road placement
                # Dev card related
        self.dev_card_buttons = []  # List to store dev card button rectangles
        self.dev_card_action_buttons = []  # List to store dev card action buttons
        self.selected_dev_card = None  # Currently selected development card
        self.resource_selection_buttons = []  # For Year of Plenty and Monopoly
        self.robber_placement_active = False  # Flag for robber placement mode
        
        self.load_images()

    def load_images(self):
        """Load and scale images for settlements and cities from their respective folders."""
        self.settlement_images = {}
        self.city_images = {}

        # Define the target size based on the spot radius; adjust multiplier as needed.
        img_size = self.spot_radius * 10

        # Load settlement images
        settlement_folder = 'assets/settlements'
        for filename in os.listdir(settlement_folder):
            if filename.endswith('.png'):
                # For example, "red_settlement.png" becomes key "red"
                key = filename.replace('_settlement.png', '')
                image_path = os.path.join(settlement_folder, filename)
                img = pygame.image.load(image_path).convert_alpha()
                img = pygame.transform.smoothscale(img, (img_size, img_size))
                self.settlement_images[key] = img

        # Load city images
        city_folder = 'assets/cities'
        for filename in os.listdir(city_folder):
            if filename.endswith('.png'):
                key = filename.replace('_city.png', '')
                image_path = os.path.join(city_folder, filename)
                img = pygame.image.load(image_path).convert_alpha()
                img = pygame.transform.smoothscale(img, (img_size, img_size))
                self.city_images[key] = img

        # sand border image
        original = pygame.image.load('assets/border.png').convert_alpha()
        border_size = int(self.window_height * 1.07)
        self.border_image = pygame.transform.smoothscale(original, (border_size, border_size))

        # dice image
        self.dice_images = {}
        dice_folder = 'assets/dice'
        for i in range(1, 7):
            filename = f"dice{i}.png"
            path = os.path.join(dice_folder, filename)
            img = pygame.image.load(path).convert_alpha()
            # Scale the dice image; here we use 10% of the window width as the dice size.
            dice_size = int(self.window_width * 0.1)
            img = pygame.transform.smoothscale(img, (dice_size, dice_size))
            self.dice_images[i] = img

        # Also load the diceroll prompt image, if you want to show it separately
        diceroll_path = os.path.join(dice_folder, "diceroll.png")
        self.diceroll_image = pygame.image.load(diceroll_path).convert_alpha()
        # Scale it as needed, for example 15% of the window width:
        diceroll_size = int(self.window_width * 0.15)
        self.diceroll_image = pygame.transform.smoothscale(self.diceroll_image, (diceroll_size, diceroll_size))
        # Load development card icons if available (they arent!)
        try:
            self.dev_card_icons = {}
            dev_card_folder = 'assets/dev_cards'
            for card_type in DevCardType:
                filename = f"{card_type.value}.png"
                path = os.path.join(dev_card_folder, filename)
                if os.path.exists(path):
                    img = pygame.image.load(path).convert_alpha()
                    card_size = int(self.window_width * 0.05)  # 5% of window width
                    img = pygame.transform.smoothscale(img, (card_size, card_size))
                    self.dev_card_icons[card_type] = img
        except (FileNotFoundError, pygame.error):
            # If files don't exist, we'll use text-based representation instead
            self.dev_card_icons = None
    
    def compute_transform(self, margin=0.1):
        """Compute scale and offset to fit the board in the window"""
        # Account for player panel width (20% of screen)
        effective_width = self.window_width * 0.8
        
        # Get min and max coordinates of the board
        positions = [spot.position for spot in self.board.spots.values()]
        min_x = min(x for x, y in positions)
        max_x = max(x for x, y in positions)
        min_y = min(y for x, y in positions)
        max_y = max(y for x, y in positions)
        
        board_width = max_x - min_x
        board_height = max_y - min_y
        
        # Calculate scale to fit in window with margin
        scale_x = (effective_width * (1 - margin)) / board_width
        scale_y = (self.window_height * (1 - margin)) / board_height
        scale = min(scale_x, scale_y)
        
        # Calculate offset to center the board
        offset_x = (effective_width - board_width * scale) / 2 - min_x * scale
        offset_y = (self.window_height - board_height * scale) / 2 - min_y * scale
        
        return scale, (offset_x, offset_y)
    
    def board_to_screen(self, point):
        """Convert board coordinates to screen coordinates"""
        x, y = point
        sx = int(x * self.scale + self.offset[0])
        sy = int(y * self.scale + self.offset[1])
        return (sx, sy)
    
    def calculate_spot_positions(self):
        """Calculate screen positions for all spots"""
        for spot_id, spot in self.board.spots.items():
            self.spot_positions[spot_id] = self.board_to_screen(spot.position)
    
    def calculate_road_positions(self):
        """Calculate positions and endpoints for all roads"""
        for road_id, road in self.board.roads.items():
            spot1_pos = self.spot_positions[road.spot1_id]
            spot2_pos = self.spot_positions[road.spot2_id]
            # Store both the midpoint and the endpoints
            midpoint = ((spot1_pos[0] + spot2_pos[0]) // 2, (spot1_pos[1] + spot2_pos[1]) // 2)
            self.road_positions[road_id] = {
                'midpoint': midpoint,
                'endpoints': (spot1_pos, spot2_pos)
            }
    
    def calculate_hex_positions(self):
        """Calculate the center and vertices for each hex for drawing"""
        hex_size = 1.0  # Same size used in Board._init_hexes
        
        for hex_id, hex_obj in self.board.hexes.items():
            # Center of the hex
            center = self.board_to_screen(hex_obj.center)
            self.hex_centers[hex_id] = center
            
            # Calculate the 6 vertices of the hex
            vertices = []
            for angle_deg in range(0, 360, 60):
                angle_rad = math.radians(angle_deg)
                vx = center[0] + hex_size * self.scale * math.cos(angle_rad)
                vy = center[1] + hex_size * self.scale * math.sin(angle_rad)
                vertices.append((int(vx), int(vy)))
            
            self.hex_vertices[hex_id] = vertices
    
    def draw_board_border(self):
        """Draws the border image (assets/border.png) behind the board.
        The image is scaled to a fraction of the window height and centered on hex 10.
        """

        # Use hex 10's screen position as the center of the board.
        center = self.hex_centers.get(10)
        if center is None:
            return

        # Get the rect of the scaled image and center it on the board.
        border_rect = self.border_image.get_rect(center=center)
        self.screen.blit(self.border_image, border_rect)

    def draw_dice(self):
        """Draw dice 1 and dice 2 on the bottom right of the screen."""
        margin_x = self.window_width * .2
        margin_y = 0

        dice_size = self.dice_images[1].get_width()

        # Compute positions for dice 1 and dice 2 based on the new offsets:
        dice_x = self.window_width - margin_x - 2 * dice_size
        dice_y = self.window_height - margin_y - dice_size

        pos1 = (dice_x, dice_y)
        pos2 = (dice_x + dice_size, dice_y)

        
        if self.game_logic.state.dice1_roll is not None:
            dice_1_image = self.dice_images[self.game_logic.state.dice1_roll]
            dice_2_image = self.dice_images[self.game_logic.state.dice2_roll]
        else:
            dice_1_image = self.dice_images[1]
            dice_2_image = self.dice_images[1]
        self.screen.blit(dice_1_image, pos1)
        self.screen.blit(dice_2_image, pos2)
            

        # Save the dice area rectangle for click detection
        self.dice_rect = pygame.Rect(dice_x, dice_y, 2 * dice_size, dice_size)

    def draw_development_cards(self):
        """Draw development cards panel and controls for the current player"""
        if not self.game_logic.is_setup_complete() or not self.game_logic.is_current_player_human():
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
        can_buy_dev_card = "buy_dev_card" in self.game_logic.state.possible_actions
        
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
                
                # Check if card is playable
                can_play = False
                just_purchased = curr_player.just_purchased_dev_card and i == len(curr_player.dev_cards) - 1
                
                if not just_purchased and not self.game_logic.state.dev_card_played_this_turn:
                    if card.card_type == DevCardType.KNIGHT and "play_knight" in self.game_logic.state.possible_actions:
                        can_play = True
                    elif card.card_type == DevCardType.ROAD_BUILDING and "play_road_building" in self.game_logic.state.possible_actions:
                        can_play = True
                    elif card.card_type == DevCardType.YEAR_OF_PLENTY and "play_year_of_plenty" in self.game_logic.state.possible_actions:
                        can_play = True
                    elif card.card_type == DevCardType.MONOPOLY and "play_monopoly" in self.game_logic.state.possible_actions:
                        can_play = True
                
                # Determine card color based on state
                if self.selected_dev_card == i:
                    card_color = DEV_CARD_SELECTED_COLOR
                elif just_purchased:
                    card_color = DEV_CARD_DISABLED_COLOR
                elif can_play:
                    card_color = DEV_CARD_ENABLED_COLOR
                else:
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
                if can_play:
                    self.dev_card_buttons.append((i, card_rect))
                elif card.card_type == DevCardType.VICTORY_POINT:
                    # Display Victory Point value
                    vp_text = self.card_desc_font.render("+1 VP", True, TEXT_COLOR)
                    vp_rect = vp_text.get_rect(center=(card_x + card_width // 2, card_y + card_height - 15))
                    self.screen.blit(vp_text, vp_rect)
                
                # Show "New" label for just purchased cards
                if just_purchased:
                    new_text = self.card_desc_font.render("NEW", True, (255, 0, 0))
                    new_rect = new_text.get_rect(center=(card_x + card_width // 2, card_y + card_height - 15))
                    self.screen.blit(new_text, new_rect)
            
            # If a card is selected, show action buttons
            if self.selected_dev_card is not None:
                action_buttons_y = card_area_y + card_area_height - 40
                
                # Play card button
                selected_card = curr_player.dev_cards[self.selected_dev_card]
                if selected_card.card_type != DevCardType.VICTORY_POINT and not (curr_player.just_purchased_dev_card and self.selected_dev_card == len(curr_player.dev_cards) - 1):
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
            count_text = f"Select {self.game_logic.awaiting_resource_selection_count} resource(s)"
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
    
    def draw_robber_placement(self):
        """Highlight hexes for robber placement when needed"""
        if not self.game_logic.state.awaiting_robber_placement and not self.robber_placement_active:
            return
            
        # Draw an overlay on current robber hex
        if self.game_logic.state.robber_hex_id is not None:
            center = self.hex_centers[self.game_logic.state.robber_hex_id]
            pygame.draw.circle(self.screen, (0, 0, 0, 128), center, self.number_circle_radius * 1.5)
            robber_text = self.info_font.render("R", True, (255, 255, 255))
            robber_rect = robber_text.get_rect(center=center)
            self.screen.blit(robber_text, robber_rect)
        
        # Instruction text
        if self.game_logic.state.awaiting_robber_placement or self.robber_placement_active:
            text = self.instruction_font.render("Click on a hex to move the robber", True, (255, 0, 0))
            text_rect = text.get_rect(center=(self.window_width * 0.4, 30))
            self.screen.blit(text, text_rect)
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
        if not hasattr(self, 'steal_buttons') or not self.steal_buttons:
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
        if not hasattr(self, 'steal_buttons'):
            return None
            
        for player_idx, button_rect in self.steal_buttons:
            if button_rect.collidepoint(mouse_pos):
                return player_idx
        return None
    
    def draw_hexes(self):
        """Draw all hexagons with their resources and numbers"""
        for hex_id, hex_obj in self.board.hexes.items():
            vertices = self.hex_vertices[hex_id]
            center = self.hex_centers[hex_id]
            
            # Draw the hex
            color = RESOURCE_COLORS[hex_obj.resource]
            pygame.draw.polygon(self.screen, color, vertices)
            pygame.draw.polygon(self.screen, TEXT_COLOR, vertices, 2)  # Border
            
            # Draw the resource name (at the top)
            # hiding for now
            """
            resource_text = self.font.render(hex_obj.resource.name, True, TEXT_COLOR)
            resource_rect = resource_text.get_rect(center=(center[0], center[1] - vertical_spacing))
            self.screen.blit(resource_text, resource_rect)
            """
            
            # Draw the number with a larger circle
            if hex_obj.number > 0:  # Desert has no number
                number_text = self.number_font.render(str(hex_obj.number), True, TEXT_COLOR)
                number_rect = number_text.get_rect(center=center)
                
                # Draw a larger white circle behind the number
                pygame.draw.circle(self.screen, SPOT_COLOR, center, self.number_circle_radius)
                pygame.draw.circle(self.screen, TEXT_COLOR, center, self.number_circle_radius, 2)
                
                # Add red for 6 and 8 (high probability numbers)
                if hex_obj.number in [6, 8]:
                    number_text = self.number_font.render(str(hex_obj.number), True, (255, 0, 0))
                
                self.screen.blit(number_text, number_rect)
            
            # Draw the hex ID (smaller, below number)
            # hiding for now
            """
            id_text = self.font.render(f"ID:{hex_id}", True, TEXT_COLOR)
            id_rect = id_text.get_rect(center=(center[0], center[1] + vertical_spacing))
            self.screen.blit(id_text, id_rect)
            """
    
    def draw_roads(self):
        """Draw all roads"""
        for road_id, road_info in self.road_positions.items():
            start_pos, end_pos = road_info['endpoints']
            road = self.board.get_road(road_id)
            
            color = ROAD_COLOR
            # If road is owned, draw with player color
            if road.owner is not None:
                color = PLAYER_COLOR_RGBS[road.owner]

            # In setup phase with settlement placed, check if road is valid
            elif self.game_logic.state.setup_phase_settlement_placed and self.last_settlement_placed is not None:
                valid = self.game_logic.is_valid_initial_road(road_id, self.last_settlement_placed)
                if valid:
                    color = ROAD_VALID_COLOR
                
            pygame.draw.line(self.screen, color, start_pos, end_pos, self.road_width)
    
    def draw_spots(self):
        """Draw all spots (vertices) with settlements if present"""
        for spot_id, pos in self.spot_positions.items():
            spot = self.board.get_spot(spot_id)
            
            # Determine spot color based on state
            if spot.player_idx is not None:
                # If spot has a settlement, color it by player
                color_key = PLAYER_COLOR_NAMES[spot.player_idx]
                
                # Select the appropriate image based on the settlement type
                if spot.settlement_type == SettlementType.SETTLEMENT:
                    img = self.settlement_images.get(color_key)
                elif spot.settlement_type == SettlementType.CITY:
                    img = self.city_images.get(color_key)
                else:
                    img = None
                
                if img:
                    # Center the image on the spot position
                    img_rect = img.get_rect(center=pos)
                    self.screen.blit(img, img_rect)
            else:   
                color = SPOT_COLOR
                pygame.draw.circle(self.screen, color, pos, self.spot_radius)
                pygame.draw.circle(self.screen, TEXT_COLOR, pos, self.spot_radius, 1)  # Border

                """
                spot_text = self.font.render(str(spot_id), True, TEXT_COLOR)
                spot_rect = spot_text.get_rect(center=pos)
                self.screen.blit(spot_text, spot_rect)
                """
    
    def check_dice_click(self, mouse_pos):
        """Return True if the mouse click is inside the dice area."""
        if hasattr(self, 'dice_rect'):
            return self.dice_rect.collidepoint(mouse_pos)
        return False

        
    def check_spot_click(self, mouse_pos):
        """Check if any spot was clicked"""
        for spot_id, pos in self.spot_positions.items():
            distance = math.sqrt((mouse_pos[0] - pos[0])**2 + (mouse_pos[1] - pos[1])**2)
            if distance <= self.spot_radius:
                return spot_id
        return None
    
    def check_road_click(self, mouse_pos):
        """Check if any road was clicked using line distance calculation"""
        for road_id, road_info in self.road_positions.items():
            p1, p2 = road_info['endpoints']
            
            # Calculate perpendicular distance from mouse to line
            # Line equation: Ax + By + C = 0
            A = p2[1] - p1[1]
            B = p1[0] - p2[0]
            C = p2[0] * p1[1] - p1[0] * p2[1]
            
            # Distance = |Ax + By + C| / sqrt(A² + B²)
            distance = abs(A * mouse_pos[0] + B * mouse_pos[1] + C) / math.sqrt(A**2 + B**2)
            
            # Check if mouse is within the line segment
            # First check if the click is close enough to the line
            if distance <= self.road_width / 2:
                # Then check if it's within the segment (not just the infinite line)
                # Use a bounding box check with a little margin
                min_x = min(p1[0], p2[0]) - self.road_width
                max_x = max(p1[0], p2[0]) + self.road_width
                min_y = min(p1[1], p2[1]) - self.road_width
                max_y = max(p1[1], p2[1]) + self.road_width
                
                if (min_x <= mouse_pos[0] <= max_x and min_y <= mouse_pos[1] <= max_y):
                    return road_id
        
        return None
    
    def display_info_panel(self):
        """Display information panel in the top left corner"""
        # Create a corner panel instead of full width
        panel_width = int(self.window_width * 0.3)  # 30% of window width
        panel_height = 80
        panel_margin = 10
        
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((255, 255, 255, 200))  # Semi-transparent white
        self.screen.blit(panel_surface, (panel_margin, panel_margin))
        
        x_offset = panel_margin * 2
        y_offset = panel_margin * 2
        
        # Display game phase instructions
        if not self.game_logic.is_setup_complete():
            instructions = self.game_logic.get_setup_instructions()
            current_player = self.game_logic.state.get_current_player()
            player_color = PLAYER_COLOR_RGBS[current_player.player_idx]
            
            # Render with player color - use smaller font for instructions in corner panel
            instruction_surface = self.info_font.render(instructions, True, player_color)
            self.screen.blit(instruction_surface, (x_offset, y_offset))
            y_offset += 25
        else:
            # Regular play instructions (to be implemented)
            pass

    
    def draw_end_turn_button(self):
        """Draw an end turn button in the bottom left corner"""
        button_width = 150
        button_height = 40
        button_margin = 20
        button_x = button_margin
        button_y = self.window_height - button_height - button_margin
    
        # Create button rectangle
        self.end_turn_button = pygame.Rect(button_x, button_y, button_width, button_height)

        if self.game_logic.user_can_end_turn():
            color = END_BUTTON_ENABLED_COLOR
        else:
            color = END_BUTTON_DISABLED_COLOR
        pygame.draw.rect(self.screen, color, self.end_turn_button)
        pygame.draw.rect(self.screen, TEXT_COLOR, self.end_turn_button, 2)  # Border
    
        # Button text
        button_text = self.info_font.render("End Turn", True, TEXT_COLOR)
        text_rect = button_text.get_rect(center=self.end_turn_button.center)
        self.screen.blit(button_text, text_rect)
    
    def draw_player_status(self):
        """Draw player status panel on the right side of the screen"""
        panel_width = int(self.window_width * 0.2)  # 20% of screen width
        panel_height = self.window_height
        panel_x = self.window_width - panel_width
        
        # Create semi-transparent panel
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((220, 220, 220, 180))  # Light gray with transparency
        self.screen.blit(panel_surface, (panel_x, 0))
        
        # Draw player information
        title_y = 20
        title = self.info_font.render("PLAYERS", True, TEXT_COLOR)
        title_rect = title.get_rect(center=(panel_x + panel_width // 2, title_y))
        self.screen.blit(title, title_rect)
        
        # Draw each player's status
        y_pos = title_y + 40
        for player in self.players:
            # Player header with background in player's color
            player_rect = pygame.Rect(panel_x + 10, y_pos, panel_width - 20, 30)
            pygame.draw.rect(self.screen, PLAYER_COLOR_RGBS[player.player_idx], player_rect)
            pygame.draw.rect(self.screen, TEXT_COLOR, player_rect, 1)  # Border
            
 
            player_name_string = player.name
            if player.player_idx == self.game_logic.state.current_player_idx:
                player_name_string = f"{player_name_string} - (curr turn)"
            player_text = self.info_font.render(player_name_string, True, TEXT_COLOR)            
            self.screen.blit(player_text, (panel_x + 15, y_pos + 5))
            
            y_pos += 35
        
            
            # Settlements and Roads count
            if player.settlements:
                settlements_text = self.font.render(f"Settlements: {len(player.settlements)}", True, TEXT_COLOR)
                self.screen.blit(settlements_text, (panel_x + 15, y_pos))
                y_pos += 20
            
            if player.roads:
                roads_text = self.font.render(f"Roads: {len(player.roads)}", True, TEXT_COLOR)
                self.screen.blit(roads_text, (panel_x + 15, y_pos))
                y_pos += 20
            
            # Resources (only show in regular play phase or to the current player in setup phase 2)
            if (self.game_logic.is_setup_complete() or 
                (self.game_logic.state.current_phase == GamePhase.SETUP_PHASE_2 and 
                 player.player_idx == self.game_logic.state.current_player_idx)):
                
                y_pos += 5
                resources_title = self.font.render("Resources:", True, TEXT_COLOR)
                self.screen.blit(resources_title, (panel_x + 15, y_pos))
                y_pos += 20
                card_margin = 5
                resources_order = [Resource.WOOD, Resource.BRICK, Resource.WHEAT, Resource.SHEEP, Resource.ORE]
                num_cards = len(resources_order)
                # Compute card width based on the available panel width (panel_width is defined earlier)
                card_width = (panel_width - (num_cards + 1) * card_margin) / num_cards
                card_height = card_width * 1.2  # Adjust this ratio as needed
                x_start = panel_x + card_margin
                y_resource = y_pos  # Use the current y position as the top of the resource cards row
                for res in resources_order:
                    rect = pygame.Rect(x_start, y_resource, card_width, card_height)
                    color = RESOURCE_COLORS.get(res, TEXT_COLOR)
                    pygame.draw.rect(self.screen, color, rect)

                    pygame.draw.rect(self.screen, TEXT_COLOR, rect, 2)
                    count = player.resources.get(res, 0)
                    count_text = self.font.render(str(count), True, TEXT_COLOR)
                    text_rect = count_text.get_rect(center=rect.center)

                    self.screen.blit(count_text, text_rect)
 

                    x_start += card_width + card_margin

                y_pos += card_height + card_margin
            
            # Add spacing between players
            y_pos += 15

    def check_end_turn_button(self, mouse_pos):
        """Check if the end turn button was clicked"""
        if hasattr(self, 'end_turn_button') and self.end_turn_button.collidepoint(mouse_pos) and self.game_logic.user_can_end_turn():
            return True
        return False
    def check_resource_selection_click(self, mouse_pos):
        """Check if a resource selection button was clicked"""
        for action_type, resource, button_rect in self.resource_selection_buttons:
            if button_rect.collidepoint(mouse_pos):
                return (action_type, resource)
        return None

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

    def check_hex_click(self, mouse_pos):
        """Check if a hex was clicked (for robber placement)"""
        if not (self.game_logic.state.awaiting_robber_placement or self.robber_placement_active):
            return None
            
        for hex_id, center in self.hex_centers.items():
            distance = math.sqrt((mouse_pos[0] - center[0])**2 + (mouse_pos[1] - center[1])**2)
            # Use a generous radius for hex click detection
            if distance <= self.number_circle_radius * 2:
                # Don't allow placing robber on same hex
                if hex_id != self.game_logic.state.robber_hex_id:
                    return hex_id
        return None
    def run(self):
        running = True
        while running:
            # Process events for human players
            if self.game_logic.is_current_player_human():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click
                        mouse_pos = pygame.mouse.get_pos()
                        
                        # Check if end turn button was clicked
                        if self.check_end_turn_button(mouse_pos):
                            self.game_logic.end_turn()
                            continue
                        
                        # Don't process clicks on the player panel area (right 20% of screen)
                        if mouse_pos[0] < self.window_width * 0.8:
                            # In placement phase - handle selection of two settlements and roads
                            if not self.game_logic.is_setup_complete():
                                if not self.game_logic.state.setup_phase_settlement_placed:
                                    # Attempt to place settlement immediately on click
                                    spot_id = self.check_spot_click(mouse_pos)
                                    if spot_id is not None:
                                        successfully_placed = self.game_logic.place_initial_settlement(spot_id)
                                        if successfully_placed:
                                            print(f"Placed settlement at spot {spot_id}")
                                            self.last_settlement_placed = spot_id
                                        else:
                                            print(f"Failed to place settlement at {spot_id}")
                                else:
                                    # Settlement already placed; attempt to place road immediately
                                    road_id = self.check_road_click(mouse_pos)
                                    if road_id is not None:
                                        successfully_placed = self.game_logic.place_initial_road(road_id, self.last_settlement_placed)
                                        if successfully_placed:
                                            print(f"Placed road at {road_id}")
                                            self.last_settlement_placed = None
                                        else:
                                            print(f"Failed to place road at {road_id}")

                            else:
                                # Regular play phase - handle special states first
                                
                                # Robber placement
                                if self.game_logic.state.awaiting_robber_placement or self.robber_placement_active:
                                    hex_id = self.check_hex_click(mouse_pos)
                                    if hex_id is not None:
                                        success = self.game_logic.move_robber(hex_id)
                                        if success:
                                            
                                            print(f"Moved robber to hex {hex_id}")
                                            self.robber_placement_active = False
                                            if hasattr(self.game_logic, 'potential_victims') and self.game_logic.state.potential_victims:
                                                self.setup_steal_from_player(self.game_logic.state.potential_victims)
                                        else:
                                            print(f"Failed to move robber to hex {hex_id}")
                                    continue
                                if hasattr(self, 'steal_buttons') and self.steal_buttons:
                                    victim_id = self.check_steal_button_click(mouse_pos)
                                    if victim_id is not None:
                                        self.game_logic.state.steal_resource_from_player(victim_id)
                                        self.steal_buttons = []
                                        continue
                                # Resource selection for Year of Plenty or Monopoly
                                resource_action = self.check_resource_selection_click(mouse_pos)
                                if resource_action:
                                    action_type, resource = resource_action
                                    if action_type == "year_of_plenty" and self.game_logic.state.awaiting_resource_selection:
                                        self.game_logic.select_year_of_plenty_resource(resource)
                                        print(f"Selected {resource.name} for Year of Plenty")
                                    elif action_type == "monopoly" and self.game_logic.state.awaiting_monopoly_selection:
                                        self.game_logic.select_monopoly_resource(resource)
                                        print(f"Selected {resource.name} for Monopoly")
                                    continue
                                
                                # Handle development card actions
                                dev_card_idx = self.check_dev_card_click(mouse_pos)
                                if dev_card_idx is not None:
                                    self.selected_dev_card = dev_card_idx
                                    continue
                                
                                dev_card_action = self.check_dev_card_action_click(mouse_pos)
                                if dev_card_action:
                                    if dev_card_action == "buy_dev_card":
                                        success = self.game_logic.buy_development_card()
                                        if success:
                                            print("Bought development card")
                                        else:
                                            print("Failed to buy development card")
                                    elif dev_card_action == "play_knight":
                                        success = self.game_logic.play_knight_card()
                                        if success:
                                            print("Played Knight card")
                                            self.robber_placement_active = True
                                    elif dev_card_action == "play_road_building":
                                        success = self.game_logic.play_road_building_card()
                                        if success:
                                            print("Played Road Building card")
                                    elif dev_card_action == "play_year_of_plenty":
                                        success = self.game_logic.play_year_of_plenty_card()
                                        if success:
                                            print("Played Year of Plenty card")
                                    elif dev_card_action == "play_monopoly":
                                        success = self.game_logic.play_monopoly_card()
                                        if success:
                                            print("Played Monopoly card")
                                    
                                    # Clear selection after action
                                    self.selected_dev_card = None
                                    continue
                                
                                # Roll dice
                                if self.check_dice_click(mouse_pos):
                                    self.game_logic.roll_dice()
                                    continue

                                # Handle road building (from dev card)
                                if self.game_logic.state.road_building_roads_placed > 0 and self.game_logic.state.road_building_roads_placed < 2:
                                    road_id = self.check_road_click(mouse_pos)
                                    if road_id is not None:
                                        success = self.game_logic.place_free_road(road_id)
                                        if success:
                                            print(f"Placed free road at {road_id}")
                                        else:
                                            print(f"Failed to place free road at {road_id}")
                                        continue
                                
                                # Regular build actions
                                spot_id = self.check_spot_click(mouse_pos)
                                if spot_id is not None:
                                    # Check for build/upgrade actions
                                    if ("build_settlement", spot_id) in self.game_logic.state.possible_actions:
                                        success = self.game_logic.do_move(("build_settlement", spot_id))
                                        if success:
                                            print(f"Built settlement at spot {spot_id}")
                                        else:
                                            print(f"Failed to build settlement at {spot_id}")
                                    elif ("upgrade_city", spot_id) in self.game_logic.state.possible_actions:
                                        success = self.game_logic.do_move(("upgrade_city", spot_id))
                                        if success:
                                            print(f"Upgraded to city at spot {spot_id}")
                                        else:
                                            print(f"Failed to upgrade to city at {spot_id}")
                                    else:
                                        print(f"No valid action available at spot {spot_id}")
                                    continue
                                
                                # Build road
                                road_id = self.check_road_click(mouse_pos)
                                if road_id is not None:
                                    if ("road", road_id) in self.game_logic.state.possible_actions:
                                        success = self.game_logic.do_move(("road", road_id))
                                        if success:
                                            print(f"Built road at {road_id}")
                                        else:
                                            print(f"Failed to build road at {road_id}")
                                    else:
                                        print(f"No valid road action at {road_id}")
                                    continue
                                

            else:
                # Handle AI player's turn with a slight delay
                self.ai_thinking_timer += 1
                
                # Process any quit events even during AI turns
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                
                # Add a small delay to make AI turns visible (every 30 frames = 0.5 seconds at 60fps)
                if self.ai_thinking_timer >= 30:
                    self.ai_thinking_timer = 0
                    
                    # Let the AI make its move
                    self.game_logic.process_ai_turn()
            
            # Clear the screen
            self.screen.fill(BACKGROUND_COLOR)
            
            # Draw all elements
            self.draw_board_border()
            self.draw_hexes()
            self.draw_roads()
            self.draw_spots()
            self.draw_player_status()
            self.draw_dice()
            self.draw_end_turn_button()
            
            # Draw development card elements
            self.draw_development_cards()
            self.draw_resource_selection()
            self.draw_robber_placement()
            self.draw_steal_selection()
            
            # Update the display
            pygame.display.flip()
            self.clock.tick(60)
    
        pygame.quit()   
    


def display_player_setup_menu():
    """Display a menu to setup players and agent types"""
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    pygame.display.set_caption("Catan Player Setup")
    
    font_title = pygame.font.SysFont('Arial', 32, bold=True)
    font_text = pygame.font.SysFont('Arial', 22)
    font_button = pygame.font.SysFont('Arial', 24, bold=True)
    
    # Default setup
    num_human_players = 1
    agent_types = [AgentType.HEURISTIC] * 3  # Default all AI to random
    
    # Start button
    start_button = pygame.Rect(200, 320, 200, 50)
    
    running = True
    
    while running:
        screen.fill((240, 240, 240))
        
        # Title
        title = font_title.render("Settlers of Catan - Player Setup", True, (0, 0, 0))
        screen.blit(title, (600//2 - title.get_width()//2, 30))
        
        # Number of human players
        human_text = font_text.render(f"Human Players: {num_human_players}", True, (0, 0, 0))
        screen.blit(human_text, (100, 100))
        
        # Increase/decrease buttons
        inc_button = pygame.Rect(400, 100, 30, 30)
        dec_button = pygame.Rect(350, 100, 30, 30)
        pygame.draw.rect(screen, (100, 200, 100), inc_button)
        pygame.draw.rect(screen, (200, 100, 100), dec_button)
        screen.blit(font_button.render("+", True, (0, 0, 0)), (inc_button.x + 8, inc_button.y + 2))
        screen.blit(font_button.render("-", True, (0, 0, 0)), (dec_button.x + 11, dec_button.y + 2))
        
        # AI player info
        y_offset = 150
        for i in range(4 - num_human_players):
            ai_text = font_text.render(f"AI Player {i+num_human_players+1}: {agent_types[i].name}", True, (0, 0, 0))
            screen.blit(ai_text, (100, y_offset))
            y_offset += 40
        
        # Draw start button
        pygame.draw.rect(screen, (100, 150, 200), start_button)
        start_text = font_button.render("Start Game", True, (0, 0, 0))
        screen.blit(start_text, (start_button.x + start_button.width//2 - start_text.get_width()//2, 
                                 start_button.y + start_button.height//2 - start_text.get_height()//2))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None, None  # Return None to exit the program
                
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                
                # Check if increment button is clicked
                if inc_button.collidepoint(mouse_pos) and num_human_players < 4:
                    num_human_players += 1
                
                # Check if decrement button is clicked
                elif dec_button.collidepoint(mouse_pos) and num_human_players > 1:
                    num_human_players -= 1
                
                # Check if start button is clicked
                elif start_button.collidepoint(mouse_pos):
                    pygame.quit()  # Close the menu window
                    return num_human_players, agent_types[:4-num_human_players]
    
    return None, None  # Should never reach here, but just in case

def main():
    # Show player setup menu
    num_human_players, agent_types = display_player_setup_menu()
    
    # Exit if menu was closed without starting
    if num_human_players is None:
        return
    
    # Start the game with selected setup
    game = CatanGame(
        num_human_players=num_human_players,
        agent_types=agent_types
    )
    game.run()

if __name__ == "__main__":
    main()