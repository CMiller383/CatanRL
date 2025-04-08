"""
Renderer for the Catan game GUI.
"""
import os
import pygame
from game.rules import is_valid_initial_road
from gui.constants import *
from game.enums import GamePhase, SettlementType
from game.development_card import DevCardType


class Renderer:
    def __init__(self, screen, window_width, window_height, board, game_logic):
        self.screen = screen
        self.window_width = window_width
        self.window_height = window_height
        self.board = board
        self.game_logic = game_logic
        self.players = game_logic.state.players

        # Font sizes scaled based on resolution
        base_font_size = int(min(window_width, window_height) / 50)
        self.font = pygame.font.SysFont('Arial', base_font_size)
        self.number_font = pygame.font.SysFont('Arial', int(base_font_size * 1.5), bold=True)
        self.info_font = pygame.font.SysFont('Arial', int(base_font_size * 1.2))
        self.instruction_font = pygame.font.SysFont('Arial', int(base_font_size * 1.8), bold=True)
        self.card_font = pygame.font.SysFont('Arial', int(base_font_size * 1.1), bold=True)
        self.card_desc_font = pygame.font.SysFont('Arial', int(base_font_size * 0.9))

        # Hit detection - scale these based on resolution
        self.spot_radius = int(min(window_width, window_height) / 80)
        self.road_width = int(min(window_width, window_height) / 100)
        self.number_circle_radius = int(min(window_width, window_height) / 40)
        self.settlement_size = int(self.spot_radius * 1.5)
        
        # Calculate scale and offset to fit the board in the window
        self.scale, self.offset = self.compute_transform()
        
        # Calculate positions
        self.spot_positions = {}
        self.calculate_spot_positions()
        
        self.road_positions = {}
        self.calculate_road_positions()
        
        # Hex centers and vertices for drawing
        self.hex_centers = {}
        self.hex_vertices = {}
        self.calculate_hex_positions()
        
        # UI Elements
        self.end_turn_button = None
        self.dice_rect = None
        
        # Load images
        self.settlement_images = {}
        self.city_images = {}
        self.dice_images = {}
        self.border_image = None
        self.diceroll_image = None
        self.dev_card_icons = {}
        self.load_images()

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
        import math
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
    
    def load_images(self):
        """Load and scale images for settlements and cities"""
        # Define the target size based on the spot radius
        img_size = self.spot_radius * 10

        # Load settlement images
        settlement_folder = 'gui/assets/settlements'
        for filename in os.listdir(settlement_folder):
            if filename.endswith('.png'):
                key = filename.replace('_settlement.png', '')
                image_path = os.path.join(settlement_folder, filename)
                img = pygame.image.load(image_path).convert_alpha()
                img = pygame.transform.smoothscale(img, (img_size, img_size))
                self.settlement_images[key] = img

        # Load city images
        city_folder = 'gui/assets/cities'
        for filename in os.listdir(city_folder):
            if filename.endswith('.png'):
                key = filename.replace('_city.png', '')
                image_path = os.path.join(city_folder, filename)
                img = pygame.image.load(image_path).convert_alpha()
                img = pygame.transform.smoothscale(img, (img_size, img_size))
                self.city_images[key] = img

        # Load sand border image
        original = pygame.image.load('gui/assets/border.png').convert_alpha()
        border_size = int(self.window_height * 1.07)
        self.border_image = pygame.transform.smoothscale(original, (border_size, border_size))

        # Load dice images
        dice_folder = 'gui/assets/dice'
        for i in range(1, 7):
            filename = f"dice{i}.png"
            path = os.path.join(dice_folder, filename)
            img = pygame.image.load(path).convert_alpha()
            dice_size = int(self.window_width * 0.1)
            img = pygame.transform.smoothscale(img, (dice_size, dice_size))
            self.dice_images[i] = img

        # Load diceroll prompt image
        diceroll_path = os.path.join(dice_folder, "diceroll.png")
        self.diceroll_image = pygame.image.load(diceroll_path).convert_alpha()
        diceroll_size = int(self.window_width * 0.15)
        self.diceroll_image = pygame.transform.smoothscale(self.diceroll_image, (diceroll_size, diceroll_size))
        
        # Load development card icons if available
        try:
            dev_card_folder = 'gui/assets/dev_cards'
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
    
    def draw_board_border(self):
        """Draws the border image behind the board."""
        # Use hex 10's screen position as the center of the board.
        center = self.hex_centers.get(10)
        if center is None:
            return

        # Get the rect of the scaled image and center it on the board.
        border_rect = self.border_image.get_rect(center=center)
        self.screen.blit(self.border_image, border_rect)

    def draw_hexes(self):
        """Draw all hexagons with their resources and numbers"""
        for hex_id, hex_obj in self.board.hexes.items():
            vertices = self.hex_vertices[hex_id]
            center = self.hex_centers[hex_id]
            
            # Draw the hex
            color = RESOURCE_COLORS[hex_obj.resource]
            pygame.draw.polygon(self.screen, color, vertices)
            pygame.draw.polygon(self.screen, TEXT_COLOR, vertices, 2)  # Border
            
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
    
    def draw_roads(self, last_settlement_placed=None):
        """Draw all roads"""
        for road_id, road_info in self.road_positions.items():
            start_pos, end_pos = road_info['endpoints']
            road = self.board.get_road(road_id)
            
            color = ROAD_COLOR
            # If road is owned, draw with player color
            if road.owner is not None:
                color = PLAYER_COLOR_RGBS[road.owner]
            # In setup phase with settlement placed, check if road is valid
            elif self.game_logic.state.setup_phase_settlement_placed and last_settlement_placed is not None:
                valid = is_valid_initial_road(self.game_logic.state, road_id, last_settlement_placed)
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
    
    def draw_dice(self):
        """Draw dice 1 and dice 2 on the bottom right of the screen."""
        margin_x = self.window_width * .2
        margin_y = 0

        dice_size = self.dice_images[1].get_width()

        # Compute positions for dice
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
    
    def draw_end_turn_button(self):
        """Draw an end turn button in the bottom left corner"""
        button_width = 150
        button_height = 40
        button_margin = 20
        button_x = button_margin
        button_y = self.window_height - button_height - button_margin
    
        # Create button rectangle
        self.end_turn_button = pygame.Rect(button_x, button_y, button_width, button_height)

        color = END_BUTTON_DISABLED_COLOR
        if self.game_logic.user_can_end_turn():
            color = END_BUTTON_ENABLED_COLOR

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
            
            if (self.game_logic.state.current_phase != GamePhase.SETUP_PHASE_1):                
                y_pos += 5
                resources_title = self.font.render("Resources:", True, TEXT_COLOR)
                self.screen.blit(resources_title, (panel_x + 15, y_pos))
                y_pos += 20
                card_margin = 5
                resources_order = [Resource.WOOD, Resource.BRICK, Resource.WHEAT, Resource.SHEEP, Resource.ORE]
                num_cards = len(resources_order)
                # Compute card width based on the available panel width
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
            
    def draw_robber_placement(self, robber_placement_active=False):
        """Highlight hexes for robber placement when needed"""
        if not self.game_logic.state.awaiting_robber_placement and not robber_placement_active:
            return
            
        # Draw an overlay on current robber hex
        if self.game_logic.state.robber_hex_id is not None:
            center = self.hex_centers[self.game_logic.state.robber_hex_id]
            pygame.draw.circle(self.screen, (0, 0, 0, 128), center, self.number_circle_radius * 1.5)
            robber_text = self.info_font.render("R", True, (255, 255, 255))
            robber_rect = robber_text.get_rect(center=center)
            self.screen.blit(robber_text, robber_rect)
        
        # Instruction text
        if self.game_logic.state.awaiting_robber_placement or robber_placement_active:
            text = self.instruction_font.render("Click on a hex to move the robber", True, (255, 0, 0))
            text_rect = text.get_rect(center=(self.window_width * 0.4, 30))
            self.screen.blit(text, text_rect)
            
    def draw_steal_selection(self, steal_buttons=None, steal_panel=None):
        """Draw UI for selecting a player to steal from"""
        if not steal_buttons:
            return
        
        # Draw panel background
        panel_rect = steal_panel['rect']
        panel_surface = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
        panel_surface.fill((240, 240, 240, 230))  # Light gray with higher opacity
        self.screen.blit(panel_surface, panel_rect)
        
        # Draw title
        title_text = self.info_font.render("Select a player to steal from:", True, TEXT_COLOR)
        title_rect = title_text.get_rect(center=(panel_rect.centerx, panel_rect.y + 20))
        self.screen.blit(title_text, title_rect)
        
        # Draw player buttons
        for player_idx, button_rect in steal_buttons:
            player = self.players[player_idx]
            color = PLAYER_COLOR_RGBS[player_idx]
            
            pygame.draw.rect(self.screen, color, button_rect)
            pygame.draw.rect(self.screen, TEXT_COLOR, button_rect, 1)  # Border
            
            text = self.info_font.render(f"{player.name}", True, TEXT_COLOR)
            text_rect = text.get_rect(center=button_rect.center)
            self.screen.blit(text, text_rect)