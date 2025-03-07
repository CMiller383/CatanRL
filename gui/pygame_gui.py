import pygame
import math
from game.board import Board
from game.resource import Resource
from game.spot import SettlementType
from game.game_logic import GameLogic, GamePhase


RESOURCE_COLORS = {
    Resource.WOOD: (60, 179, 113),    # Medium Sea Green
    Resource.BRICK: (205, 92, 92),    # Indian Red
    Resource.WHEAT: (255, 215, 0),     # Gold
    Resource.SHEEP: (152, 251, 152),   # Pale Green
    Resource.ORE: (169, 169, 169),     # Dark Gray
    Resource.DESERT: (244, 164, 96)    # Sandy Brown
}

PLAYER_COLORS = {
    1: (255, 99, 71),    # Tomato Red
    2: (65, 105, 225),   # Royal Blue
    3: (255, 255, 255),  # White
    4: (138, 43, 226)    # Blue Violet
}

ROAD_COLOR = (160, 82, 45)           # Sienna
ROAD_HIGHLIGHT_COLOR = (255, 69, 0)    # Orange Red
ROAD_INVALID_COLOR = (169, 169, 169)   # Dark Gray

SPOT_COLOR = (255, 255, 255)           # White
SPOT_HIGHLIGHT_COLOR = (255, 215, 0)   # Gold
SPOT_INVALID_COLOR = (169, 169, 169)   # Dark Gray

TEXT_COLOR = (0, 0, 0)                 # Black
BACKGROUND_COLOR = (176, 224, 230)     # Powder Blue


# Global screen proportion (how much of the screen to use)
SCREEN_PROPORTION = 0.75

from agent.base import AgentType

class CatanGame:
    def __init__(self, window_width=None, window_height=None, num_human_players=1, agent_types=None):
        pygame.init()
        
        # Auto-detect screen size if not provided
        if window_width is None or window_height is None:
            screen_info = pygame.display.Info()
            max_width, max_height = screen_info.current_w, screen_info.current_h
            
            # Apply the screen proportion
            window_width = int(max_width * SCREEN_PROPORTION)
            window_height = int(max_height * SCREEN_PROPORTION)
            
            # Ensure it's a square or landscape
            window_width = max(window_width, window_height)
        
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
        
        self.board = Board()
        
        # Initialize game logic with player setup
        self.game_logic = GameLogic(self.board, num_human_players, agent_types)
        self.players = self.game_logic.players  # Reference to players for easier access
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
        self.selected_spot = None
        self.selected_road = None
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
        self.selected_spot = None
        self.selected_road = None
        self.last_settlement_placed = None  # To track the most recent settlement for road placement
    
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
    
    def draw_hexes(self):
        """Draw all hexagons with their resources and numbers"""
        for hex_id, hex_obj in self.board.hexes.items():
            vertices = self.hex_vertices[hex_id]
            center = self.hex_centers[hex_id]
            
            # Draw the hex
            color = RESOURCE_COLORS[hex_obj.resource]
            pygame.draw.polygon(self.screen, color, vertices)
            pygame.draw.polygon(self.screen, TEXT_COLOR, vertices, 2)  # Border
            
            # Calculate vertical spacing based on circle size
            vertical_spacing = self.number_circle_radius + 5
            
            # Draw the resource name (at the top)
            resource_text = self.font.render(hex_obj.resource.name, True, TEXT_COLOR)
            resource_rect = resource_text.get_rect(center=(center[0], center[1] - vertical_spacing))
            self.screen.blit(resource_text, resource_rect)
            
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
            id_text = self.font.render(f"ID:{hex_id}", True, TEXT_COLOR)
            id_rect = id_text.get_rect(center=(center[0], center[1] + vertical_spacing))
            self.screen.blit(id_text, id_rect)
    
    def draw_roads(self):
        """Draw all roads"""
        for road_id, road_info in self.road_positions.items():
            start_pos, end_pos = road_info['endpoints']
            road = self.board.get_road(road_id)
            
            # If road is owned, draw with player color
            if road.owner is not None:
                color = PLAYER_COLORS[road.owner]
            # In setup phase with settlement placed, check if road is valid
            elif self.game_logic.setup_phase_settlement_placed and self.last_settlement_placed is not None:
                valid = self.game_logic.is_valid_initial_road(road_id, self.last_settlement_placed)
                if road_id == self.selected_road:
                    color = ROAD_HIGHLIGHT_COLOR
                elif valid:
                    color = ROAD_COLOR
                else:
                    color = ROAD_INVALID_COLOR
            # Normal road
            else:
                color = ROAD_HIGHLIGHT_COLOR if road_id == self.selected_road else ROAD_COLOR
                
            pygame.draw.line(self.screen, color, start_pos, end_pos, self.road_width)
    
    def draw_spots(self):
        """Draw all spots (vertices) with settlements if present"""
        current_player = self.game_logic.get_current_player()
        
        for spot_id, pos in self.spot_positions.items():
            spot = self.board.get_spot(spot_id)
            
            # Determine spot color based on state
            if spot.player is not None:
                # If spot has a settlement, color it by player
                player_color = PLAYER_COLORS[spot.player]
                
                # Draw settlement
                if spot.settlement_type == SettlementType.SETTLEMENT:
                    # Draw a house shape
                    house_points = [
                        (pos[0], pos[1] - self.settlement_size),  # Top point
                        (pos[0] - self.settlement_size, pos[1]),  # Bottom left
                        (pos[0] + self.settlement_size, pos[1])   # Bottom right
                    ]
                    pygame.draw.polygon(self.screen, player_color, house_points)
                    pygame.draw.polygon(self.screen, TEXT_COLOR, house_points, 2)  # Border
                elif spot.settlement_type == SettlementType.CITY:
                    # Draw a larger square for city (not used in setup phase)
                    city_rect = pygame.Rect(
                        pos[0] - self.settlement_size, 
                        pos[1] - self.settlement_size,
                        self.settlement_size * 2, 
                        self.settlement_size * 2
                    )
                    pygame.draw.rect(self.screen, player_color, city_rect)
                    pygame.draw.rect(self.screen, TEXT_COLOR, city_rect, 2)  # Border
            else:
                # Empty spot - determine if it's valid for placement in setup phase
                if not self.game_logic.setup_phase_settlement_placed and not self.game_logic.is_setup_complete():
                    valid = self.game_logic.is_valid_initial_settlement(spot_id)
                    
                    if spot_id == self.selected_spot:
                        color = SPOT_HIGHLIGHT_COLOR
                    elif valid:
                        color = SPOT_COLOR
                    else:
                        color = SPOT_INVALID_COLOR
                        
                    pygame.draw.circle(self.screen, color, pos, self.spot_radius)
                    pygame.draw.circle(self.screen, TEXT_COLOR, pos, self.spot_radius, 1)  # Border
                    
                    # Render spot ID
                    spot_text = self.font.render(str(spot_id), True, TEXT_COLOR)
                    spot_rect = spot_text.get_rect(center=pos)
                    self.screen.blit(spot_text, spot_rect)
    
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
            current_player = self.game_logic.get_current_player()
            player_color = PLAYER_COLORS[current_player.player_id]
            
            # Render with player color - use smaller font for instructions in corner panel
            instruction_surface = self.info_font.render(instructions, True, player_color)
            self.screen.blit(instruction_surface, (x_offset, y_offset))
            y_offset += 25
        else:
            # Regular play instructions (to be implemented)
            pass
        
        # Display selected element info (shortened)
        if self.selected_spot is not None:
            spot = self.board.spots[self.selected_spot]
            info_text = f"Spot {self.selected_spot}"
            text_surface = self.font.render(info_text, True, TEXT_COLOR)
            self.screen.blit(text_surface, (x_offset, y_offset))
        
        elif self.selected_road is not None:
            road = self.board.roads[self.selected_road]
            info_text = f"Road {self.selected_road}: {road.spot1_id}-{road.spot2_id}"
            text_surface = self.font.render(info_text, True, TEXT_COLOR)
            self.screen.blit(text_surface, (x_offset, y_offset))
    
    def draw_end_turn_button(self):
        """Draw an end turn button in the bottom left corner"""
        button_width = 150
        button_height = 40
        button_margin = 20
        button_x = button_margin
        button_y = self.window_height - button_height - button_margin
    
        # Create button rectangle
        self.end_turn_button = pygame.Rect(button_x, button_y, button_width, button_height)
    
        # Color based on whether it's a human player's turn and if an action is selected
        if self.game_logic.is_current_player_human():
            # Highlight button if a valid action is selected
            if (not self.game_logic.is_setup_complete() and 
                ((not self.game_logic.setup_phase_settlement_placed and self.selected_spot is not None) or
                (self.game_logic.setup_phase_settlement_placed and self.selected_road is not None))):
                button_color = PLAYER_COLORS[self.game_logic.get_current_player().player_id]
            else:
                # Dimmed color when no valid action is selected
                base_color = PLAYER_COLORS[self.game_logic.get_current_player().player_id]
                button_color = (max(base_color[0] - 50, 0), 
                            max(base_color[1] - 50, 0), 
                            max(base_color[2] - 50, 0))
        else:
            button_color = (150, 150, 150)  # Gray for AI turns
    
        # Draw button
        pygame.draw.rect(self.screen, button_color, self.end_turn_button)
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
            pygame.draw.rect(self.screen, PLAYER_COLORS[player.player_id], player_rect)
            pygame.draw.rect(self.screen, TEXT_COLOR, player_rect, 1)  # Border
            
            player_text = self.info_font.render(f"{player.name}", True, TEXT_COLOR)
            self.screen.blit(player_text, (panel_x + 15, y_pos + 5))
            
            y_pos += 35
            
            # Mark current player
            if player.player_id == self.game_logic.get_current_player().player_id:
                current_text = self.font.render("CURRENT TURN", True, PLAYER_COLORS[player.player_id])
                self.screen.blit(current_text, (panel_x + 15, y_pos))
                y_pos += 25
            
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
                (self.game_logic.current_phase == GamePhase.SETUP_PHASE_2 and 
                 player.player_id == self.game_logic.get_current_player().player_id)):
                
                y_pos += 5
                resources_title = self.font.render("Resources:", True, TEXT_COLOR)
                self.screen.blit(resources_title, (panel_x + 15, y_pos))
                y_pos += 20
                
                for resource, count in player.resources.items():
                    if count > 0:  # Only show resources the player has
                        res_color = RESOURCE_COLORS.get(resource, TEXT_COLOR)
                        res_text = self.font.render(f"{resource.name}: {count}", True, res_color)
                        self.screen.blit(res_text, (panel_x + 25, y_pos))
                        y_pos += 20
            
            # Add spacing between players
            y_pos += 15

    def check_end_turn_button(self, mouse_pos):
        """Check if the end turn button was clicked"""
        if hasattr(self, 'end_turn_button') and self.end_turn_button.collidepoint(mouse_pos):
            return True
        return False
    
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
                            self.handle_end_turn()
                            continue
                        
                        # Don't process clicks on the player panel area (right 20% of screen)
                        if mouse_pos[0] < self.window_width * 0.8:
                            # In setup phase - handle selection of settlement and road
                            if not self.game_logic.is_setup_complete():
                                if not self.game_logic.setup_phase_settlement_placed:
                                    # We need to select a settlement
                                    spot_id = self.check_spot_click(mouse_pos)
                                    if spot_id is not None:
                                        # Only select the spot if it's valid for initial settlement
                                        if self.game_logic.is_valid_initial_settlement(spot_id):
                                            self.selected_spot = spot_id
                                            self.selected_road = None
                                            print(f"Selected Spot {spot_id}: {self.board.spots[spot_id]}")
                                        else:
                                            print(f"Invalid settlement location: {spot_id}")
                                else:
                                    # We need to select a road
                                    road_id = self.check_road_click(mouse_pos)
                                    if road_id is not None:
                                        # Only select the road if it's valid for initial road
                                        if self.game_logic.is_valid_initial_road(road_id, self.last_settlement_placed):
                                            self.selected_road = road_id
                                            self.selected_spot = None
                                            print(f"Selected Road {road_id}: {self.board.roads[road_id]}")
                                        else:
                                            print(f"Invalid road location: {road_id}")
                            else:
                                # Regular play phase (to be implemented)
                                spot_id = self.check_spot_click(mouse_pos)
                                if spot_id is not None:
                                    self.selected_spot = spot_id
                                    self.selected_road = None
                                    print(f"Selected Spot {spot_id}: {self.board.spots[spot_id]}")
                                else:
                                    road_id = self.check_road_click(mouse_pos)
                                    if road_id is not None:
                                        self.selected_road = road_id
                                        self.selected_spot = None
                                        print(f"Selected Road {road_id}: {self.board.roads[road_id]}")
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
                    
                    # Clear selection after AI move
                    self.selected_spot = None
                    self.selected_road = None
            
            # Clear the screen
            self.screen.fill(BACKGROUND_COLOR)
            
            # Draw all elements
            self.draw_hexes()
            self.draw_roads()
            self.draw_spots()
            self.draw_player_status()  # Draw player status panel
            self.display_info_panel()
            self.draw_end_turn_button()  # Draw end turn button
            
            # Update the display
            pygame.display.flip()
            self.clock.tick(60)
    
        pygame.quit()   
    
    def handle_end_turn(self):
        """Handle end turn button click"""
        # In setup phase
        if not self.game_logic.is_setup_complete():
            if not self.game_logic.setup_phase_settlement_placed and self.selected_spot is not None:
                # Try to place settlement
                if self.game_logic.place_initial_settlement(self.selected_spot):
                    print(f"Placed settlement at spot {self.selected_spot}")
                    # Remember this spot for road placement
                    self.last_settlement_placed = self.selected_spot
                    self.game_logic.last_settlement_placed = self.selected_spot
                    self.selected_spot = None
            
            elif self.game_logic.setup_phase_settlement_placed and self.selected_road is not None:
                # Try to place road
                if self.game_logic.place_initial_road(self.selected_road, self.last_settlement_placed):
                    print(f"Placed road at {self.selected_road}")
                    self.selected_road = None
                    self.last_settlement_placed = None
                    # End turn after placing both settlement and road
                    self.game_logic._advance_setup_phase()
        else:
            # Regular play phase
            # Here you would handle the various actions a player can take in the main game
            # For now, just end the current player's turn
            self.game_logic._advance_setup_phase()
            
            # Clear selections when ending turn
            self.selected_spot = None
            self.selected_road = None

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
    agent_types = [AgentType.RANDOM] * 3  # Default all AI to random
    
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