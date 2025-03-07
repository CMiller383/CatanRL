import pygame
import math
from game.board import Board
from game.resource import Resource
from game.spot import SettlementType
from game.game_logic import GameLogic, GamePhase

# Define colors
RESOURCE_COLORS = {
    Resource.WOOD: (34, 139, 34),    # Forest Green
    Resource.BRICK: (178, 34, 34),   # Firebrick
    Resource.WHEAT: (218, 165, 32),  # Goldenrod
    Resource.SHEEP: (144, 238, 144), # Light Green
    Resource.ORE: (105, 105, 105),   # Dim Gray
    Resource.DESERT: (238, 232, 170) # Light Tan
}

PLAYER_COLORS = {
    1: (220, 20, 20),    # Red
    2: (20, 20, 220),    # Blue
    3: (220, 220, 220),  # White
    4: (255, 140, 0)     # Orange
}

ROAD_COLOR = (139, 69, 19)           # Brown
ROAD_HIGHLIGHT_COLOR = (255, 140, 0) # Dark Orange
ROAD_INVALID_COLOR = (150, 150, 150) # Gray
SPOT_COLOR = (255, 255, 255)         # White
SPOT_HIGHLIGHT_COLOR = (255, 255, 0) # Yellow
SPOT_INVALID_COLOR = (150, 150, 150) # Gray
TEXT_COLOR = (0, 0, 0)               # Black
BACKGROUND_COLOR = (135, 206, 235)   # Sky Blue

# Global screen proportion (how much of the screen to use)
SCREEN_PROPORTION = 0.75

class CatanGame:
    def __init__(self, window_width=None, window_height=None):
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
        
        # Initialize game logic
        self.game_logic = GameLogic(self.board)
        self.players = self.game_logic.players  # Reference to players for easier access
        
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
            info_text = f"Spot {self.selected_spot} "
            valid_text = "(Valid)" if self.game_logic.is_valid_initial_settlement(self.selected_spot) else "(Invalid)"
            text_surface = self.font.render(info_text + valid_text, True, TEXT_COLOR)
            self.screen.blit(text_surface, (x_offset, y_offset))
        
        elif self.selected_road is not None:
            road = self.board.roads[self.selected_road]
            info_text = f"Road {self.selected_road}: {road.spot1_id}-{road.spot2_id}"
            valid_text = " (Valid)" if self.game_logic.is_valid_initial_road(self.selected_road, self.last_settlement_placed) else " (Invalid)"
            text_surface = self.font.render(info_text + valid_text, True, TEXT_COLOR)
            self.screen.blit(text_surface, (x_offset, y_offset))
    
    def run(self):
        """Main game loop"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click
                    mouse_pos = pygame.mouse.get_pos()
                    
                    # Don't process clicks on the player panel area (right 20% of screen)
                    if mouse_pos[0] < self.window_width * 0.8:
                        # In setup phase - handle settlement and road placement
                        if not self.game_logic.is_setup_complete():
                            if not self.game_logic.setup_phase_settlement_placed:
                                # We need to place a settlement
                                spot_id = self.check_spot_click(mouse_pos)
                                if spot_id is not None:
                                    self.selected_spot = spot_id
                                    self.selected_road = None
                                    print(f"Selected Spot {spot_id}: {self.board.spots[spot_id]}")
                            else:
                                # We need to place a road
                                road_id = self.check_road_click(mouse_pos)
                                if road_id is not None:
                                    self.selected_road = road_id
                                    self.selected_spot = None
                                    print(f"Selected Road {road_id}: {self.board.roads[road_id]}")
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
                
                elif event.type == pygame.KEYDOWN:
                    # Confirm placement with Enter/Return key
                    if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                        if not self.game_logic.is_setup_complete():
                            if not self.game_logic.setup_phase_settlement_placed and self.selected_spot is not None:
                                # Try to place settlement
                                if self.game_logic.place_initial_settlement(self.selected_spot):
                                    print(f"Placed settlement at spot {self.selected_spot}")
                                    # Remember this spot for road placement
                                    self.last_settlement_placed = self.selected_spot
                                    self.selected_spot = None
                            
                            elif self.game_logic.setup_phase_settlement_placed and self.selected_road is not None:
                                # Try to place road
                                if self.game_logic.place_initial_road(self.selected_road, self.last_settlement_placed):
                                    print(f"Placed road at {self.selected_road}")
                                    self.selected_road = None
                        
                        # In regular play phase (to be implemented)
            
            # Clear the screen
            self.screen.fill(BACKGROUND_COLOR)
            
            # Draw all elements
            self.draw_hexes()
            self.draw_roads()
            self.draw_spots()
            self.draw_player_status()  # Draw player status panel
            self.display_info_panel()
            
            # Update the display
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()

def main():
    # You can adjust the SCREEN_PROPORTION global variable at the top of the file
    # to change how much of the screen is used
    
    # Alternatively, you can specify an exact size:
    # game = CatanGame(1280, 720)
    
    game = CatanGame()  # Uses auto-detection with SCREEN_PROPORTION
    game.run()

if __name__ == "__main__":
    main()