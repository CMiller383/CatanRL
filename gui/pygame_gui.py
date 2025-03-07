import pygame
import math
from game.board import Board
from game.resource import Resource

# Define colors
RESOURCE_COLORS = {
    Resource.WOOD: (34, 139, 34),    # Forest Green
    Resource.BRICK: (178, 34, 34),   # Firebrick
    Resource.WHEAT: (218, 165, 32),  # Goldenrod
    Resource.SHEEP: (144, 238, 144), # Light Green
    Resource.ORE: (105, 105, 105),   # Dim Gray
    Resource.DESERT: (238, 232, 170) # Light Tan
}

ROAD_COLOR = (139, 69, 19)           # Brown
ROAD_HIGHLIGHT_COLOR = (255, 140, 0) # Dark Orange
SPOT_COLOR = (255, 255, 255)         # White
SPOT_HIGHLIGHT_COLOR = (255, 255, 0) # Yellow
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
        
        self.board = Board()
        
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
        
        # Selected elements
        self.selected_spot = None
        self.selected_road = None
    
    def compute_transform(self, margin=0.1):
        """Compute scale and offset to fit the board in the window"""
        # Get min and max coordinates of the board
        positions = [spot.position for spot in self.board.spots.values()]
        min_x = min(x for x, y in positions)
        max_x = max(x for x, y in positions)
        min_y = min(y for x, y in positions)
        max_y = max(y for x, y in positions)
        
        board_width = max_x - min_x
        board_height = max_y - min_y
        
        # Calculate scale to fit in window with margin
        scale_x = (self.window_width * (1 - margin)) / board_width
        scale_y = (self.window_height * (1 - margin)) / board_height
        scale = min(scale_x, scale_y)
        
        # Calculate offset to center the board
        offset_x = (self.window_width - board_width * scale) / 2 - min_x * scale
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
            color = ROAD_HIGHLIGHT_COLOR if road_id == self.selected_road else ROAD_COLOR
            pygame.draw.line(self.screen, color, start_pos, end_pos, self.road_width)
    
    def draw_spots(self):
        """Draw all spots (vertices)"""
        for spot_id, pos in self.spot_positions.items():
            color = SPOT_HIGHLIGHT_COLOR if spot_id == self.selected_spot else SPOT_COLOR
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
        """Display information panel for selected elements"""
        # Create a semi-transparent background for the info panel
        if self.selected_spot is not None or self.selected_road is not None:
            panel_height = 80 if (self.selected_spot is not None and self.selected_road is not None) else 40
            panel_surface = pygame.Surface((self.window_width, panel_height), pygame.SRCALPHA)
            panel_surface.fill((255, 255, 255, 200))  # Semi-transparent white
            self.screen.blit(panel_surface, (0, 0))
        
        y_offset = 10
        
        if self.selected_spot is not None:
            spot = self.board.spots[self.selected_spot]
            info_text = f"Spot {self.selected_spot}: Hexes {spot.adjacent_hex_ids}"
            text_surface = self.info_font.render(info_text, True, TEXT_COLOR)
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 30
        
        if self.selected_road is not None:
            road = self.board.roads[self.selected_road]
            info_text = f"Road {self.selected_road}: Spots {road.spot1_id}-{road.spot2_id}, Hexes {road.adjacent_hex_ids}"
            text_surface = self.info_font.render(info_text, True, TEXT_COLOR)
            self.screen.blit(text_surface, (10, y_offset))
    
    def run(self):
        """Main game loop"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    
                    # Check for spot clicks first (they're smaller)
                    spot_id = self.check_spot_click(mouse_pos)
                    if spot_id is not None:
                        self.selected_spot = spot_id
                        self.selected_road = None
                        print(f"Clicked Spot {spot_id}: {self.board.spots[spot_id]}")
                    else:
                        # If no spot was clicked, check for road clicks
                        road_id = self.check_road_click(mouse_pos)
                        if road_id is not None:
                            self.selected_road = road_id
                            self.selected_spot = None
                            print(f"Clicked Road {road_id}: {self.board.roads[road_id]}")
            
            # Clear the screen
            self.screen.fill(BACKGROUND_COLOR)
            
            # Draw all elements
            self.draw_hexes()
            self.draw_roads()
            self.draw_spots()
            self.display_info_panel()
            
            # Update the display
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()

def main():

    game = CatanGame()
    game.run()

if __name__ == "__main__":
    main()