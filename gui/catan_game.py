"""
Main game class for Catan.
"""
import pygame
from gui.constants import BACKGROUND_COLOR, SCREEN_PROPORTION
from gui.renderer import Renderer
from gui.input_handler import InputHandler
from gui.ui_components import ResourceSelectionHandler, StealSelectionHandler
from game.board import Board
from game.game_logic import GameLogic, check_game_over

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
        pygame.display.set_caption("Settlers of Catan")
        self.clock = pygame.time.Clock()
        
        # Initialize game components
        self.board = Board()
        self.game_logic = GameLogic(self.board, agent_types)
        self.players = self.game_logic.state.players
        
        # Font sizes scaled based on resolution
        base_font_size = int(min(window_width, window_height) / 50)
        self.fonts = {
            'font': pygame.font.SysFont('Arial', base_font_size),
            'number_font': pygame.font.SysFont('Arial', int(base_font_size * 1.5), bold=True),
            'info_font': pygame.font.SysFont('Arial', int(base_font_size * 1.2)),
            'instruction_font': pygame.font.SysFont('Arial', int(base_font_size * 1.8), bold=True),
            'card_font': pygame.font.SysFont('Arial', int(base_font_size * 1.1), bold=True),
            'card_desc_font': pygame.font.SysFont('Arial', int(base_font_size * 0.9))
        }
        
        # Initialize renderer
        self.renderer = Renderer(self.screen, window_width, window_height, self.board, self.game_logic)
                
        self.resource_selection_handler = ResourceSelectionHandler(
            self.game_logic,
            self.screen,
            window_width,
            window_height,
            self.fonts
        )
        
        self.steal_selection_handler = StealSelectionHandler(
            self.game_logic,
            self.screen,
            window_width,
            window_height,
            self.fonts,
            self.players
        )
        
        # Initialize input handler with UI components
        self.input_handler = InputHandler(
            self.game_logic, 
            self.renderer,
            {
                'resource': self.resource_selection_handler,
                'steal': self.steal_selection_handler
            }
        )
        
        # AI thinking timer
        self.ai_thinking_timer = 0

    def display_winner(self):
        winner = self.game_logic.state.winner
        msg = f"Player {winner} wins!"
        font = self.fonts['instruction_font']
        text_surf = font.render(msg, True, (255, 255, 255))
        # center it
        x = (self.window_width  - text_surf.get_width())  / 2
        y = (self.window_height - text_surf.get_height()) / 2
        self.screen.fill(BACKGROUND_COLOR)
        self.screen.blit(text_surf, (x, y))
        pygame.display.flip()
        pygame.time.delay(3000)   # pause 3 seconds

    def start_screen(self):
        msg = "Press any key to start"
        font = self.fonts['instruction_font']
        text_surf = font.render(msg, True, (255, 255, 255))
        x = (self.window_width  - text_surf.get_width())  / 2
        y = (self.window_height - text_surf.get_height()) / 2

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN, pygame.QUIT):
                    waiting = False
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
            self.screen.fill(BACKGROUND_COLOR)
            self.screen.blit(text_surf, (x, y))
            pygame.display.flip()
            self.clock.tick(30)

    def run(self):
        """Main game loop"""
        self.start_screen()
        running = True
        while running:
            # Process events for human players
            if self.game_logic.is_current_player_human():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click
                        mouse_pos = pygame.mouse.get_pos()
                        self.input_handler.handle_click(mouse_pos)
            else:
                # Handle AI player's turn with a slight delay
                self.ai_thinking_timer += 1
                
                # Process any quit events even during AI turns
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                
                # Add a small delay to make AI turns visible (every 30 frames = 0.5 seconds at 60fps)
                if self.ai_thinking_timer >= 10:
                    self.ai_thinking_timer = 0
                    
                    # Let the AI make its move
                    self.game_logic.process_ai_turn()
            if check_game_over(self.game_logic.state):
                self.display_winner()
                break
            # Clear the screen
            self.screen.fill(BACKGROUND_COLOR)
            
            # Draw all elements
            self.renderer.draw_board_border()
            self.renderer.draw_hexes()
            self.renderer.draw_roads(self.input_handler.last_settlement_placed)
            self.renderer.draw_spots()
            self.renderer.draw_player_status()
            self.renderer.draw_dice()
            self.renderer.draw_end_turn_button()
            self.renderer.draw_robber_placement(self.input_handler.robber_placement_active)
            self.renderer.draw_dev_card_menu()  # Draw dev card menu
            
            # Draw UI components
            self.resource_selection_handler.draw_resource_selection()
            self.steal_selection_handler.draw_steal_selection()
            
            # Update the display
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()