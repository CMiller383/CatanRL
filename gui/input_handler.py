"""
Input handler for the Catan game GUI.
"""
import math

from game.action import Action
from game.enums import ActionType
from game.setup import place_initial_road, place_initial_settlement

class InputHandler:
    def __init__(self, game_logic, renderer, ui_handlers=None):
        self.game_logic = game_logic
        self.renderer = renderer
        self.ui_handlers = ui_handlers or {}
        
        # Game state
        self.last_settlement_placed = None
        self.robber_placement_active = False
    
    def check_spot_click(self, mouse_pos):
        """Check if any spot was clicked"""
        for spot_id, pos in self.renderer.spot_positions.items():
            distance = math.sqrt((mouse_pos[0] - pos[0])**2 + (mouse_pos[1] - pos[1])**2)
            if distance <= self.renderer.spot_radius:
                return spot_id
        return None
    
    def check_road_click(self, mouse_pos):
        """Check if any road was clicked using line distance calculation"""
        for road_id, road_info in self.renderer.road_positions.items():
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
            if distance <= self.renderer.road_width / 2:
                # Then check if it's within the segment (not just the infinite line)
                # Use a bounding box check with a little margin
                min_x = min(p1[0], p2[0]) - self.renderer.road_width
                max_x = max(p1[0], p2[0]) + self.renderer.road_width
                min_y = min(p1[1], p2[1]) - self.renderer.road_width
                max_y = max(p1[1], p2[1]) + self.renderer.road_width
                
                if (min_x <= mouse_pos[0] <= max_x and min_y <= mouse_pos[1] <= max_y):
                    return road_id
        
        return None
    
    def check_end_turn_button(self, mouse_pos):
        """Check if the end turn button was clicked"""
        if hasattr(self.renderer, 'end_turn_button') and self.renderer.end_turn_button.collidepoint(mouse_pos):
            return True
        return False
    
    def check_dice_click(self, mouse_pos):
        """Return True if the mouse click is inside the dice area."""
        if hasattr(self.renderer, 'dice_rect'):
            return self.renderer.dice_rect.collidepoint(mouse_pos)
        return False
    
    def check_hex_click(self, mouse_pos):
        """Check if a hex was clicked (for robber placement)"""
        if not (self.game_logic.state.awaiting_robber_placement or self.robber_placement_active):
            return None
            
        for hex_id, center in self.renderer.hex_centers.items():
            distance = math.sqrt((mouse_pos[0] - center[0])**2 + (mouse_pos[1] - center[1])**2)
            # Use a generous radius for hex click detection
            if distance <= self.renderer.number_circle_radius * 2:
                # Don't allow placing robber on same hex
                if hex_id != self.game_logic.state.robber_hex_id:
                    return hex_id
        return None
    
    def check_dev_card_button(self, mouse_pos):
        """Check if any dev card button was clicked"""
        if hasattr(self.renderer, 'check_dev_card_button'):
            return self.renderer.check_dev_card_button(mouse_pos)
        return None
    
    # Update the dev card handling section of the InputHandler class
    def handle_click(self, mouse_pos):
        """Process a mouse click at the given position"""
        # Don't process clicks on the player panel area (right 20% of screen)
        if mouse_pos[0] >= self.renderer.window_width * 0.8:
            return
            
        # Check if end turn button was clicked
        if self.check_end_turn_button(mouse_pos):
            self.game_logic.do_action(Action(ActionType.END_TURN))
            return

        # In placement phase - handle selection of two settlements and roads
        if not self.game_logic.is_setup_complete():
            if not self.game_logic.state.setup_phase_settlement_placed:
                # Attempt to place settlement immediately on click
                spot_id = self.check_spot_click(mouse_pos)
                if spot_id is not None:
                    successfully_placed = place_initial_settlement(self.game_logic.state, spot_id)
                    if successfully_placed:
                        print(f"Placed settlement at spot {spot_id}")
                        self.last_settlement_placed = spot_id
                    else:
                        print(f"Failed to place settlement at {spot_id}")
            else:
                # Settlement already placed; attempt to place road immediately
                road_id = self.check_road_click(mouse_pos)
                if road_id is not None:
                    successfully_placed = place_initial_road(self.game_logic.state, road_id, self.last_settlement_placed)
                    if successfully_placed:
                        print(f"Placed road at {road_id}")
                        self.last_settlement_placed = None
                    else:
                        print(f"Failed to place road at {road_id}")
            return
                
        # Regular play phase - handle special states first
        
        # Robber placement
        if self.game_logic.state.awaiting_robber_placement or self.robber_placement_active:
            hex_id = self.check_hex_click(mouse_pos)
            if hex_id is not None:
                success = self.game_logic.do_action(Action(ActionType.MOVE_ROBBER, hex_id))
                if success:
                    print(f"Moved robber to hex {hex_id}")
                    self.robber_placement_active = False
                    if hasattr(self.game_logic.state, 'potential_victims') and self.game_logic.state.potential_victims:
                        self.ui_handlers['steal'].setup_steal_from_player(self.game_logic.state.potential_victims)
                else:
                    print(f"Failed to move robber to hex {hex_id}")
            return
            
        # Steal from player
        if 'steal' in self.ui_handlers and self.ui_handlers['steal'].steal_buttons:
            victim_idx = self.ui_handlers['steal'].check_steal_button_click(mouse_pos)
            if victim_idx is not None:
                self.game_logic.do_action(Action(ActionType.STEAL, victim_idx))
                self.ui_handlers['steal'].clear_steal_selection()
                return
                
        # Resource selection for Year of Plenty or Monopoly
        if 'resource' in self.ui_handlers:
            resource_action = self.ui_handlers['resource'].check_resource_selection_click(mouse_pos)
            if resource_action:
                action_type, resource = resource_action
                if action_type == "year_of_plenty":
                    self.game_logic.do_action(Action(ActionType.SELECT_YEAR_OF_PLENTY_RESOURCE, resource))
                    print(f"Selected {resource.name} for Year of Plenty")
                elif action_type == "monopoly":
                    self.game_logic.select_monopoly_resource(resource)
                    self.game_logic.do_action(Action(ActionType.SELECT_MONOPOLY_RESOURCE, resource))
                    print(f"Selected {resource.name} for Monopoly")
                return
        
        # Check dev card buttons
        dev_card_action = self.check_dev_card_button(mouse_pos)
        if dev_card_action:
            if dev_card_action == "buy_dev_card":
                success = self.game_logic.do_action(Action(ActionType.BUY_DEV_CARD))
                if success:
                    print("Bought development card")
                else:
                    print("Failed to buy development card")
            elif dev_card_action == "play_knight":
                success = self.game_logic.do_action(Action(ActionType.PLAY_KNIGHT_CARD))
                if success:
                    print("Played Knight card")
                    self.robber_placement_active = True
                else:
                    print("Failed to play Knight card")
            elif dev_card_action == "play_road_building":
                success = self.game_logic.do_action(Action(ActionType.PLAY_ROAD_BUILDING_CARD))
                if success:
                    print("Played Road Building card")
                else:
                    print("Failed to play Road Building card")
            elif dev_card_action == "play_year_of_plenty":
                success = self.game_logic.do_action(Action(ActionType.PLAY_YEAR_OF_PLENTY_CARD))
                if success:
                    print("Played Year of Plenty card")
                else:
                    print("Failed to play Year of Plenty card")
            elif dev_card_action == "play_monopoly":
                success = self.game_logic.do_action(Action(ActionType.PLAY_MONOPOLY_CARD))
                if success:
                    print("Played Monopoly card")
                else:
                    print("Failed to play Monopoly card")
            return
        
        # Roll dice
        if self.check_dice_click(mouse_pos):
            self.game_logic.do_action(Action(ActionType.ROLL_DICE))
            return
        
        # Regular build actions
        spot_id = self.check_spot_click(mouse_pos)
        if spot_id is not None:
            if self.game_logic.do_action(Action(ActionType.UPGRADE_TO_CITY, spot_id)):
                print(f"Upgraded to city at spot {spot_id}")
            elif self.game_logic.do_action(Action(ActionType.BUILD_SETTLEMENT, spot_id)):
                print(f"Built settlement at spot {spot_id}")
            else:
                print(f"No valid action available at spot {spot_id}")
        
        # Build road
        road_id = self.check_road_click(mouse_pos)
        if road_id is not None:
            if self.game_logic.do_action(Action(ActionType.PLACE_FREE_ROAD, road_id)):
                print(f"Placed free road at {road_id}")
            elif self.game_logic.do_action(Action(ActionType.BUILD_ROAD, road_id)):
                print(f"Built road at {road_id}")
            else:
                print(f"Failed to build road at {road_id}")