from game.player_actions import build_settlement, buy_development_card, end_turn, move_robber, place_free_road, place_road, play_knight_card, play_monopoly_card, play_road_building_card, play_year_of_plenty_card, roll_dice, select_monopoly_resource, select_year_of_plenty_resource, upgrade_to_city
from game.possible_action_generator import get_possible_actions
from game.setup import place_initial_road, place_initial_settlement
from .enums import DevCardType, GamePhase
from game.game_state import GameState
from .spot import SettlementType
from .enums import Resource
from .player import Player


from agent.base import AgentType, create_agent

class GameLogic:
    def __init__(self, board, agent_types=None):
        self.state = GameState(board)
        self.agents = []
        
        # Default all agents to human if not specified
        if agent_types is None:
            agent_types = [AgentType.HUMAN] * 4
        
        for i in range(4):
            agent_type = agent_types[i]
            player = Player(i, f"Player {i} {agent_type}")
            player.is_human = agent_type == AgentType.HUMAN
            self.state.players.append(player)
            self.agents.append(create_agent(i, agent_type))

    def get_current_agent(self):
        """Returns the current agent object"""
        return self.agents[self.state.current_player_idx]
    
    def is_current_player_human(self):
        """Check if current player is human"""
        return self.get_current_agent().is_human()
    
    def is_setup_complete(self):
        """Check if the setup phase is complete"""
        return self.state.current_phase == GamePhase.REGULAR_PLAY
    
    def user_can_end_turn(self):
        return "end_turn" in self.state.possible_actions and self.is_current_player_human()
    
    def do_action(self, move):
        """Execute a game move"""
        state = self.state

        print("doing a move")
        print(state.possible_actions)
        print(move)

        if state.current_phase != GamePhase.REGULAR_PLAY or move not in state.possible_actions:
            return False
        
        if not isinstance(move, tuple):
            # Handle string moves
            if move == "roll_dice":
                success =  roll_dice(state)
            elif move == "end_turn":
                success = end_turn(state)
            elif move == "buy_dev_card":
                success =  buy_development_card(state)
            elif move == "play_knight":
                success = play_knight_card(state)
            elif move == "play_road_building":
                success = play_road_building_card(state)
            elif move == "play_year_of_plenty":
                success = play_year_of_plenty_card(state)
            elif move == "play_monopoly":
                success = play_monopoly_card(state)
        else:
            # Handle tuple moves
            action, data = move
            if action == "build_settlement":
                success = build_settlement(state, data)
            elif action == "upgrade_city":
                success = upgrade_to_city(state, data)
            elif action == "road":
                success = place_road(state, data)
            elif action == "free_road":
                success = place_free_road(state, data)
            elif action == "select_resource":
                success = select_year_of_plenty_resource(state, data)
            elif action == "select_monopoly":
                success = select_monopoly_resource(state, data)
            elif action == "move_robber":
                success = move_robber(state, data)
        
        if success:
            state.possible_actions = get_possible_actions(state)

        return success
    

    def process_ai_turn(self):
        """Process a turn for an AI player"""
        state = self.state

        if self.is_current_player_human():
            # Not an AI player, do nothing
            state.waiting_for_human_input = True
            return False
        
        # Get the current agent
        agent = self.get_current_agent()
        
        # Handle the setup phase
        if not self.is_setup_complete():
            if not state.setup_phase_settlement_placed:
                spot_id = agent.get_initial_settlement(state)
                state.last_settlement_placed = spot_id
                return place_initial_settlement(state, spot_id)
            else:
                road_id = agent.get_initial_road(state, state.last_settlement_placed)
                place_initial_road(state, road_id, state.last_settlement_placed)
                state.last_settlement_placed = None
                return True
        
        action = agent.get_action(state)
        while action != "end_turn":
            self.do_action(action)
            action = agent.get_action(state)

        self.do_action("end_turn")

    def create_agent(player_idx, agent_type):
        if agent_type == AgentType.HUMAN:
            return None
        elif agent_type == AgentType.RANDOM:
            from agent.random_agent import RandomAgent
            return RandomAgent(player_idx)
        elif agent_type == AgentType.HEURISTIC:
            from agent.simple_heuristic_agent import SimpleHeuristicAgent
            return SimpleHeuristicAgent(player_idx)
        else:
            print("Unsupported agent type")