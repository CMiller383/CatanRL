from game.action import Action
from game.player_actions import *
from game.possible_action_generator import get_possible_actions
from game.setup import place_initial_road, place_initial_settlement
from .enums import ActionType, GamePhase
from game.game_state import GameState
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
    
    def is_current_player_alpha_zero(self):
        """Check if current player is AlphaZero"""
        return self.get_current_agent().is_AlphaZero()
    
    def is_setup_complete(self):
        """Check if the setup phase is complete"""
        return self.state.current_phase == GamePhase.REGULAR_PLAY
    
    def user_can_end_turn(self):
        return Action(ActionType.END_TURN) in self.state.possible_actions and self.is_current_player_human()
    
    def do_action(self, action):
        """Execute a game move"""
        state = self.state
        # print(state.current_player_idx)
        # print(action)
        # print(state.possible_actions)
        

        if state.current_phase != GamePhase.REGULAR_PLAY or action not in state.possible_actions:
            return False
        
        match action.type:
            case ActionType.ROLL_DICE:
                success = roll_dice(state)
            case ActionType.END_TURN:
                success = end_turn(state)
            case ActionType.BUY_DEV_CARD:
                success = buy_development_card(state)
            case ActionType.PLAY_KNIGHT_CARD:
                success = play_knight_card(state)
            case ActionType.PLAY_ROAD_BUILDING_CARD:
                success = play_road_building_card(state)
            case ActionType.PLAY_YEAR_OF_PLENTY_CARD:
                success = play_year_of_plenty_card(state)
            case ActionType.PLAY_MONOPOLY_CARD:
                success = play_monopoly_card(state)
            case ActionType.BUILD_SETTLEMENT:
                success = build_settlement(state, action.payload)
            case ActionType.UPGRADE_TO_CITY:
                success = upgrade_to_city(state, action.payload)
            case ActionType.BUILD_ROAD:
                success = place_road(state, action.payload)
            case ActionType.PLACE_FREE_ROAD:
                success = place_free_road(state, action.payload)
            case ActionType.SELECT_YEAR_OF_PLENTY_RESOURCE:
                success = select_year_of_plenty_resource(state, action.payload)
            case ActionType.SELECT_MONOPOLY_RESOURCE:
                success = select_monopoly_resource(state, action.payload)
            case ActionType.MOVE_ROBBER:
                success = move_robber(state, action.payload)
            case ActionType.STEAL:
                success = steal_resource_from_player(state, action.payload)
            case ActionType.TRADE_RESOURCES:
                success = trade_resources(state, action.payload)
            case _:
                success = False

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
        # if self.is_current_player_alpha_zero():
        #     print("AlphaZero possible actions:", state.possible_actions)
        #     print("AlphaZero action:", action)
        while action.type != ActionType.END_TURN:
            self.do_action(action)
            action = agent.get_action(state)

        self.do_action(action)

    # def create_agent(player_idx, agent_type):
    #     if agent_type == AgentType.HUMAN:
    #         return None
    #     elif agent_type == AgentType.RANDOM:
    #         from agent.random_agent import RandomAgent
    #         return RandomAgent(player_idx)
    #     elif agent_type == AgentType.HEURISTIC:
    #         from agent.simple_heuristic_agent import SimpleHeuristicAgent
    #         return SimpleHeuristicAgent(player_idx)
    #     else:
    #         print("Unsupported agent type")