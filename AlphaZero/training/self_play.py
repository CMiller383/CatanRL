from tqdm import tqdm
from game.game_state import check_game_over
import torch 
from multiprocessing import Pool, cpu_count, get_context
from AlphaZero.core.network import DeepCatanNetwork
from AlphaZero.agent.alpha_agent import create_alpha_agent
import functools

def play_one_game_entry(args):
    """
    Worker entry point. args = (game_creator, agent_creator, config, game_idx)
    """
    game_idx, game_creator, agent_creator, config = args

    # pin torch to CPU threads
    torch.set_num_threads(1)

    # 1) build the game and agents
    game = game_creator()
    for pid in range(len(game.agents)):
        agent = agent_creator(pid)
        agent.set_training_mode(True)
        # enforce CPU
        agent.network = agent.network.cpu()
        agent.mcts.network = agent.network
        game.agents[pid] = agent

    # 2) setup phase
    while not game.is_setup_complete():
        game.process_ai_turn()

    # 3) main loop
    move_count = 0
    max_moves = config.get('max_moves', 200)
    while not check_game_over(game.state) and move_count < max_moves:
        move_count += 1
        game.process_ai_turn()

    # 4) collect data
    all_data = []
    for pid, agent in enumerate(game.agents):
        reward = calculate_reward(game.state, pid)
        agent.record_game_result(reward)
        all_data.extend(agent.get_game_history())

    return all_data

class SelfPlayWorker:
    """
    Worker that generates self-play games for training
    """
    def __init__(self, game_creator, agent_creator, config, pool=6):
        """
        Initialize the self-play worker
        
        Args:
            game_creator: Function that creates a new game instance
            agent_creator: Function that creates an agent
            config: Configuration dictionary
        """
        self.game_creator = game_creator
        self.agent_creator = agent_creator
        self.config = config
        self.pool = pool

    def _play_single_game(self, _):
        # 1) Force PyTorch to use only the CPU in this worker
        torch.set_num_threads(1)

        # 2) Build a fresh game & agents
        game = self.game_creator()
        for pid in range(len(game.agents)):
            agent = self.agent_creator(pid)
            agent.set_training_mode(True)

            # 3) Immediately move this agent’s net to CPU
            agent.network = agent.network.cpu()
            # also update the MCTS object to use that CPU net
            agent.mcts.network = agent.network

            game.agents[pid] = agent

        # 4) Play through the setup phase
        while not game.is_setup_complete():
            game.process_ai_turn()

        # 5) Main game loop
        move_count = 0
        max_moves = self.config.get('max_moves', 200)
        while not self._is_game_over(game.state) and move_count < max_moves:
            move_count += 1
            game.process_ai_turn()

        # 6) Collect all the history & rewards
        all_data = []
        for pid, agent in enumerate(game.agents):
            reward = self._calculate_reward(game.state, player_id=pid)
            agent.record_game_result(reward)
            all_data.extend(agent.get_game_history())

        return all_data


    def generate_games(self, num_games):
        # Build a small args tuple list
        args_list = [
            (i, self.game_creator, self.agent_creator, self.config)
            for i in range(num_games)
        ]

        results = []
        for game_data in tqdm(
            self.pool.imap_unordered(play_one_game_entry, args_list, chunksize=1),
            total=num_games,
            desc="Self-play games"
        ):
            results.append(game_data)
            # print(game_data)

        # Flatten and return
        return [step for game_data in results for step in game_data]

    def _is_game_over(self, game_state):
        """Check if the game is over"""
        if game_state.winner is not None:
            return True
        return check_game_over(game_state)
    
    def _get_winner(self, game_state):
        """Get the winner's player index"""
        if game_state.winner is not None:
            return game_state.winner
        max_vp = -1
        winner = None
        
        for i, player in enumerate(game_state.players):
            vp = player.victory_points
            if vp > max_vp:
                max_vp = vp
                winner = i
        
        return winner
    
    def _calculate_reward(self, game_state, player_id):
        """
        Calculate a more sophisticated reward for the AlphaZero agent
        
        Args:
            game_state: The game state
            player_id: The player ID for which to calculate the reward
            
        Returns:
            reward: The calculated reward
        """
        # Get the player and their victory points
        player = game_state.players[player_id]
        player_vp = player.victory_points
        
        # Get the winner and maximum VP
        winner = self._get_winner(game_state)
        max_vp = game_state.players[winner].victory_points
        
        # Base reward component: win/loss
        if winner == player_id:
            base_reward = 1.0  # Win
        else:
            base_reward = -1.0  # Loss
        
        # VP-based component: reward based on VP difference from winner
        # This helps distinguish between close losses and big losses
        vp_diff = player_vp - max_vp
        vp_component = vp_diff / 10.0  # Normalize by max VP (10)
        
        # Progress component: reward progress toward victory (settlements, cities, etc.)
        progress_score = 0.0
        
        # Add 0.1 for each settlement
        progress_score += len(player.settlements) * 0.1
        
        # Add 0.2 for each city
        progress_score += len(player.cities) * 0.2
        
        # Add 0.05 for each road
        progress_score += len(player.roads) * 0.05
        
        # Add 0.1 for each development card
        progress_score += len(player.dev_cards) * 0.1
        
        # Normalize progress score
        progress_component = progress_score / 5.0  # Max possible around 5
        
        # Resource diversity component: reward having diverse resources
        resource_count = sum(1 for r, count in player.resources.items() if count > 0)
        diversity_component = resource_count / 5.0  # 5 resource types
        
        # Combine components with weights
        reward = (
            0.6 * base_reward +      # Win/loss is the most important
            0.2 * vp_component +     # VP difference matters
            0.15 * progress_component + # Progress toward victory
            0.05 * diversity_component  # Resource diversity
        )
        
        return reward

def calculate_reward(state, player_id):
    player = state.players[player_id]
    player_vp = player.victory_points

    # Determine the winner index robustly
    if state.winner is not None:
        winner_idx = state.winner
    else:
        # pick the player with the most VP
        winner_idx = max(
            range(len(state.players)),
            key=lambda i: state.players[i].victory_points
        )
        state.winner = winner_idx  # optional: record it

    max_vp = state.players[winner_idx].victory_points

    # Base reward
    base_reward = 1.0 if winner_idx == player_id else -1.0

    # VP‑difference component
    vp_diff = player_vp - max_vp
    vp_component = vp_diff / 10.0

    # Progress, diversity, etc. as before…
    progress_score = (len(player.settlements) * 0.1 +
                      len(player.cities)     * 0.2 +
                      len(player.roads)      * 0.05 +
                      len(player.dev_cards)  * 0.1)
    progress_component = progress_score / 5.0

    diversity = sum(1 for r,c in player.resources.items() if c>0) / 5.0

    return (0.6 * base_reward +
            0.2 * vp_component +
            0.15 * progress_component +
            0.05 * diversity)