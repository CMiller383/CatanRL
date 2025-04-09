"""
Training module for AlphaZero Catan.
Handles self-play, neural network training, and evaluation.
"""
import os
import time
import random
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import deque

class SelfPlayWorker:
    """
    Worker that generates self-play games for training
    """
    def __init__(self, game_creator, agent_creator, config):
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
    
    def generate_games(self, num_games):
        """
        Generate self-play games
        
        Args:
            num_games: Number of games to generate
            
        Returns:
            game_data: List of game data
        """
        all_game_data = []
        
        for game_idx in tqdm(range(num_games), desc="Self-play games"):
            # Create a new game and agent
            game = self.game_creator()
            agent = self.agent_creator(player_id=0)
            agent.set_training_mode(True)
            
            # Replace the first agent in the game
            game.agents[0] = agent
            
            # Handle the setup phase
            while not game.is_setup_complete():
                game.process_ai_turn()
            
            # Play the game
            move_count = 0
            max_moves = self.config.get('max_moves', 200)
            
            # Main game loop
            while not self._is_game_over(game.state) and move_count < max_moves:
                move_count += 1
                
                # If it's our agent's turn
                if game.state.current_player_idx == 0:
                    # Get action from agent
                    action = agent.get_action(game.state)
                    game.do_action(action)
                else:
                    # Other players' turns
                    game.process_ai_turn()
            
            # Game is over, calculate rewards
            reward = self._calculate_reward(game.state, player_id=0)
            
            # Record final reward in agent's game history
            agent.record_game_result(reward)
            
            # Get game data from agent
            game_data = agent.get_game_history()
            all_game_data.extend(game_data)
            
            # Log game results
            winner = self._get_winner(game.state)
            print(f"Game {game_idx+1}: Player {winner} won with "
                  f"{game.state.players[winner].get_victory_points()} VP "
                  f"(Our agent: {game.state.players[0].get_victory_points()} VP)")
        
        return all_game_data
    
    def _is_game_over(self, game_state):
        """Check if a game is over"""
        # Check if any player has 10+ victory points
        for player in game_state.players:
            if player.get_victory_points() >= 10:
                return True
        return False
    
    def _get_winner(self, game_state):
        """Get the winner's player index"""
        max_vp = -1
        winner = None
        
        for i, player in enumerate(game_state.players):
            vp = player.get_victory_points()
            if vp > max_vp:
                max_vp = vp
                winner = i
        
        return winner
    
    def _calculate_reward(self, game_state, player_id):
        """Calculate the reward for a player"""
        # Simple binary reward: 1 for win, -1 for loss
        winner = self._get_winner(game_state)
        return 1.0 if winner == player_id else -1.0


class NetworkTrainer:
    """
    Trainer for the neural network using self-play data
    """
    def __init__(self, network, optimizer, config):
        """
        Initialize the network trainer
        
        Args:
            network: Neural network to train
            optimizer: Optimizer for training
            config: Configuration dictionary
        """
        self.network = network
        self.optimizer = optimizer
        self.config = config
        
        # Training data buffer
        self.data_buffer = deque(maxlen=config.get('buffer_size', 100000))
    
    def add_game_data(self, game_data):
        """
        Add game data to the buffer
        
        Args:
            game_data: List of state, action probs, and reward tuples
        """
        self.data_buffer.extend(game_data)
    
    def train(self, epochs=None, batch_size=None):
        """
        Train the network on the current data buffer
        
        Args:
            epochs: Number of training epochs
            batch_size: Size of training batches
            
        Returns:
            losses: List of losses for each epoch
        """
        if epochs is None:
            epochs = self.config.get('epochs', 10)
        
        if batch_size is None:
            batch_size = self.config.get('batch_size', 128)
        
        if len(self.data_buffer) < batch_size:
            print(f"Not enough data for training: {len(self.data_buffer)} < {batch_size}")
            return []
        
        losses = []
        value_losses = []
        policy_losses = []
        
        for epoch in range(epochs):
            # Sample from buffer
            indices = np.random.choice(len(self.data_buffer), min(10000, len(self.data_buffer)), replace=False)
            samples = [self.data_buffer[i] for i in indices]
            
            # Train in batches
            epoch_loss = 0
            epoch_value_loss = 0
            epoch_policy_loss = 0
            batches = 0
            
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i+batch_size]
                if len(batch) < batch_size:
                    continue
                
                # Prepare batch data
                states = torch.stack([torch.FloatTensor(step['state']) for step in batch])
                action_probs = [step['action_probs'] for step in batch]
                rewards = torch.FloatTensor([step['reward'] for step in batch]).unsqueeze(1)
                
                # Convert action_probs dictionaries to tensors
                # This requires mapping from action objects to indices
                from AlphaZero.model.action_mapper import ActionMapper
                action_mapper = ActionMapper(self.config.get('action_dim', 200))
                
                policy_targets = torch.zeros(len(batch), self.config.get('action_dim', 200))
                for j, probs in enumerate(action_probs):
                    for action, prob in probs.items():
                        # Convert action to index
                        action_idx = action_mapper.action_to_index(action)
                        policy_targets[j, action_idx] = prob
                
                # Forward pass
                policy_logits, value = self.network(states)
                
                # Calculate losses
                # Value loss: MSE
                value_loss = F.mse_loss(value, rewards)
                
                # Policy loss: Cross entropy
                policy_loss = -torch.sum(policy_targets * F.log_softmax(policy_logits, dim=1)) / len(batch)
                
                # Combined loss (weighted)
                loss = value_loss + policy_loss
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Track losses
                epoch_loss += loss.item()
                epoch_value_loss += value_loss.item()
                epoch_policy_loss += policy_loss.item()
                batches += 1
            
            # Calculate average losses
            if batches > 0:
                avg_loss = epoch_loss / batches
                avg_value_loss = epoch_value_loss / batches
                avg_policy_loss = epoch_policy_loss / batches
            else:
                avg_loss = avg_value_loss = avg_policy_loss = 0
            
            losses.append(avg_loss)
            value_losses.append(avg_value_loss)
            policy_losses.append(avg_policy_loss)
            
            print(f"Epoch {epoch+1}/{epochs}: Loss: {avg_loss:.4f} "
                  f"(Value: {avg_value_loss:.4f}, Policy: {avg_policy_loss:.4f})")
        
        return losses


class Evaluator:
    """
    Evaluates the AlphaZero agent against baseline agents
    """
    def __init__(self, game_creator, agent_creator, config):
        """
        Initialize the evaluator
        
        Args:
            game_creator: Function that creates a new game instance
            agent_creator: Function that creates an agent
            config: Configuration dictionary
        """
        self.game_creator = game_creator
        self.agent_creator = agent_creator
        self.config = config
    
    def evaluate(self, num_games=None):
        """
        Evaluate the agent
        
        Args:
            num_games: Number of evaluation games
            
        Returns:
            results: Evaluation results
        """
        if num_games is None:
            num_games = self.config.get('eval_games', 20)
        
        # Results tracking
        wins = 0
        vp_total = 0
        game_lengths = []
        
        # Create evaluation agent
        agent = self.agent_creator(player_id=0)
        agent.set_training_mode(False)  # Turn off training mode for evaluation
        
        for game_idx in tqdm(range(num_games), desc="Evaluation games"):
            # Create a new game
            game = self.game_creator()
            game.agents[0] = agent
            
            # Handle the setup phase
            while not game.is_setup_complete():
                game.process_ai_turn()
            
            # Track game length
            moves = 0
            max_moves = self.config.get('max_moves', 200)
            
            # Main game loop
            while not self._is_game_over(game.state) and moves < max_moves:
                moves += 1
                
                if game.state.current_player_idx == 0:
                    # AlphaZero agent's turn
                    action = agent.get_action(game.state)
                    game.do_action(action)
                else:
                    # Other agents' turns
                    game.process_ai_turn()
            
            # Record results
            game_lengths.append(moves)
            vp_total += game.state.players[0].get_victory_points()
            
            # Check if our agent won
            winner = self._get_winner(game.state)
            if winner == 0:
                wins += 1
            
            print(f"Game {game_idx+1}: Player {winner} won with "
                  f"{game.state.players[winner].get_victory_points()} VP "
                  f"(Our agent: {game.state.players[0].get_victory_points()} VP)")
        
        # Calculate statistics
        win_rate = wins / num_games
        avg_vp = vp_total / num_games
        avg_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0
        
        results = {
            'win_rate': win_rate,
            'avg_vp': avg_vp,
            'avg_game_length': avg_length,
            'num_games': num_games
        }
        
        print(f"Evaluation results:")
        print(f"  Win rate: {win_rate:.2f}")
        print(f"  Average VP: {avg_vp:.2f}")
        print(f"  Average game length: {avg_length:.2f} moves")
        
        return results
    
    def _is_game_over(self, game_state):
        """Check if the game is over"""
        for player in game_state.players:
            if player.get_victory_points() >= 10:
                return True
        return False
    
    def _get_winner(self, game_state):
        """Get the winner's player index"""
        max_vp = -1
        winner = None
        
        for i, player in enumerate(game_state.players):
            vp = player.get_victory_points()
            if vp > max_vp:
                max_vp = vp
                winner = i
        
        return winner


class TrainingPipeline:
    """
    Coordinates the overall training process
    """
    def __init__(self, config=None):
        """
        Initialize the training pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = {
            # Default configuration
            'state_dim': 1000,
            'action_dim': 200,
            'hidden_dim': 256,
            'learning_rate': 0.001,
            'num_iterations': 50,
            'self_play_games': 20,
            'eval_games': 10,
            'epochs': 10,
            'batch_size': 128,
            'buffer_size': 100000,
            'max_moves': 200,
            'model_dir': 'models',
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Create components
        self._init_components()
    
    def _init_components(self):
        """Initialize network, optimizer, and workers"""
        from AlphaZero.core.network import CatanNetwork
        from game.board import Board
        from game.game_logic import GameLogic
        from agent.base import AgentType
        
        # Create network
        self.network = CatanNetwork(
            state_dim=self.config['state_dim'],
            action_dim=self.config['action_dim'],
            hidden_dim=self.config['hidden_dim']
        )
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config['learning_rate']
        )
        
        # Game creator function
        def create_game():
            board = Board()
            game = GameLogic(board, agent_types=[AgentType.HEURISTIC] + [AgentType.RANDOM] * 3)
            return game
        
        # Agent creator function
        def create_agent(player_id):
            from AlphaZero.agent.alpha_agent import create_alpha_agent
            return create_alpha_agent(
                player_id=player_id,
                state_dim=self.config['state_dim'],
                action_dim=self.config['action_dim'],
                hidden_dim=self.config['hidden_dim']
            )
        
        # Create workers
        self.self_play_worker = SelfPlayWorker(create_game, create_agent, self.config)
        self.trainer = NetworkTrainer(self.network, self.optimizer, self.config)
        self.evaluator = Evaluator(create_game, create_agent, self.config)
        
        # Create model directory
        os.makedirs(self.config['model_dir'], exist_ok=True)
    
    def train(self, num_iterations=None):
        """
        Run the training pipeline for a number of iterations
        
        Args:
            num_iterations: Number of training iterations
        """
        if num_iterations is None:
            num_iterations = self.config['num_iterations']
        
        for iteration in range(num_iterations):
            print(f"=== Iteration {iteration+1}/{num_iterations} ===")
            
            # Step 1: Self-play to generate data
            start_time = time.time()
            game_data = self.self_play_worker.generate_games(self.config['self_play_games'])
            self.trainer.add_game_data(game_data)
            print(f"Self-play completed in {time.time() - start_time:.2f}s")
            print(f"Generated {len(game_data)} training examples")
            
            # Step 2: Train network
            start_time = time.time()
            losses = self.trainer.train(
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size']
            )
            print(f"Training completed in {time.time() - start_time:.2f}s")
            
            # Step 3: Evaluate
            if (iteration + 1) % 5 == 0:  # Evaluate every 5 iterations
                start_time = time.time()
                results = self.evaluator.evaluate(self.config['eval_games'])
                print(f"Evaluation completed in {time.time() - start_time:.2f}s")
                
                # Save evaluation results
                self._save_eval_results(results, iteration + 1)
            
            # Step 4: Save model
            if (iteration + 1) % 10 == 0:  # Save every 10 iterations
                self.save_model(iteration + 1)
    
    def save_model(self, iteration):
        """Save the model"""
        path = os.path.join(self.config['model_dir'], f"model_{iteration}.pt")
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'iteration': iteration
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a saved model"""
        if not os.path.exists(path):
            print(f"Model file not found: {path}")
            return False
            
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Optionally load config
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
            
        print(f"Model loaded from {path}")
        return True
    
    def _save_eval_results(self, results, iteration):
        """Save evaluation results to a file"""
        import json
        path = os.path.join(self.config['model_dir'], f"eval_{iteration}.json")
        with open(path, 'w') as f:
            json.dump(results, f, indent=4)