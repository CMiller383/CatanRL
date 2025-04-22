"""
Integrated training for the initial placement network within the main training pipeline.
"""
import os
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import time

from AlphaZero.model.initial_placement_network import (
    InitialPlacementNetwork, 
    InitialPlacementTrainer,
    InitialPlacementEncoder
)
from game.board import Board
from game.game_state import GameState
from game.enums import GamePhase, ActionType
from game.setup import is_valid_initial_settlement
from game.action import Action

class PlacementTrainingManager:
    """
    Manages the training of the initial placement network within the main AlphaZero pipeline.
    """
    def __init__(self, config, log_fn=None):
        """
        Initialize the placement training manager
        
        Args:
            config: Configuration dictionary
            log_fn: Function for logging messages
        """
        self.config = config
        self.log = log_fn or print
        
        # Create directories
        os.makedirs(config['model_dir'], exist_ok=True)
        
        # Initialize the network
        self.init_network()
        
        # Create encoder
        self.encoder = InitialPlacementEncoder()
        
        # Training params
        self.epochs = config.get('placement_epochs', 10)
        self.batch_size = config.get('placement_batch_size', 32)
        self.lr = config.get('placement_lr', 0.001)
        
        # Initialize trainer
        self.trainer = InitialPlacementTrainer(self.network, lr=self.lr)
        
        # Training frequency
        self.train_frequency = config.get('placement_train_frequency', 5)
        
        # Metrics
        self.metrics = {
            'loss': [],
            'accuracy': [],
            'examples_collected': 0
        }
    
    def init_network(self):
        """Initialize or load the initial placement network"""
        input_dim = 260  # Default value
        hidden_dim = self.config.get('placement_hidden_dim', 128)
        output_dim = 54  # 54 settlement spots
        
        # Check for existing model
        model_path = os.path.join(self.config['model_dir'], 'placement_network.pt')
        if os.path.exists(model_path):
            try:
                self.log(f"Loading initial placement network from {model_path}")
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                
                # Get state dict
                state_dict = checkpoint.get('network_state_dict')
                
                # Infer dimensions from state dict
                if state_dict and 'fc1.weight' in state_dict:
                    input_dim = state_dict['fc1.weight'].shape[1]
                    hidden_dim = state_dict['fc1.weight'].shape[0]
                
                # Create network
                self.network = InitialPlacementNetwork(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim
                )
                
                # Load weights
                self.network.load_state_dict(state_dict)
                
                # Set to eval mode
                self.network.eval()
                
                self.log(f"Successfully loaded placement network with input_dim={input_dim}, hidden_dim={hidden_dim}")
                return
            except Exception as e:
                self.log(f"Error loading placement network: {e}")
                self.log("Creating new placement network")
        
        # Create new network
        self.network = InitialPlacementNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        self.log(f"Created new placement network with input_dim={input_dim}, hidden_dim={hidden_dim}")
    
    def extract_placement_data(self, game_data):
        """
        Extract initial placement examples from game data
        
        Args:
            game_data: Game data from self-play
            
        Returns:
            placement_data: List of initial placement examples
        """
        # Group examples by game ID and player
        examples_by_game = defaultdict(list)
        
        for example in game_data:
            game_id = example.get('game_id', example.get('player', 0))
            examples_by_game[game_id].append(example)
        
        self.log(f"Processing {len(examples_by_game)} games for initial placement data")
        
        # Extracted placement examples
        placement_data = []
        
        # Process each game
        for game_id, examples in examples_by_game.items():
            # Initialize a fresh state for this game
            board = Board()
            state = GameState(board)
            
            # Find reward from the last example
            reward = None
            for example in reversed(examples):
                if example.get('reward') is not None:
                    reward = example.get('reward')
                    break
            
            if reward is None:
                continue  # Skip if no reward
            
            # Look for settlement placement actions in setup phase
            for example in examples:
                # Find actions related to initial placement
                for action, prob in example.get('action_probs', {}).items():
                    if isinstance(action, Action) and action.type == ActionType.BUILD_SETTLEMENT:
                        spot_id = action.payload
                        
                        # Check if this is a valid initial placement
                        if (state.current_phase in [GamePhase.SETUP_PHASE_1, GamePhase.SETUP_PHASE_2] and 
                            is_valid_initial_settlement(state, spot_id)):
                            
                            # Extract state features
                            try:
                                encoded_state = self.encoder.encode_board(state)
                                valid_mask = self.encoder.get_valid_placement_mask(state)
                                
                                # Create target (one-hot for the chosen spot)
                                target = np.zeros(54, dtype=np.float32)
                                target[spot_id-1] = 1.0  # -1 for 0-indexing
                                
                                # Add example
                                placement_data.append({
                                    'state': encoded_state,
                                    'target': target,
                                    'valid_mask': valid_mask,
                                    'reward': reward,
                                    'phase': state.current_phase.value,
                                    'spot_id': spot_id
                                })
                                
                                # Apply the action to update state
                                from game.setup import place_initial_settlement
                                success = place_initial_settlement(state, spot_id)
                                
                                # Only continue processing if we successfully placed
                                if not success:
                                    break
                                    
                                # Skip the road placement
                                if state.current_phase == GamePhase.SETUP_PHASE_1:
                                    # After first settlement, go to next player
                                    state.current_player_idx = (state.current_player_idx + 1) % 4
                                    if state.current_player_idx == 0:
                                        state.current_phase = GamePhase.SETUP_PHASE_2
                                        # Last player starts in Phase 2
                                        state.current_player_idx = 3
                                elif state.current_phase == GamePhase.SETUP_PHASE_2:
                                    # After second settlement, go to previous player
                                    state.current_player_idx = (state.current_player_idx - 1) % 4
                            except Exception as e:
                                # Just skip this example if something goes wrong
                                continue
        
        # Log the number of examples collected
        self.log(f"Extracted {len(placement_data)} initial placement examples")
        self.metrics['examples_collected'] += len(placement_data)
        
        return placement_data
    
    def should_train(self, iteration):
        """Determine if we should train the network this iteration"""
        # Train every n iterations
        return iteration % self.train_frequency == 0
    
    def process_game_data(self, game_data, iteration):
        """
        Process game data to extract and train on initial placement examples
        
        Args:
            game_data: Game data from self-play
            iteration: Current training iteration
            
        Returns:
            metrics: Training metrics if trained, None otherwise
        """
        # Extract placement data
        placement_data = self.extract_placement_data(game_data)
        
        # Add to trainer's buffer
        for example in placement_data:
            self.trainer.data_buffer.append(example)
        
        # Check if we should train
        if self.should_train(iteration) and len(self.trainer.data_buffer) >= self.batch_size:
            self.log(f"Training initial placement network on {len(self.trainer.data_buffer)} examples")
            
            # Train the network
            start_time = time.time()
            metrics = self.trainer.train(epochs=self.epochs, batch_size=self.batch_size)
            training_time = time.time() - start_time
            
            self.log(f"Placement network training completed in {training_time:.2f}s")
            self.log(f"Metrics: Loss = {metrics.get('loss', 0):.4f}, Accuracy = {metrics.get('accuracy', 0):.4f}")
            
            # Save the model
            self.save_network()
            
            # Update metrics
            self.metrics['loss'].append(metrics.get('loss', 0))
            self.metrics['accuracy'].append(metrics.get('accuracy', 0))
            
            return metrics
        
        return None
    
    def save_network(self):
        """Save the placement network to disk"""
        save_path = os.path.join(self.config['model_dir'], 'placement_network.pt')
        self.trainer.save_model(save_path)
        self.log(f"Saved placement network to {save_path}")
    
    def get_network(self):
        """Get the current placement network"""
        return self.network