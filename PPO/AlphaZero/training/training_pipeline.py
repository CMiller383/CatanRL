import time
import os
from datetime import datetime
import torch
import psutil
from AlphaZero.training.self_play import SelfPlayWorker
from AlphaZero.training.network_trainer import NetworkTrainer
from AlphaZero.training.evaluator import Evaluator
from AlphaZero.core.network import DeepCatanNetwork
from game.board import Board
from game.game_logic import GameLogic
from agent.base import AgentType
from AlphaZero.agent.alpha_agent import create_alpha_agent
from functools import partial
import functools
from multiprocessing import get_context, cpu_count
import multiprocessing
from copy import deepcopy
import pickle
from collections import defaultdict
from game.game_state import GameState

# For placement network training - these imports will only be used if placement_training is enabled
import numpy as np
import random
from game.enums import GamePhase, ActionType
from game.setup import is_valid_initial_settlement
from game.action import Action

#for pickle reasons
def top_level_create_game():
    board = Board()
    return GameLogic(
        board,
        agent_types=[AgentType.ALPHAZERO] + [AgentType.RANDOM] * 3
    )

def top_level_create_agent(player_id, config, network, device):
    agent = create_alpha_agent(
        player_id=player_id,
        config=config,
        network=network
    )
    agent.network.to(device)
    return agent

class TrainingPipeline:
    """
    Coordinates the overall training process
    """
    def __init__(self, config=None, use_placement_network=False):
        """
        Initialize the training pipeline
        Args:
            config: Configuration dictionary
            use_placement_network: Whether to use and train the placement network
        """
        # Configuration
        if config is None:
            from AlphaZero.utils.config import get_config
            self.config = get_config()
        else:
            self.config = config
            
        # Set placement network flag
        self.config['use_placement_network'] = use_placement_network
        # Open log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = open(f"logs/training_log_{timestamp}.txt", 'w')
        self.log(f"AlphaZero Catan Training started at {timestamp}")
        self.log(f"Configuration: {self.config}")
        # Add placement network settings if needed
        if self.config.get('train_placement_network', True):
            if 'placement_epochs' not in self.config:
                self.config['placement_epochs'] = 10
            if 'placement_batch_size' not in self.config:
                self.config['placement_batch_size'] = 32
            if 'placement_lr' not in self.config:
                self.config['placement_lr'] = 0.001
            if 'placement_hidden_dim' not in self.config:
                self.config['placement_hidden_dim'] = 128
            if 'placement_train_frequency' not in self.config:
                self.config['placement_train_frequency'] = 5
                
            # Initialize placement network components
            self._init_placement_components()

        # Determine device
        self.device = torch.device('cpu')

        # Metrics storage
        self.training_metrics = {
            'iteration': [],
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'win_rate': [],
            'avg_vp': [],
            'avg_game_length': [],
            'total_moves': []
        }
        
        # Add placement metrics if enabled
        if self.config.get('train_placement_network', True):
            self.placement_metrics = {
                'iteration': [],
                'loss': [],
                'accuracy': [],
                'examples_collected': 0
            }
            
        self.current_iteration = 0

        # Prepare log and plot directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('plots', exist_ok=True)


        
        if self.config.get('train_placement_network', True):
            self.log("Initial placement network training is ENABLED")
            self.log(f"Placement network settings:")
            self.log(f"  - Training epochs: {self.config['placement_epochs']}")
            self.log(f"  - Batch size: {self.config['placement_batch_size']}")
            self.log(f"  - Learning rate: {self.config['placement_lr']}")
            self.log(f"  - Hidden dimensions: {self.config['placement_hidden_dim']}")
            self.log(f"  - Training frequency: Every {self.config['placement_train_frequency']} iterations")

        # Initialize components
        self._init_components()

    def _init_placement_components(self):
        """Initialize the placement network components"""
        try:
            from AlphaZero.core.initial_placement_network import InitialPlacementNetwork, InitialPlacementEncoder, InitialPlacementTrainer
            
            # Check for existing model
            placement_path = os.path.join(self.config['model_dir'], 'placement_network.pt')
            if os.path.exists(placement_path):
                try:
                    self.log(f"Loading initial placement network from {placement_path}")
                    checkpoint = torch.load(placement_path, map_location=torch.device('cpu'))
                    
                    # Get state dict
                    state_dict = checkpoint.get('network_state_dict')
                    
                    # Create network - we don't know the exact dimensions but can infer them
                    input_dim = next(iter(state_dict.values())).shape[1] if 'fc1.weight' in state_dict else 260
                    hidden_dim = next(iter(state_dict.values())).shape[0] if 'fc1.weight' in state_dict else 128
                    output_dim = next(iter(state_dict.values())).shape[0] if 'fc3.weight' in state_dict else 54
                    
                    self.placement_network = InitialPlacementNetwork(
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim
                    )
                    
                    # Load weights
                    self.placement_network.load_state_dict(state_dict)
                    
                    # Make available globally for agents
                    globals()['placement_network'] = self.placement_network
                    
                    self.log(f"Successfully loaded placement network with input_dim={input_dim}, hidden_dim={hidden_dim}")
                except Exception as e:
                    self.log(f"Error loading placement network: {e}, creating new one")
                    self._create_new_placement_network()
            else:
                self._create_new_placement_network()
                
            # Create encoder and trainer
            self.placement_encoder = InitialPlacementEncoder()
            self.placement_trainer = InitialPlacementTrainer(
                self.placement_network, 
                lr=self.config['placement_lr']
            )
            
            self.log("Placement network components initialized successfully")
        except Exception as e:
            self.log(f"Error initializing placement network components: {e}")
            self.config['use_placement_network'] = False
            
    def _create_new_placement_network(self):
        """Create a new placement network"""
        from AlphaZero.core.initial_placement_network import InitialPlacementNetwork
        
        # Create network with config parameters
        self.placement_network = InitialPlacementNetwork(
            input_dim=260,  # Default value
            hidden_dim=self.config.get('placement_hidden_dim', 128),
            output_dim=54  # 54 settlement spots
        )
        
        # Make available globally for agents
        # globals()['placement_network'] = self.placement_network
        
        self.log(f"Created new placement network with hidden_dim={self.config.get('placement_hidden_dim', 128)}")

    def log(self, message):
        """
        Log a message to console and file with timestamp, then flush
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{ts}] {message}"
        print(entry)
        self.log_file.write(entry + "\n")
        self.log_file.flush()

    def _init_components(self):
        # 1) Build and move your GPU network
        self.network = DeepCatanNetwork(
            state_dim  = self.config['state_dim'],
            action_dim = self.config['action_dim'],
            hidden_dim = self.config['hidden_dim']
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config['learning_rate']
        )

        # 2) Make a CPU-only copy for the workers
        cpu_net = deepcopy(self.network).cpu()

        # 3) Build a picklable agent_creator partial
        agent_creator = partial(
            top_level_create_agent,
            config=self.config,
            network=cpu_net,
            device='cpu'
        )

        # 4) Fork a Pool *once*
        pool_size = min(cpu_count()-1, self.config['self_play_games'])
        self.self_play_pool = multiprocessing.Pool(processes=pool_size)

        # 5) Instantiate the worker with creators and the pool
        self.self_play_worker = SelfPlayWorker(
            game_creator  = top_level_create_game,
            agent_creator = agent_creator,
            config        = self.config,
            pool          = self.self_play_pool
        )

        # 6) Trainer & evaluator as you already had…
        self.trainer   = NetworkTrainer(self.network, self.optimizer, self.config)
        self.evaluator = Evaluator(
            game_creator  = top_level_create_game,
            agent_creator = partial(
                top_level_create_agent,
                config = self.config,
                network=self.network,
                device = self.device
            ),
            config = self.config,
            log_fn = self.log
        )
        os.makedirs(self.config['model_dir'], exist_ok=True)
    
    def extract_placement_data(self, game_data):
        """
        Extract initial placement examples from game data
        
        Args:
            game_data: Game data from self-play
            
        Returns:
            placement_data: List of initial placement examples
        """
        if not self.config.get('train_placement_network', True) or not game_data:
            return []
            
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
                                encoded_state = self.placement_encoder.encode_board(state)
                                valid_mask = self.placement_encoder.get_valid_placement_mask(state)
                                
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
        
        # Update metrics
        if hasattr(self, 'placement_metrics'):
            self.placement_metrics['examples_collected'] += len(placement_data)
        
        return placement_data
    
    def train_placement_network(self, placement_data, iteration):
        """
        Train the placement network on the extracted data
        
        Args:
            placement_data: List of placement examples
            iteration: Current training iteration
            
        Returns:
            metrics: Training metrics
        """
        if not self.config.get('train_placement_network', True) or not placement_data:
            return None
            
        # Check if we should train this iteration
        if iteration % self.config.get('placement_train_frequency', 5) != 0:
            self.log(f"Skipping placement network training for iteration {iteration} (training every {self.config.get('placement_train_frequency', 5)} iterations)")
            return None
            
        # Add data to trainer's buffer
        for example in placement_data:
            self.placement_trainer.data_buffer.append(example)
        
        # Check if we have enough data
        if len(self.placement_trainer.data_buffer) < self.config.get('placement_batch_size', 32):
            self.log(f"Not enough data for placement network training: {len(self.placement_trainer.data_buffer)} < {self.config.get('placement_batch_size', 32)}")
            return None
            
        # Train the network
        self.log(f"Training initial placement network on {len(self.placement_trainer.data_buffer)} examples")
        start_time = time.time()
        metrics = self.placement_trainer.train(
            epochs=self.config.get('placement_epochs', 10),
            batch_size=self.config.get('placement_batch_size', 32)
        )
        training_time = time.time() - start_time
        
        self.log(f"Placement network training completed in {training_time:.2f}s")
        self.log(f"Metrics: Loss = {metrics.get('loss', 0):.4f}, Accuracy = {metrics.get('accuracy', 0):.4f}")
        
        # Save the placement network
        self.save_placement_network()
        
        # Update metrics
        if hasattr(self, 'placement_metrics'):
            self.placement_metrics['iteration'].append(iteration)
            self.placement_metrics['loss'].append(metrics.get('loss', 0))
            self.placement_metrics['accuracy'].append(metrics.get('accuracy', 0))
        
        return metrics
    
    def save_placement_network(self):
        """Save the placement network to disk"""
        if not hasattr(self, 'placement_trainer') or not hasattr(self, 'placement_network'):
            return
            
        # Save model
        save_path = os.path.join(self.config['model_dir'], 'placement_network.pt')
        self.placement_trainer.save_model(save_path)
        self.log(f"Saved placement network to {save_path}")

    def train(self, num_iterations=None, resume_from=None, testing=False):
        """
        Run the training pipeline for a number of iterations
        """
        num_iterations = num_iterations or self.config['num_iterations']

        # Resume checkpoint if provided
        if resume_from:
            if not self.load_model(resume_from):
                self.log("Starting training from scratch")
            else:
                self.log(f"Resuming training from iteration {self.current_iteration}")

        start_time = time.time()
        try:
            for iteration in range(self.current_iteration, self.current_iteration + num_iterations):
                checkpoint_path = os.path.join(self.config['model_dir'], 'latest_cpu.pth')
                torch.save(self.network.cpu().state_dict(), checkpoint_path)
                self.network.to(self.device)  # move it back to GPU
                self.config['checkpoint_path'] = checkpoint_path

                self.log(f"\n=== Iteration {iteration+1}/{num_iterations} ===")
                iter_start = time.time()

                # 1. Self-play
                self.log("Starting self-play...")
                sp_start = time.time()
                game_data = self.self_play_worker.generate_games(self.config['self_play_games'])
                self.trainer.add_game_data(game_data)
                sp_time = time.time() - sp_start
                rate = len(game_data) / sp_time if sp_time > 0 else 0
                self.log(f"Self-play completed in {sp_time:.2f}s, generated {len(game_data)} examples ({rate:.1f} games/s)")

                # 1.5 (New) Train placement network if enabled
                if self.config.get('train_placement_network', True):
                    placement_data = self.extract_placement_data(game_data)
                    self.train_placement_network(placement_data, iteration+1)

                # 2. Training
                self.log("Training network...")
                tr_start = time.time()
                metrics = self.trainer.train(epochs=self.config['epochs'], batch_size=self.config['batch_size'])
                self.log(f"Training completed in {time.time() - tr_start:.2f}s")

                # 3. Evaluation every 2 iters
                eval_metrics = {'win_rate': 0, 'avg_vp': 0, 'avg_game_length': 0}
                if (iteration + 1) % 5 == 0:
                    self.network.eval()
                    self.log("Evaluating network...")
                    ev_start = time.time()
                    eval_metrics = self.evaluator.evaluate(self.config['eval_games'])
                    self.log(f"Evaluation completed in {time.time() - ev_start:.2f}s")
                    self.network.train()

                # 4. Metrics and plotting
                self.update_metrics(iteration+1, metrics, eval_metrics)
                if (iteration + 1) % 5 == 0:
                    self.plot_metrics()

                # 5. Checkpointing
                is_best = eval_metrics['win_rate'] >= 0.75
                if is_best:
                    self.log(f"New best model at iteration {iteration+1} (win_rate={eval_metrics['win_rate']:.2f})")
                if (iteration + 1) % 5 == 0 or is_best:
                    self.save_model(iteration+1, is_best)

                # Log iteration duration and resource usage
                iter_time = time.time() - iter_start
                self.log(f"Iteration {iteration+1} done in {iter_time:.2f}s")
                # System resource logging
                cpu = psutil.cpu_percent()
                ram = psutil.virtual_memory().percent
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.max_memory_allocated() / 1e9
                    torch.cuda.reset_peak_memory_stats()
                    self.log(f"Resource usage: CPU {cpu:.1f}%, RAM {ram:.1f}%, GPU peak memory {gpu_mem:.2f} GB")
                else:
                    self.log(f"Resource usage: CPU {cpu:.1f}%, RAM {ram:.1f}%")

                self.current_iteration = iteration + 1
                param_sum = sum(p.sum().item() for p in self.network.parameters())
                self.log(f"Network parameter sum after training: {param_sum:.6f}")

        except KeyboardInterrupt:
            self.log("Training interrupted by user; saving current model...")
            self.save_model(self.current_iteration)
        finally:
            total = time.time() - start_time
            self.log(f"\n=== Training Finished ({total:.2f}s / {total/3600:.2f}h) ===")
            if not testing:
                self.save_model(self.current_iteration)
                self.plot_metrics()
            self.log_file.close()
            if hasattr(self, 'self_play_pool'):
                self.self_play_pool.close()
                self.self_play_pool.join()

    def save_model(self, iteration, is_best=False):
      """Save a model checkpoint and replay buffer"""
      # Create the standard checkpoint
      ckpt = {
          'iteration': iteration,
          'network_state_dict': self.network.state_dict(),
          'optimizer_state_dict': self.optimizer.state_dict(),
          'config': self.config,
          'metrics': self.training_metrics
      }
      path = os.path.join(self.config['model_dir'], f"model_iter_{iteration}.pt")
      torch.save(ckpt, path)
      self.log(f"Checkpoint saved: {path}")
      
      if is_best:
          best_path = os.path.join(self.config['model_dir'], "best_model.pt")
          torch.save(ckpt, best_path)
          self.log(f"Best model saved: {best_path}")
          
      # Save placement network if enabled
      if self.config.get('train_placement_network', True):
          self.save_placement_network()
          
          # If best model, create a copy with iteration number
          if is_best:
              placement_path = os.path.join(self.config['model_dir'], 'placement_network.pt')
              if os.path.exists(placement_path):
                  import shutil
                  dst = os.path.join(self.config['model_dir'], f'placement_network_iter_{iteration}.pt')
                  try:
                      shutil.copy2(placement_path, dst)
                      self.log(f"Copied best placement network to {dst}")
                  except Exception as e:
                      self.log(f"Warning: Could not save placement network checkpoint: {e}")
      
      # Save only the latest replay buffer
      try:
          
          # Use a fixed path for the latest buffer
          buffer_path = os.path.join(self.config['model_dir'], "latest_buffer.pkl")
          
          # Also keep a backup of the previous buffer
          backup_path = os.path.join(self.config['model_dir'], "backup_buffer.pkl")
          
          # If a buffer already exists, make it the backup before saving the new one
          if os.path.exists(buffer_path):
              if os.path.exists(backup_path):
                  os.remove(backup_path)  # Remove old backup
              os.rename(buffer_path, backup_path)  # Current becomes backup
          
          # Save the current buffer
          buffer_data = self.trainer.data_buffer
          with open(buffer_path, 'wb') as f:
              pickle.dump(buffer_data, f)
          
          buffer_size_mb = os.path.getsize(buffer_path) / (1024 * 1024)
          self.log(f"Replay buffer saved: {buffer_path} ({len(buffer_data)} examples, {buffer_size_mb:.1f} MB)")
          
      except Exception as e:
          self.log(f"Warning: Failed to save replay buffer: {e}")

    def load_model(self, path):
      """Load a model checkpoint and replay buffer if available"""
      if not os.path.exists(path):
          self.log(f"Model file not found: {path}")
          return False
      
      try:
          checkpoint = torch.load(path)
          self.network.load_state_dict(checkpoint['network_state_dict'])
          self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
          
          # Restore metrics and configuration if available
          if 'metrics' in checkpoint:
              self.training_metrics = checkpoint['metrics']
          
          if 'config' in checkpoint:
              # Only update certain config parameters, not all
              for key in ['learning_rate', 'num_simulations', 'c_puct']:
                  if key in checkpoint['config']:
                      self.config[key] = checkpoint['config'][key]
                      
              # Preserve placement network settings if they exist
              if self.config.get('train_placement_network', True):
                  for key in ['placement_epochs', 'placement_batch_size', 'placement_lr', 
                             'placement_hidden_dim', 'placement_train_frequency']:
                      if key in self.config:
                          checkpoint['config'][key] = self.config[key]
                #   checkpoint['config']['use_placement_network'] = True
          
          # Update current iteration
          if 'iteration' in checkpoint:
              self.current_iteration = checkpoint['iteration']
          
          # Try to load buffer
          import pickle
          buffer_path = os.path.join(self.config['model_dir'], "latest_buffer.pkl")
          
          if os.path.exists(buffer_path):
              try:
                  with open(buffer_path, 'rb') as f:
                      buffer_data = pickle.load(f)
                  
                  # Replace the current replay buffer
                  self.trainer.data_buffer = buffer_data
                  buffer_size_mb = os.path.getsize(buffer_path) / (1024 * 1024)
                  self.log(f"Replay buffer loaded: {len(buffer_data)} examples, {buffer_size_mb:.1f} MB")
              except Exception as e:
                  self.log(f"Warning: Failed to load replay buffer: {e}")
                  
                  # Try backup if main buffer failed
                  backup_path = os.path.join(self.config['model_dir'], "backup_buffer.pkl")
                  if os.path.exists(backup_path):
                      try:
                          with open(backup_path, 'rb') as f:
                              buffer_data = pickle.load(f)
                          
                          self.trainer.data_buffer = buffer_data
                          buffer_size_mb = os.path.getsize(backup_path) / (1024 * 1024)
                          self.log(f"Backup buffer loaded: {len(buffer_data)} examples, {buffer_size_mb:.1f} MB")
                      except Exception as e2:
                          self.log(f"Warning: Also failed to load backup buffer: {e2}")
          
          self.log(f"Checkpoint loaded from {path}, resuming from iteration {self.current_iteration}")
          return True
      
      except Exception as e:
          self.log(f"Error loading checkpoint: {e}")
          return False

    def plot_metrics(self):
        """
        Plot training metrics using Plotly.
        """
        if not self.training_metrics['iteration']:
            self.log("No metrics to plot yet")
            return
        
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import os
            
            # Create plots directory
            os.makedirs('plots', exist_ok=True)
            
            # Determine number of rows based on whether placement metrics exist
            num_rows = 4 if hasattr(self, 'placement_metrics') and self.placement_metrics['iteration'] else 3
            
            # Create a subplot with rows:
            # 1 - Training Losses, 2 - Performance Metrics, 3 - Game Length Metrics, 4 - Placement Metrics (if enabled)
            subplot_titles = [
                'AlphaZero Training Losses', 
                'Performance Metrics', 
                'Game Length Metrics'
            ]
            
            if num_rows == 4:
                subplot_titles.append('Placement Network Metrics')
                
            fig = make_subplots(
                rows=num_rows, 
                cols=1,
                subplot_titles=subplot_titles,
                vertical_spacing=0.1
            )
            
            # Subplot 1: Training Losses
            fig.add_trace(
                go.Scatter(
                    x=self.training_metrics['iteration'], 
                    y=self.training_metrics['policy_loss'],
                    mode='lines+markers',
                    name='Policy Loss'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=self.training_metrics['iteration'], 
                    y=self.training_metrics['value_loss'],
                    mode='lines+markers',
                    name='Value Loss'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=self.training_metrics['iteration'], 
                    y=self.training_metrics['total_loss'],
                    mode='lines+markers',
                    name='Total Loss'
                ),
                row=1, col=1
            )
            
            # Subplot 2: Performance Metrics
            fig.add_trace(
                go.Scatter(
                    #plot only even iterations
                    x=[i for i in self.training_metrics['iteration'] if i % 2 == 0],
                    y=[wr for i, wr in zip(self.training_metrics['iteration'], self.training_metrics['win_rate']) if i % 2 == 0],
                    mode='lines+markers',
                    name='Win Rate'
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    #plot only even iterations
                    x=[i for i in self.training_metrics['iteration'] if i % 2 == 0],
                    y=[vp for i, vp in zip(self.training_metrics['iteration'], self.training_metrics['avg_vp']) if i % 2 == 0],
                    mode='lines+markers',
                    name='Avg VP / 10'
                ),
                row=2, col=1
            )
            
            # Subplot 3: Game Length Metrics
            fig.add_trace(
                go.Scatter(
                    #plot only even iterations
                    x=[i for i in self.training_metrics['iteration'] if i % 2 == 0],
                    y=[self.training_metrics['avg_game_length'][i] for i in range(len(self.training_metrics['iteration'])) if self.training_metrics['iteration'][i] % 2 == 0],
                    mode='lines+markers',
                    name='Avg Game Length (moves)'
                ),
                row=3, col=1
            )
            
            # Subplot 4: Placement Network Metrics (if enabled)
            if num_rows == 4:
                if hasattr(self, 'placement_metrics') and self.placement_metrics['iteration']:
                    fig.add_trace(
                        go.Scatter(
                            x=self.placement_metrics['iteration'],
                            y=self.placement_metrics['loss'],
                            mode='lines+markers',
                            name='Placement Loss'
                        ),
                        row=4, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=self.placement_metrics['iteration'],
                            y=self.placement_metrics['accuracy'],
                            mode='lines+markers',
                            name='Placement Accuracy'
                        ),
                        row=4, col=1
                    )
            
            # Update layout
            fig.update_layout(
                height=250 * num_rows,
                width=800,
                title_text="AlphaZero Catan Training Metrics",
                showlegend=True
            )
            
            # Add axis labels
            fig.update_xaxes(title_text="Iteration", row=num_rows, col=1)
            fig.update_yaxes(title_text="Loss", row=1, col=1)
            fig.update_yaxes(title_text="Performance", row=2, col=1)
            fig.update_yaxes(title_text="Moves", row=3, col=1)
            if num_rows == 4:
                fig.update_yaxes(title_text="Placement Metrics", row=4, col=1)
            
            # Save as HTML file
            fig.write_html("plots/training_metrics.html")
            
            # Also try saving as an image, if possible (requires kaleido)
            try:
                fig.write_image("plots/training_metrics.png")
            except Exception as e:
                self.log(f"Could not save image (requires kaleido): {e}")
            
            self.log("Plotly metrics visualization saved to plots/training_metrics.html")
            
        except ImportError as e:
            self.log(f"Could not import Plotly for visualization. Error: {e}")
            self.log("Run 'pip install plotly' to enable visualization.")

    def update_metrics(self, iteration, train_metrics, eval_metrics):
        self.training_metrics['iteration'].append(iteration)
        self.training_metrics['total_loss'].append(train_metrics['total_loss'])
        self.training_metrics['policy_loss'].append(train_metrics['policy_loss'])
        self.training_metrics['value_loss'].append(train_metrics['value_loss'])
        self.training_metrics['win_rate'].append(eval_metrics['win_rate'])
        self.training_metrics['avg_vp'].append(eval_metrics['avg_vp'])
        self.training_metrics['avg_game_length'].append(eval_metrics['avg_game_length'])
        self.training_metrics['total_moves'].append(eval_metrics.get('total_moves', 0))
