
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
    def __init__(self, config=None):
        """
        Initialize the training pipeline
        Args:
            config: Configuration dictionary
        """
        # Configuration
        if config is None:
            from AlphaZero.utils.config import get_config
            self.config = get_config()
        else:
            self.config = config

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
        self.current_iteration = 0

        # Prepare log and plot directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('plots', exist_ok=True)

        # Open log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = open(f"logs/training_log_{timestamp}.txt", 'w')
        self.log(f"AlphaZero Catan Training started at {timestamp}")
        self.log(f"Configuration: {self.config}")

        # Initialize components
        self._init_components()

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
            
            # Create a subplot with 3 rows:
            # 1 - Training Losses, 2 - Performance Metrics, 3 - Game Length Metrics
            fig = make_subplots(
                rows=3, 
                cols=1,
                subplot_titles=(
                    'AlphaZero Training Losses', 
                    'Performance Metrics', 
                    'Game Length Metrics'
                ),
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
            # fig.add_trace(
            #     go.Scatter(
            #         x=self.training_metrics['iteration'], 
            #         y=self.training_metrics['total_moves'],
            #         mode='lines+markers',
            #         name='Total Moves'
            #     ),
            #     row=3, col=1
            # )
            
            # Update layout
            fig.update_layout(
                height=900,
                width=800,
                title_text="AlphaZero Catan Training Metrics",
                showlegend=True
            )
            
            # Add axis labels
            fig.update_xaxes(title_text="Iteration", row=3, col=1)
            fig.update_yaxes(title_text="Loss", row=1, col=1)
            fig.update_yaxes(title_text="Performance", row=2, col=1)
            fig.update_yaxes(title_text="Moves", row=3, col=1)
            
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

    def _save_eval_results(self, results, iteration):
        """Save evaluation results to a file"""
        import json
        path = os.path.join(self.config['model_dir'], f"eval_{iteration}.json")
        with open(path, 'w') as f:
            json.dump(results, f, indent=4)


    def update_metrics(self, iteration, train_metrics, eval_metrics):
        self.training_metrics['iteration'].append(iteration)
        self.training_metrics['total_loss'].append(train_metrics['total_loss'])
        self.training_metrics['policy_loss'].append(train_metrics['policy_loss'])
        self.training_metrics['value_loss'].append(train_metrics['value_loss'])
        self.training_metrics['win_rate'].append(eval_metrics['win_rate'])
        self.training_metrics['avg_vp'].append(eval_metrics['avg_vp'])
        self.training_metrics['avg_game_length'].append(eval_metrics['avg_game_length'])
        self.training_metrics['total_moves'].append(eval_metrics.get('total_moves', 0))
