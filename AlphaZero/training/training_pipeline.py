import time 
import os
from AlphaZero.training.self_play import SelfPlayWorker
from AlphaZero.training.network_trainer import NetworkTrainer
from AlphaZero.training.evaluator import Evaluator
import torch

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
        if config is None:
            from AlphaZero.utils.config import get_config
            self.config = get_config()
        else:
            self.config = config
        # Current iteration
        self.current_iteration = 0

        # Create log directory
        os.makedirs('logs', exist_ok=True)
        os.makedirs('plots', exist_ok=True)

        # Create log file with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = open(f"logs/training_log_{timestamp}.txt", 'w')
        self.log(f"AlphaZero Catan Training started at {timestamp}")
        self.log(f"Configuration: {self.config}")

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
            game = GameLogic(board, agent_types=[AgentType.ALPHAZERO] + [AgentType.RANDOM] * 3)
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
        
    def train(self, num_iterations=None, resume_from=None):
        """
        Run the training pipeline for a number of iterations
        
        Args:
            num_iterations: Number of training iterations
            resume_from: Path to checkpoint to resume from (optional)
        """
        if num_iterations is None:
            num_iterations = self.config['num_iterations']
        
        # Load checkpoint if specified
        if resume_from:
            success = self.load_model(resume_from)
            if not success:
                self.log("Starting training from scratch")
            else:
                self.log(f"Resuming training from iteration {self.current_iteration}")
        
        # Main training loop
        start_time = time.time()
        for iteration in range(self.current_iteration, self.current_iteration + num_iterations):
            self.log(f"\n=== Iteration {iteration+1}/{self.current_iteration + num_iterations} ===")
            iteration_start = time.time()
            
            # Step 1: Self-play to generate data
            self.log("Starting self-play...")
            self_play_start = time.time()
            game_data = self.self_play_worker.generate_games(self.config['self_play_games'])
            self.trainer.add_game_data(game_data)
            self_play_time = time.time() - self_play_start
            self.log(f"Self-play completed in {self_play_time:.2f}s")
            self.log(f"Generated {len(game_data)} training examples")
            
            # Step 2: Train network
            self.log("Training network...")
            train_start = time.time()
            loss_metrics = self.trainer.train(
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size']
            )
            train_time = time.time() - train_start

            # Extract training metrics directly from the returned dictionary
            train_metrics = {
                'total_loss': loss_metrics.get('total_loss', 0),
                'value_loss': loss_metrics.get('value_loss', 0),
                'policy_loss': loss_metrics.get('policy_loss', 0),
            }
            
            eval_metrics = {'win_rate': 0, 'avg_vp': 0, 'avg_game_length': 0}
            #update to every 2 or so iterations leave comment so I can change for testing
            if (iteration + 1) % 2 == 0:
                self.log("Evaluating network...")
                eval_start = time.time()
                eval_metrics = self.evaluator.evaluate(self.config['eval_games'])
                eval_time = time.time() - eval_start
                self._save_eval_results(eval_metrics, iteration + 1)
                self.log(f"Evaluation completed in {eval_time:.2f}s")


            # self.log("Evaluating network...")
            # eval_start = time.time()
            # eval_metrics = self.evaluator.evaluate(self.config['eval_games'])
            # eval_time = time.time() - eval_start
            
            # self._save_eval_results(eval_metrics, iteration + 1)
            # self.log(f"Evaluation completed in {eval_time:.2f}s")
            
            # Step 4: Update and save metrics
            self.update_metrics(iteration + 1, train_metrics, eval_metrics)
            if (iteration + 1) % 5 == 0:
                self.plot_metrics()
            
            # Step 5: Save model
            is_best = False
            if eval_metrics['win_rate'] >= 0.75:  # If win rate is good
                is_best = True
                self.log(f"New best model with win rate {eval_metrics['win_rate']:.2f}!")
            
            if (iteration + 1) % 5 == 0 or is_best:  # Save every 5 iterations or if best
                self.save_model(iteration + 1, is_best)
            
            
            # Log iteration summary
            iteration_time = time.time() - iteration_start
            self.log(f"Iteration {iteration+1} completed in {iteration_time:.2f}s")
            
            # Update current iteration
            self.current_iteration = iteration + 1
        
        # Training completed
        total_time = time.time() - start_time
        self.log(f"\n=== Training Completed ===")
        self.log(f"Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
        
        # Save final model and plots
        self.save_model(self.current_iteration)
        self.plot_metrics()
        # Close log file
        self.log_file.close()
    
    def save_model(self, iteration, is_best=False):
        """Save a model checkpoint"""
        checkpoint_name = f"model_iter_{iteration}.pt"
        checkpoint_path = os.path.join(self.config['model_dir'], checkpoint_name)
        
        # Save model, optimizer, and configuration
        torch.save({
            'iteration': iteration,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': self.training_metrics
        }, checkpoint_path)
        
        self.log(f"Checkpoint saved to {checkpoint_path}")
        
        # If this is the best model, create a copy
        if is_best:
            best_path = os.path.join(self.config['model_dir'], "best_model.pt")
            torch.save({
                'iteration': iteration,
                'network_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'metrics': self.training_metrics
            }, best_path)
            self.log(f"Best model saved to {best_path}")
    
    def load_model(self, path):
        """Load a model checkpoint"""
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
                    x=self.training_metrics['iteration'], 
                    y=self.training_metrics['win_rate'],
                    mode='lines+markers',
                    name='Win Rate'
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=self.training_metrics['iteration'], 
                    y=[vp/10 for vp in self.training_metrics['avg_vp']],
                    mode='lines+markers',
                    name='Avg VP / 10'
                ),
                row=2, col=1
            )
            
            # Subplot 3: Game Length Metrics
            fig.add_trace(
                go.Scatter(
                    x=self.training_metrics['iteration'], 
                    y=self.training_metrics['avg_game_length'],
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

    def log(self, message):
        """Write a message to the log file and print it"""
        print(message)
        self.log_file.write(f"{message}\n")
        self.log_file.flush()

    def update_metrics(self, iteration, train_metrics, eval_metrics):
        self.training_metrics['iteration'].append(iteration)
        self.training_metrics['total_loss'].append(train_metrics['total_loss'])
        self.training_metrics['policy_loss'].append(train_metrics['policy_loss'])
        self.training_metrics['value_loss'].append(train_metrics['value_loss'])
        self.training_metrics['win_rate'].append(eval_metrics['win_rate'])
        self.training_metrics['avg_vp'].append(eval_metrics['avg_vp'])
        self.training_metrics['avg_game_length'].append(eval_metrics['avg_game_length'])
        self.training_metrics['total_moves'].append(eval_metrics.get('total_moves', 0))
