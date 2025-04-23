import torch
from AlphaZero.core.network import DeepCatanNetwork
from AlphaZero.model.state_encoder import StateEncoder
from AlphaZero.model.action_mapper import ActionMapper
from AlphaZero.core.mcts import MCTS
from AlphaZero.agent.alpha_agent import AlphaZeroAgent


def load_alphazero_agent(player_id, model_path="models/best_model.pt", config=None):
    """
    Load a trained AlphaZero agent from a saved model file
    
    Args:
        player_id (int): The player ID for the agent
        model_path (str): Path to the saved model file
        config (dict, optional): Override configuration
        
    Returns:
        AlphaZeroAgent: The loaded agent with the trained model
    """
    try:
        # Set device based on config or availability
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if config and 'device' in config:
            device = config['device']
            
        # Load the checkpoint with proper device mapping
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get configuration from checkpoint
        checkpoint_config = checkpoint.get('config', {})
        
        # Merge configurations in priority order: passed config > checkpoint config > defaults
        merged_config = {}
        
        # Start with checkpoint config
        if checkpoint_config:
            merged_config.update(checkpoint_config)
        
        # Override with provided config if any
        if config:
            merged_config.update(config)
            
        # Extract required parameters with fallbacks
        state_dim = merged_config.get('state_dim', 992)
        action_dim = merged_config.get('action_dim', 200)
        hidden_dim = merged_config.get('hidden_dim', 256)
        
        # MCTS parameters
        num_simulations = merged_config.get('num_simulations', 100)
        c_puct = merged_config.get('c_puct', 1.5)
        batch_size = merged_config.get('mcts_batch_size', 8)
        
        # Noise parameters - explicitly set defaults to avoid unwanted exploration
        noise_eps = merged_config.get('noise_eps', 0.0)  # Default to 0 for evaluation
        noise_alpha = merged_config.get('noise_alpha', 0.3)
        
        # Create network and load the trained weights
        network = DeepCatanNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
        network.load_state_dict(checkpoint['network_state_dict'])
        network.eval()  # Set to evaluation mode
        network.to(device)  # Move to appropriate device
        
        # Create state encoder and action mapper
        state_encoder = StateEncoder(max_actions=action_dim)
        action_mapper = ActionMapper(max_actions=action_dim)
        
        # Create MCTS with the trained network and all parameters
        mcts = MCTS(
            network=network, 
            state_encoder=state_encoder, 
            action_mapper=action_mapper, 
            num_simulations=num_simulations,
            c_puct=c_puct,
            batch_size=batch_size,
            noise_eps=noise_eps,
            noise_alpha=noise_alpha
        )
        
        # Create and return the agent
        agent = AlphaZeroAgent(
            player_id=player_id, 
            network=network, 
            state_encoder=state_encoder, 
            action_mapper=action_mapper, 
            mcts=mcts
        )
        
        # Configure agent behavior
        agent.set_training_mode(False)  # Evaluation mode
        
        # Set deterministic mode if specified
        if merged_config.get('deterministic', True):
            agent.deterministic = True
            
        # Set temperature
        if 'temperature' in merged_config:
            agent.temperature = merged_config['temperature']
            
        # Enable debug if requested
        if merged_config.get('debug', False):
            agent.debug = True
            print(f"Loaded AlphaZero agent from {model_path} with:")
            print(f"  - num_simulations: {num_simulations}")
            print(f"  - c_puct: {c_puct}")
            print(f"  - noise_eps: {noise_eps}")
            print(f"  - on device: {device}")
        
        return agent
    
    except Exception as e:
        print(f"Error loading AlphaZero agent: {e}")
        # Fallback to creating a new untrained agent
        print("Creating a new untrained AlphaZero agent as fallback")
        from AlphaZero.agent.alpha_agent import create_alpha_agent
        return create_alpha_agent(player_id)