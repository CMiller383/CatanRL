import torch
from AlphaZero.core.network import CatanNetwork
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
        # Load the checkpoint
        checkpoint = torch.load(model_path)
        
        # Get configuration from checkpoint
        checkpoint_config = checkpoint.get('config', {})
        
        # Use provided config or checkpoint config or defaults
        if config is None:
            config = checkpoint_config
            
        state_dim = config.get('state_dim', 992)
        action_dim = config.get('action_dim', 200)
        hidden_dim = config.get('hidden_dim', 256)
        num_simulations = config.get('num_simulations', 100)
        
        # Create network and load the trained weights
        network = CatanNetwork(state_dim, action_dim, hidden_dim)
        network.load_state_dict(checkpoint['network_state_dict'])
        network.eval()  # Set to evaluation mode
        
        # Create state encoder and action mapper
        state_encoder = StateEncoder(max_actions=action_dim)
        action_mapper = ActionMapper(max_actions=action_dim)
        
        # Create MCTS with the trained network
        mcts = MCTS(network, state_encoder, action_mapper, num_simulations=num_simulations)
        
        # Create and return the agent
        agent = AlphaZeroAgent(player_id, network, state_encoder, action_mapper, mcts)
        agent.set_training_mode(False)  # Make sure training mode is off
        
        print(f"Successfully loaded AlphaZero agent from {model_path}")
        return agent
    
    except Exception as e:
        print(f"Error loading AlphaZero agent: {e}")
        # Fallback to creating a new untrained agent
        print("Creating a new untrained AlphaZero agent as fallback")
        from AlphaZero.agent.alpha_agent import create_alpha_agent
        return create_alpha_agent(player_id)