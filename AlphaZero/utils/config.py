import torch
DEFAULT_CONFIG = {
    # Network parameters
    'state_dim': 992,
    'action_dim': 200,
    'hidden_dim': 256,
    
    # Training parameters
    'learning_rate': 0.001,
    'num_iterations': 50,
    'self_play_games': 20,
    'eval_games': 10,
    'epochs': 10,
    'batch_size': 256,
    'buffer_size': 200000,
    
    # MCTS parameters
    'num_simulations': 100,
    'c_puct': 1.5,
    'mcts_batch_size': 32,  # Size of batches for MCTS network evaluation
    
    # Game parameters
    'max_moves': 200,
    'device': 'cpu',
    
    # Paths
    'model_dir': 'models'
}

def get_config():
    """Return a copy of the default config that can be modified"""
    import copy
    return copy.deepcopy(DEFAULT_CONFIG)