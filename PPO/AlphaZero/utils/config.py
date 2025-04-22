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
    'buffer_size': 50000,
    
    # MCTS parameters
    'num_simulations': 100,
    'c_puct': 1.5,
    'mcts_batch_size': 12,  # Size of batches for MCTS network evaluation
    # MCTS Dirichlet noise
    'noise_eps': 0.25,       # fraction of noise vs. learned prior
    'noise_alpha': 0.3,      # concentration parameter for Dirichlet
    
    # Game parameters
    'max_moves': 200,
    'device': 'cpu',

    #placement
    'placement_epochs': 10,           # Number of training epochs per update
    'placement_batch_size': 32,       # Batch size for training
    'placement_lr': 0.001,            # Learning rate
    'placement_hidden_dim': 128,      # Hidden dimension size
    'placement_train_frequency': 5,   # Train every N iterations
    'train_placement_network': True,  # Train placement, might as well leave on
    'use_placement_network': False,  # Use placement network for training (DO NOT TURN ON) I REPEAT DO NOT TURN ON
    # Paths
    'model_dir': 'models'
}

def get_config():
    """Return a copy of the default config that can be modified"""
    import copy
    return copy.deepcopy(DEFAULT_CONFIG)