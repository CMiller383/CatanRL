"""
AlphaZero Catan Test Script
Tests each component of the AlphaZero implementation and runs a minimal training iteration.
"""
import os
import sys
import torch
import random
import numpy as np
from collections import deque

# Add the parent directory to sys.path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_random_seeds()

# Load configuration from the config utility
from AlphaZero.utils.config import get_config
test_config = get_config()

# Override config for testing
test_config.update({
    'num_simulations': 10,  # Small number for testing
    'self_play_games': 1,   # Just one game for testing
    'eval_games': 1,
    'epochs': 1,
    'batch_size': 16,
    'max_moves': 50,        # Limit moves for testing
    'mcts_batch_size': 4,   # Smaller batch size for testing
})

print("=== AlphaZero Catan Component Tests ===")

# Test 1: Neural Network
print("\n--- Test 1: Neural Network ---")
try:
    from AlphaZero.core.network import CatanNetwork
    
    # Create network
    network = CatanNetwork(
        state_dim=test_config['state_dim'],
        action_dim=test_config['action_dim'],
        hidden_dim=test_config['hidden_dim']
    )
    
    # Test forward pass
    test_input = torch.rand(1, test_config['state_dim'])
    policy_logits, value = network(test_input)
    
    print(f"Network created successfully")
    print(f"Input shape: {test_input.shape}")
    print(f"Policy output shape: {policy_logits.shape}")
    print(f"Value output shape: {value.shape}")
    print("Network test passed")
except Exception as e:
    print(f"Network test failed: {e}")
    raise

# Test 2: State Encoder
print("\n--- Test 2: State Encoder ---")
try:
    from AlphaZero.model.state_encoder import StateEncoder
    from game.board import Board
    from game.game_state import GameState
    
    # Create a board and game state
    board = Board()
    state = GameState(board)
    
    # Create encoder
    encoder = StateEncoder(max_actions=test_config['action_dim'])
    
    # Encode state
    encoded_state = encoder.encode_state(state)
    
    print(f"State encoder created successfully")
    print(f"Encoded state shape: {encoded_state.shape}")
    print(f"Expected state dimension: {test_config['state_dim']}")
    print("State encoder test passed")
except Exception as e:
    print(f"State encoder test failed: {e}")
    raise

# Test 3: Action Mapper
print("\n--- Test 3: Action Mapper ---")
try:
    from AlphaZero.model.action_mapper import ActionMapper
    from game.action import Action
    from game.enums import ActionType
    
    # Create action mapper
    action_mapper = ActionMapper(max_actions=test_config['action_dim'])
    
    # Test mapping for a few actions
    actions_to_test = [
        Action(ActionType.ROLL_DICE),
        Action(ActionType.END_TURN),
        Action(ActionType.BUILD_SETTLEMENT, payload=1),
        Action(ActionType.BUILD_ROAD, payload=5),
        Action(ActionType.MOVE_ROBBER, payload=3)
    ]
    
    print("Testing action mapping:")
    for action in actions_to_test:
        index = action_mapper.action_to_index(action)
        decoded_action = action_mapper.index_to_action(index)
        print(f"  {action.type.name} -> index {index} -> {decoded_action.type.name}")
    
    print("Action mapper test passed")
except Exception as e:
    print(f"Action mapper test failed: {e}")
    raise

# Test 4: MCTS Node and Tree
print("\n--- Test 4: MCTS ---")
try:
    from AlphaZero.core.mcts import MCTS, MCTSNode
    
    # Create test game state
    board = Board()
    state = GameState(board)
    
    # Create root node
    root = MCTSNode(game_state=state)
    
    # Initialize MCTS with the optimized version, including batch processing
    mcts = MCTS(
        network=network, 
        state_encoder=encoder, 
        action_mapper=action_mapper, 
        num_simulations=test_config['num_simulations'],
        batch_size=test_config['mcts_batch_size']
    )
    
    print("MCTS components initialized successfully")
    print("Trying a test search...")
    action_probs, value = mcts.search(state)
    print(f"Search returned {len(action_probs)} action probabilities and value {value}")
    print("MCTS test passed")
except Exception as e:
    print(f"MCTS test failed: {e}")
    raise

# Test 5: AlphaZero Agent
print("\n--- Test 5: AlphaZero Agent ---")
try:
    from AlphaZero.agent.alpha_agent import AlphaZeroAgent, create_alpha_agent
    
    # Create agent using the new config-based approach
    agent = create_alpha_agent(
        player_id=0,
        config=test_config,
        network=network  # Share the network we already created
    )
    
    print(f"Agent created successfully: {type(agent)}")
    print("AlphaZero agent test passed")
except Exception as e:
    print(f"AlphaZero agent test failed: {e}")
    raise

# Test 6: Game Logic Integration
print("\n--- Test 6: Game Logic Integration ---")
try:
    from game.game_logic import GameLogic
    from agent.base import AgentType
    
    # Create a game with our agent
    board = Board()
    game = GameLogic(board, agent_types=[AgentType.RANDOM] * 4)
    
    # Replace first agent with our AlphaZero agent
    game.agents[0] = agent
    
    # Run through setup phase
    print("Running setup phase...")
    while not game.is_setup_complete():
        game.process_ai_turn()
    
    print("Game setup completed successfully")
    print("Game logic integration test passed")
except Exception as e:
    print(f"Game logic integration test failed: {e}")
    raise

# Test 7: Self-Play Single Game
print("\n--- Test 7: Self-Play Single Game ---")
try:
    from AlphaZero.training.self_play import SelfPlayWorker
    
    # Create game factory
    def create_game():
        board = Board()
        return GameLogic(board, agent_types=[AgentType.ALPHAZERO] * 4)
    
    # Create agent factory that uses the shared network and config
    def create_test_agent(player_id):
        return create_alpha_agent(
            player_id=player_id,
            config=test_config,
            network=network  # All agents share the same network in self-play
        )
    
    # Create self-play worker
    self_play_worker = SelfPlayWorker(create_game, create_test_agent, test_config)
    
    print("Starting self-play test game...")
    game_data = self_play_worker.generate_games(1)
    
    print(f"Self-play game completed")
    print(f"Generated {len(game_data)} training examples")
    print("Self-play test passed")
except Exception as e:
    print(f"Self-play test failed: {e}")
    raise

# Test 8: Minimal Training
print("\n--- Test 8: Minimal Training ---")
try:
    from AlphaZero.training.network_trainer import NetworkTrainer
    
    # Create optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=test_config['learning_rate'])
    
    # Create trainer
    trainer = NetworkTrainer(network, optimizer, test_config)
    
    # Add game data to buffer
    if 'game_data' in locals() and game_data:
        trainer.add_game_data(game_data)
        
        # Run a single epoch
        print("Training for one epoch...")
        losses = trainer.train(epochs=1, batch_size=test_config['batch_size'])
        
        print("Training completed successfully")
        print(f"Training losses: {losses}")
        print("Training test passed")
    else:
        print("No game data available for training, skipping")
except Exception as e:
    print(f"Training test failed: {e}")
    raise

# Test 9: Full Pipeline (Mini)
print("\n--- Test 9: Mini Pipeline ---")
try:
    from AlphaZero.training.training_pipeline import TrainingPipeline

    # Create mini pipeline
    mini_config = test_config.copy()
    mini_config['num_iterations'] = 1
    
    print("Creating training pipeline...")
    pipeline = TrainingPipeline(mini_config)
    
    print("Running one iteration...")
    original_generate_games = pipeline.self_play_worker.generate_games
    
    # Create new method for faster testing
    def generate_one_game(n):
        return original_generate_games(1)

    pipeline.self_play_worker.generate_games = generate_one_game

    # Now run one iteration
    pipeline.train(1, testing=True)
    
    print("Pipeline test passed")
except Exception as e:
    print(f"Pipeline test failed: {e}")
    raise

print("\n=== All Tests Completed ===")