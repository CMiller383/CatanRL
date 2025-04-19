import torch
from AlphaZero.core.network import DeepCatanNetwork
from AlphaZero.utils.config import get_config

# Load model and check basic properties
model_path = "models/model_iter_38.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(model_path, map_location=device)

# Print basic information
print(f"Model iteration: {checkpoint.get('iteration', 'unknown')}")
print(f"Config: {checkpoint.get('config', {})}")

# Check network values
network = DeepCatanNetwork(992, 200, 256)
network.load_state_dict(checkpoint['network_state_dict'])
network.eval()

# Create a dummy input
dummy_input = torch.rand(1, 992)
policy, value = network(dummy_input)

# Check statistics of network outputs
print(f"Policy output: min={policy.min().item()}, max={policy.max().item()}, mean={policy.mean().item()}")
print(f"Value output: {value.item()}")

# Check weights statistics
for name, param in network.named_parameters():
    print(f"{name}: min={param.min().item()}, max={param.max().item()}, mean={param.mean().item()}")

from AlphaZero.core.mcts import MCTS
from AlphaZero.agent.alpha_agent import AlphaZeroAgent
from game.board import Board
from game.game_logic import GameLogic
from agent.base import AgentType
from AlphaZero.model.state_encoder import StateEncoder
from AlphaZero.model.action_mapper import ActionMapper
# Create a game with identical state
board = Board()
game = GameLogic(board, agent_types=[AgentType.ALPHAZERO] + [AgentType.RANDOM] * 3)

# Manually set up the model exactly as in training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load("models/model_iter_38.pt", map_location=device)
network = DeepCatanNetwork(992, 200, 256).to(device)
network.load_state_dict(checkpoint['network_state_dict'])
network.eval()
network.to(device)

# Create MCTS and agent
state_encoder = StateEncoder(max_actions=200)
action_mapper = ActionMapper(max_actions=200)
mcts = MCTS(network, state_encoder, action_mapper, num_simulations=100, c_puct=1.5)
agent = AlphaZeroAgent(0, network, state_encoder, action_mapper, mcts)
agent.set_training_mode(False)
game.agents[0] = agent

# Run setup phase to get to a valid game state
while not game.is_setup_complete():
    game.process_ai_turn()

game.process_ai_turn()  # Process the first AI turn

# Now dump the MCTS search results for analysis
action_probs, value = mcts.search(game.state)
print(f"Top actions from MCTS search:")
sorted_actions = sorted(action_probs.items(), key=lambda x: x[1], reverse=True)
for i, (act, prob) in enumerate(sorted_actions[:5]):
    print(f"  {i+1}. {act} with probability {prob:.4f}")


def debug_mcts_expansion(game_state):
    """Test MCTS node expansion logic"""
    # Set up encoding
    state_encoder = StateEncoder(max_actions=200)
    action_mapper = ActionMapper(max_actions=200)
    
    # Get valid actions directly from game state
    direct_actions = list(game_state.possible_actions)
    print(f"Game state reports {len(direct_actions)} valid actions")
    
    # Get valid actions through encoder/mapper
    valid_mask = state_encoder.get_valid_action_mask(game_state)
    mask_actions = []
    for i, valid in enumerate(valid_mask):
        if valid:
            action = action_mapper.index_to_action(i, game_state)
            mask_actions.append(action)
    
    print(f"Encoder/mapper found {len(mask_actions)} valid actions")
    
    # Check for mismatches
    in_direct_not_mask = [a for a in direct_actions if a not in mask_actions]
    in_mask_not_direct = [a for a in mask_actions if a not in direct_actions]
    
    if in_direct_not_mask:
        print("Actions in game state but not in mask:")
        for a in in_direct_not_mask[:5]:  # Show first 5
            print(f"  {a}")
    
    if in_mask_not_direct:
        print("Actions in mask but not in game state:")
        for a in in_mask_not_direct[:5]:  # Show first 5
            print(f"  {a}")


def analyze_random_agent(n_trials=100):
    """Test if RandomAgent makes consistent decisions"""
    outcomes = []
    
    for _ in range(n_trials):
        board = Board()
        game = GameLogic(board, agent_types=[AgentType.RANDOM] * 4)
        
        # Run setup phase
        while not game.is_setup_complete():
            game.process_ai_turn()
        
        # Play a few moves
        for _ in range(10):  # Run 10 moves
            if game.state.current_player_idx == 0:
                # Get all possible actions
                possible = list(game.state.possible_actions)
                # Let the agent choose
                agent = game.agents[0]
                action = agent.get_action(game.state)
                # Record choice info
                outcomes.append({
                    'n_choices': len(possible),
                    'choice_idx': possible.index(action) if action in possible else -1
                })
                
                # Apply the action
                game.do_action(action)
            else:
                game.process_ai_turn()
    
    # Analyze the distribution
    print(f"Random agent analysis ({n_trials} trials):")
    print(f"Average options: {sum(o['n_choices'] for o in outcomes) / len(outcomes)}")
    print(f"Invalid choices: {sum(1 for o in outcomes if o['choice_idx'] == -1)}")
    
    # Check for bias (should be uniform)
    positions = [o['choice_idx'] / o['n_choices'] for o in outcomes if o['n_choices'] > 0]
    hist = [0] * 10
    for pos in positions:
        bucket = min(int(pos * 10), 9)
        hist[bucket] += 1
    
    print("Distribution (should be roughly uniform):")
    for i, count in enumerate(hist):
        print(f"  {i/10:.1f}-{(i+1)/10:.1f}: {count}")

from game.enums import Resource
def debug_model(model_path):
    # Create a game
    board = Board()
    game = GameLogic(board, agent_types=[AgentType.ALPHAZERO] + [AgentType.RANDOM] * 3)
    
    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    network = DeepCatanNetwork(992, 200, 256).to(device)
    network.load_state_dict(checkpoint['network_state_dict'])
    network.eval()
    
    # Create the agent
    state_encoder = StateEncoder(max_actions=200)
    action_mapper = ActionMapper(max_actions=200)
    mcts = MCTS(network, state_encoder, action_mapper, num_simulations=100, c_puct=1.5)
    agent = AlphaZeroAgent(0, network, state_encoder, action_mapper, mcts)
    agent.set_training_mode(False)  # IMPORTANT: Try with both True and False
    game.agents[0] = agent
    
    # Handle the setup phase
    while not game.is_setup_complete():
        game.process_ai_turn()
    
    # Give resources to test building
    player = game.state.players[0]
    player.resources[Resource.BRICK] = 10
    player.resources[Resource.WOOD] = 10
    player.resources[Resource.SHEEP] = 10
    player.resources[Resource.WHEAT] = 10
    player.resources[Resource.ORE] = 10
    
    # Now check if building actions are available
    print(f"Available actions: {len(game.state.possible_actions)}")
    for action in game.state.possible_actions:
        print(f"  {action}")
    
    # Run MCTS search
    action_probs, value = mcts.search(game.state)
    
    
    # Print top actions
    print("\nTop actions from MCTS:")
    sorted_actions = sorted(action_probs.items(), key=lambda x: x[1], reverse=True)
    for i, (act, prob) in enumerate(sorted_actions[:10]):
        print(f"  {i+1}. {act} with probability {prob:.4f}")
    
    # Check agent's final choice
    action = agent.get_action(game.state)
    print(f"\nAgent's final choice: {action}")

if __name__ == "__main__":
    # Run the tests
    debug_mcts_expansion(game.state)
    analyze_random_agent()  
    debug_model("models/best_model.pt")
# from game.board import Board
# from game.game_logic import GameLogic
# from agent.base import AgentType
# import random
# import torch
# from AlphaZero.agent.alpha_agent import AlphaZeroAgent
# # Create identical game states
# board1 = Board()
# game1 = GameLogic(board1, agent_types=[AgentType.ALPHAZERO] + [AgentType.RANDOM] * 3)
# board2 = Board()
# game2 = GameLogic(board2, agent_types=[AgentType.ALPHAZERO] + [AgentType.RANDOM] * 3)

# # Set identical random seeds
# torch.manual_seed(42)
# random.seed(42)

# # Create agent using training pipeline method
# # [code from your training pipeline]

# # Create agent using evaluation script method
# # [code from your evaluation script]

# # Run identical simulated games
# for i in range(10):  # For 10 moves
#     # Get action from training pipeline agent
#     action1 = agent1.get_action(game1.state)
#     # Get action from evaluation script agent
#     action2 = agent2.get_action(game2.state)
    
#     print(f"Move {i}:")
#     print(f"  Training agent chose: {action1}")
#     print(f"  Evaluation agent chose: {action2}")
    
#     # Apply the same action to both games to keep them in sync
#     # (use action1 to keep states identical)
#     game1.do_action(action1)
#     game2.do_action(action1)