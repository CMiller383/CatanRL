# Settlers of Catan AlphaZero

## Overview
This project implements an AlphaZero-based reinforcement learning agent for playing the board game Settlers of Catan. By combining Monte Carlo Tree Search (MCTS) with deep neural networks, our agent learns to play competitive Catan without human-crafted game heuristics.

## Features
- **Custom Game Engine**: Headless Catan implementation using a graph-based hexagonal board representation
- **Reinforcement Learning Agent**: AlphaZero-style agent that learns through self-play
- **Baseline Agents**: Random and heuristic-based bots for training and benchmarking
- **Visualization Tools**: Performance tracking and game visualization capabilities
- **Training Pipeline**: Complete infrastructure for self-play, training, and evaluation

## Technical Implementation

### Game Engine Architecture
The game engine implements the complete Settlers of Catan rule set in a highly optimized, graph-based representation:

- **Board Representation**: The board is modeled as a collection of hexes, spots (vertices), and roads (edges)
  - `Board` class manages the relationships between these components
  - Each `Hex` contains a resource type and number token
  - `Spot` represents a vertex where settlements/cities can be placed
  - `Road` connects two spots and belongs to one player

- **Game State**: The `GameState` class encapsulates the complete game state
  - Tracks current player, phase, resources, victory points, and development cards
  - Implements game rules and constraints
  - Provides validation for player actions

- **Action System**: Actions are represented as typed data structures with payloads
  - `ActionType` enum defines all possible game actions
  - `Action` class pairs action types with relevant parameters
  - System handles the full range of Catan actions: building, trading, playing development cards, etc.

### AlphaZero Implementation

Our AlphaZero implementation follows the core principles from the original DeepMind paper with adaptations for Catan:

#### Neural Network Architecture

- **Deep Residual Network**: Uses a modified ResNet with dual heads
  - **State Encoding**: Rich encoding of the game state (board, resources, development cards)
  - **Policy Head**: Outputs probability distribution over all possible actions
  - **Value Head**: Estimates expected outcome (win probability) from current state
  - **Implementation**: PyTorch-based with configurable layers and parameters

#### Monte Carlo Tree Search (MCTS)

- **Search Algorithm**: Enhanced MCTS with neural network guidance
  - **Selection**: Uses PUCT algorithm to balance exploration and exploitation
  - **Expansion**: Creates child nodes for valid actions with probabilities from policy network
  - **Evaluation**: Uses value network to estimate position strength
  - **Backpropagation**: Updates node statistics based on evaluation results
  - **Action Selection**: Temperature-based sampling for exploration during training

- **Optimizations**:
  - Batched neural network evaluation for efficiency
  - Virtual loss for parallel search
  - Early pruning of low-value branches

#### Training Pipeline

- **Self-Play**: Agents generate training data by playing against themselves
  - Automatic generation of diverse, high-quality gameplay examples
  - Parallel execution for faster data collection
  - Action selection with temperature parameter for exploration
 
- **Neural Network Training**:
  - MSE loss for value head (predicting game outcome)
  - Cross-entropy loss for policy head (matching improved MCTS policy)
  - Gradient clipping for stable learning
  - Replay buffer for experience reuse

- **Evaluation**:
  - Regular benchmarking against baseline agents
  - Detailed metrics tracking (win rate, VP acquisition, building patterns)
  - Model versioning and checkpointing

### Adaptations for Catan

Catan poses unique challenges compared to games like Chess or Go:

1. **Stochasticity**: Dice rolls introduce randomness, requiring value network adaptation
2. **Partial Observability**: Limited information about opponent resources
3. **Complex Action Space**: Many action types with different constraints
4. **Resource Management**: Economy and trading aspects
5. **Negotiation**: Simplified trading model for AI training

Our implementation addresses these challenges through:

- Enhanced state representation capturing resource dynamics
- Sophisticated reward function balancing immediate and long-term objectives
- Action space design that encompasses the full range of Catan strategies
- Training with explicit exploration to handle the game's branching factor

## Repository Structure

```
├── game/                  # Core game engine
│   ├── board.py           # Board representation
│   ├── game_state.py      # Game state management
│   ├── rules.py           # Game rules implementation
│   ├── player.py          # Player state and actions
│   └── ...
├── agent/                 # Agent implementations
│   ├── base.py            # Base agent interface
│   ├── random_agent.py    # Random baseline agent
│   ├── human_agent.py     # Human player interface
│   └── ...
├── AlphaZero/             # AlphaZero implementation
│   ├── core/              # Core MCTS and neural network
│   │   ├── network.py     # Neural network implementation
│   │   └── mcts.py        # Monte Carlo Tree Search
│   ├── agent/             # AlphaZero agent
│   │   └── alpha_agent.py # Main agent implementation
│   ├── model/             # Model components
│   │   ├── state_encoder.py  # State encoding
│   │   └── action_mapper.py  # Action space mapping
│   ├── training/          # Training infrastructure
│   │   ├── self_play.py      # Self-play data generation
│   │   ├── network_trainer.py # Neural network training
│   │   ├── evaluator.py       # Agent evaluation
│   │   └── ...
│   └── utils/             # Utilities
├── gui/                   # Game visualization (if applicable)
├── models/                # Saved model checkpoints
└── scripts/               # Utility scripts
```

## Playing and Training
Coming soon

## Contributors
- Christian Lindler ([clindler3@gatech.edu](mailto:clindler3@gatech.edu))
- Cole Miller ([cmiller383@gatech.edu](mailto:cmiller383@gatech.edu))

## License
This project is licensed under the MIT License.
