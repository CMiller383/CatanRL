# Settlers of Catan AlphaZero

## Overview
This project implements an AlphaZero-based reinforcement learning agent for playing the board game Settlers of Catan. By combining Monte Carlo Tree Search (MCTS) with deep neural networks, our agent learns to play competitive Catan without human-crafted game heuristics.

## Environment Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/CMiller383/CatanRL.git
   cd catanrl
   ```
2. **Create a virtual environment**  
   ```bash
    conda create -n catan-env python=3.11.11
    conda activate catan-env
   ```
3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

## Playing

Use the `main.py` entrypoint to launch a game:
```bash
python main.py [options]
```

**Examples:**
- **Play vs. AlphaZero**  
  ```bash
  python main.py --alphazero --model models/best_model.pt --player-position 1
  ```
- **Watch AI-only game**  
  ```bash
  python main.py --alphazero --all-ai
  ```
- **Custom agents** (H=Human, R=Random, E=Heuristic, A=AlphaZero):  
  ```bash
  python main.py --agents H,A,R,R
  ```
See `main.py` for more options and details.


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

- **Deep Residual Network**: Modified ResNet with dual heads (policy & value)
- **State Encoding**: Rich representation of board, resources, and development cards
- **Policy Head**: Outputs move probabilities
- **Value Head**: Estimates win probability
- **Implementation**: PyTorch-based, configurable via `AlphaZero/utils/config.py`

### Monte Carlo Tree Search (MCTS)

- **Selection**: PUCT algorithm for exploration/exploitation
- **Expansion**: Child nodes created with policy priors
- **Evaluation**: Value network estimates position strength
- **Backpropagation**: Statistics updated based on evaluations
- **Action Selection**: Temperature-controlled sampling during training

### Optimizations

- Batched neural network evaluation for throughput
- Virtual loss to support parallel search
- Early pruning of low-value branches

## TODO
- Ports support in the engine
- Full trading system
- Manual resource discarding logic

## Training

Training can take a long time depending on mode and hardware—be patient. All available training configuration options are defined in `AlphaZero/utils/config.py`.

Recommended to use notebook provided and cloud based training—local is not gonna be fast.

Start training with the `train.py` script:
```bash
python train.py [options]
```

**Common modes:**
- `--quick`: 1 iteration, 2 games, 10 simulations
- `--medium`: 10 iterations, 5 games, 50 simulations
- `--full`: 50 iterations, 20 games, 100 simulations
- `--overnight`: 100 iterations, 30 games, 50 simulations

**Other flags:**
- `--iterations N`: number of training iterations
- `--games N`: number of self-play games per iteration
- `--sims N`: number of MCTS simulations per move
- `--eval-games N`: number of evaluation games
- `--resume PATH`: resume from model checkpoint

See `train.py` for more details.

## Contributors
- Cole Miller ([cmiller383@gatech.edu](mailto:cmiller383@gatech.edu))
- Christian Lindler ([clindler3@gatech.edu](mailto:clindler3@gatech.edu))

## License
This project is licensed under the MIT License.

