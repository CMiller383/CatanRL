# Settlers of Catan RL Agent

## Overview
This project aims to develop a Reinforcement Learning (RL) agent capable of playing Settlers of Catan at a competitive level. By leveraging the Proximal Policy Optimization (PPO) algorithm, our agent will outperform existing heuristic-based bots, particularly those on Colonist.io.

## Features
- **Custom Game Engine**: A headless Catan game implementation using a graph-based representation.
- **Heuristic Baseline Agents**: Simple rule-based bots for training and benchmarking.
- **RL Agent**: A deep reinforcement learning model trained against heuristic agents.
- **Colonist.io Integration**: Enables the RL agent to interact with an online Catan platform.

## Implementation Details
- **Game Engine**: Tracks game state, resources, settlements, and roads.
- **State Representation**: Encodes board configurations, resource availability, and player actions.
- **Training Pipeline**: Uses PPO to iteratively refine decision-making strategies.
- **Evaluation**: Measures performance via win rates against heuristic and Colonist.io bots.

## Installation & Usage
Coming soon

## Contributors
- Christian Lindler ([clindler3@gatech.edu](mailto:clindler3@gatech.edu))
- Cole Miller ([cmiller383@gatech.edu](mailto:cmiller383@gatech.edu))

## License
This project is licensed under the MIT License.
