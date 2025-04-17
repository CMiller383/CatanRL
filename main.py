#!/usr/bin/env python
import argparse
import os
from agent.base import AgentType
from gui.catan_game import CatanGame

def main():
    parser = argparse.ArgumentParser(description="Settlers of Catan with AlphaZero")
    
    parser.add_argument("--alphazero", action="store_true", 
                      help="Play against AlphaZero agent")
    parser.add_argument("--model", type=str, default="models/best_model.pt", 
                      help="Path to AlphaZero model file (default: models/best_model.pt)")
    parser.add_argument("--player-position", type=int, default=0, choices=[0, 1, 2, 3],
                      help="Position of the human player (0-3), default: 0")
    parser.add_argument("--all-ai", action="store_true", 
                      help="Watch AI-only game (no human players)")
    
    parser.add_argument("--agents", type=str, default="H,R,R,R",
                      help="Comma-separated agent types: H=Human, R=Random, E=Heuristic, A=AlphaZero")
    
    args = parser.parse_args()
    
    # Default agent types setup
    if args.agents and not args.alphazero:
        # Parse custom agent setup
        agent_codes = args.agents.split(",")
        agent_types = []
        
        for code in agent_codes[:4]:  # Limit to 4 players
            code = code.strip().upper()
            if code == 'H':
                agent_types.append(AgentType.HUMAN)
            elif code == 'R':
                agent_types.append(AgentType.RANDOM)
            elif code == 'E':
                agent_types.append(AgentType.HEURISTIC)
            elif code == 'A':
                agent_types.append(AgentType.ALPHAZERO)
                # model path because im too lazy to fix all the other functions that rely on this
                os.environ['ALPHAZERO_MODEL_PATH'] = args.model
            else:
                # Default to random for invalid codes
                agent_types.append(AgentType.RANDOM)
        
        # Fill remaining slots with random agents
        while len(agent_types) < 4:
            agent_types.append(AgentType.RANDOM)
    
    # AlphaZero specific setup
    elif args.alphazero:
        os.environ['ALPHAZERO_MODEL_PATH'] = args.model
        
        if args.all_ai:
            # AI-only game with AlphaZero
            agent_types = [AgentType.ALPHAZERO, AgentType.RANDOM, 
                          AgentType.RANDOM, AgentType.RANDOM]
        else:
            # Human vs AlphaZero game
            agent_types = [AgentType.RANDOM] * 4
            human_position = args.player_position
            agent_types[human_position] = AgentType.HUMAN
            
            # Place AlphaZero at position 0 or the position after the human
            alphazero_position = 0 if human_position != 0 else 1
            agent_types[alphazero_position] = AgentType.ALPHAZERO
    else:
        # Default game with human as player 0 
        agent_types = [AgentType.HUMAN, AgentType.RANDOM, 
                      AgentType.RANDOM, AgentType.RANDOM]
    
    # Start the game with selected setup
    game = CatanGame(agent_types=agent_types)
    game.run()

if __name__ == "__main__":
    main()