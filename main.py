from agent.base import AgentType
from gui.catan_game import CatanGame

def main():
    agent_types = [AgentType.RANDOM] * 4
    agent_types[0] = AgentType.HEURISTIC

    # Start the game with selected setup
    game = CatanGame(
        agent_types=agent_types
    )
    game.run()

if __name__ == "__main__":
    main()