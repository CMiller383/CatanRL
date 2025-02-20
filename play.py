import argparse
from game.game import Game
from game.enums import PlayerId
from ui.display import Display

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trades-on', action='store_true', default=False, help="Turn trades on.")
    args = parser.parse_args()

    players = [PlayerId.White, PlayerId.Red, PlayerId.Orange, PlayerId.Blue]
    policies = {player: "human" for player in players}

    game = Game(interactive=True, debug_mode=False, policies=policies)
    game.reset()

    display = Display(env=game, game=game, interactive=True, policies=policies, test=False, debug_mode=False)
