#!/usr/bin/env python
"""
Head‑less evaluation of an AlphaZero Catan agent against baselines.

After **each game** the script prints:

    Game 17 | winner 2 (Random) | VP 10 | moves 137

and at the end a rich stats table.

CLI
---
python eval_runner.py --model models/best_model.pt --games 500 --seats 0,1,2,3
"""
import os, sys, argparse, random
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

# ------------------------------------------------------------------ #
#  Project imports                                                   #
# ------------------------------------------------------------------ #
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AlphaZero.utils.config import get_config
from agent.base import AgentType, create_agent
from agent.random_agent import RandomAgent
from game.board import Board
from game.game_logic import GameLogic


# ------------------------------------------------------------------ #
#  Helper functions                                                  #
# ------------------------------------------------------------------ #
def set_random_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def winner_idx(state):
    """Return winner index (tie‑break: highest VP)."""
    return state.winner if state.winner is not None else max(
        range(4), key=lambda i: state.players[i].victory_points
    )


# ------------------------------------------------------------------ #
#  Single head‑less game                                             #
# ------------------------------------------------------------------ #
def play_one_game(agent_factories, max_moves):
    """Return (winner, vps:list[int], moves:int)."""
    board = Board()
    game = GameLogic(board, agent_types=[AgentType.RANDOM] * 4)

    for seat in range(4):
        game.agents[seat] = agent_factories[seat]()

    while not game.is_setup_complete():
        game.process_ai_turn()

    moves = 0
    while game.state.winner is None and moves < max_moves:
        game.process_ai_turn()
        moves += 1

    vps = [p.victory_points for p in game.state.players]
    return winner_idx(game.state), vps, moves


# ------------------------------------------------------------------ #
#  Evaluation loop                                                   #
# ------------------------------------------------------------------ #
def evaluate(cfg, model_path, n_games, max_moves, seats, seed):
    set_random_seeds(seed)

    # ── stats collectors ────────────────────────────────────────────
    wins = defaultdict(int)
    vp_log = [[] for _ in range(4)]
    move_log = []

    for g in range(n_games):
        az_seat = seats[g % len(seats)]

        # factories per seat
        factories = []
        for seat in range(4):
            if seat == az_seat:
                factories.append(
                    lambda _seat=seat: create_agent(
                        player_id=_seat,
                        agent_type=AgentType.ALPHAZERO,
                        model_path=model_path,
                    )
                )
            else:
                factories.append(lambda _seat=seat: RandomAgent(_seat))

        winner, vps, moves = play_one_game(factories, max_moves)

        # ── per‑game print ───────────────────────────────────────────
        role = "AlphaZero" if winner == az_seat else "Random"
        print(
            f"Game {g+1:<3} | winner {winner} ({role}) | VP {vps[winner]:<2} | "
            f"moves {moves}"
        )

        # ── aggregate stats ─────────────────────────────────────────
        wins[winner] += 1
        for i in range(4):
            vp_log[i].append(vps[i])
        move_log.append(moves)

    # ── final summary ───────────────────────────────────────────────
    moves_arr = np.array(move_log)
    print("\n========== Evaluation Summary ==========")
    print(f"Games played   : {n_games}")
    print(f"AZ seat cycle  : {', '.join(map(str, seats))}")
    print(f"Move cap/game  : {max_moves}")
    print(f"Avg length     : {moves_arr.mean():.1f} ± {moves_arr.std():.1f}")
    print(f"Min / Max      : {moves_arr.min()} / {moves_arr.max()}\n")

    header = (
        "Player | Wins | Win‑Rate | VP mean±std | VP median"
    )
    print(header)
    print("-" * len(header))
    for i in range(4):
        vps = np.array(vp_log[i])
        print(
            f"{i:^6} | "
            f"{wins[i]:^4} | "
            f"{wins[i]/n_games:>8.2%} | "
            f"{vps.mean():>6.2f}±{vps.std():<5.2f} | "
            f"{np.median(vps):>8.2f}"
        )

    # VP gap AZ vs best random (if AZ seat list is unique)
    az_vp = []
    rnd_best = []
    for idx, seat in enumerate(seats * (n_games // len(seats) + 1))[: n_games]:
        az_vp.append(vp_log[seat][idx])
        rnd_vps = [vp_log[p][idx] for p in range(4) if p != seat]
        rnd_best.append(max(rnd_vps))
    diff = np.array(az_vp) - np.array(rnd_best)
    print(f"\nAvg VP diff (AZ – best Random): {diff.mean():.2f}")
    print("========================================")


# ------------------------------------------------------------------ #
#  CLI                                                               #
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="Head‑less evaluation of AlphaZero Catan agent"
    )
    parser.add_argument("--model", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--games", type=int, default=200, help="#games to play")
    parser.add_argument("--max-moves", type=int, default=200, help="Cap moves")
    parser.add_argument("--seats", type=str, default="0", help="AZ seats CSV")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    seats = [int(s) for s in args.seats.split(",")]
    cfg = get_config()  # only needed if your loader reads it

    evaluate(cfg, args.model, args.games, args.max_moves, seats, args.seed)


if __name__ == "__main__":
    main()
