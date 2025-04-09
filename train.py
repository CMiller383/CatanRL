#!/usr/bin/env python
"""
Simple training script for AlphaZero Catan.
Save this as train.py in your project root directory.
"""
import os
import sys
import argparse
from AlphaZero.training.training_pipeline import TrainingPipeline
from AlphaZero.utils.config import get_config

def main():
    parser = argparse.ArgumentParser(description="AlphaZero Catan Training")
    
    # Basic configuration
    parser.add_argument("--iterations", type=int, default=50, help="Number of training iterations")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--games", type=int, default=20, help="Number of self-play games per iteration")
    parser.add_argument("--sims", type=int, default=100, help="Number of MCTS simulations per move")
    parser.add_argument("--eval-games", type=int, default=10, help="Number of evaluation games")
    
    # Training modes
    parser.add_argument("--quick", action="store_true", help="Quick training (1 iteration, 2 games)")
    parser.add_argument("--medium", action="store_true", help="Medium training (10 iterations, 5 games)")
    parser.add_argument("--full", action="store_true", help="Full training (50 iterations, 20 games)")
    parser.add_argument("--overnight", action="store_true", help="Overnight training (100 iterations, 30 games)")
    
    args = parser.parse_args()
    
    # pre configed
    if args.quick:
        args.iterations = 1
        args.games = 2
        args.sims = 10
        args.eval_games = 2
        print("Running in QUICK mode (minimal training for testing)")
    elif args.medium:
        args.iterations = 10
        args.games = 5
        args.sims = 50
        args.eval_games = 5
        print("Running in MEDIUM mode (shorter training run)")
    elif args.full:
        args.iterations = 50
        args.games = 20
        args.sims = 100
        args.eval_games = 10
        print("Running in FULL mode (standard training run)")
    elif args.overnight:
        args.iterations = 100
        args.games = 30
        args.sims = 150
        args.eval_games = 15
        print("Running in OVERNIGHT mode (extended training run)")
    config = get_config()
    # Set up configuration
    config['num_iterations'] = args.iterations
    config['self_play_games'] = args.games
    config['num_simulations'] = args.sims
    config['eval_games'] = args.eval_games
    
    print(f"\n=== AlphaZero Catan Training ===")
    print(f"Iterations: {args.iterations}")
    print(f"Self-play games per iteration: {args.games}")
    print(f"MCTS simulations per move: {args.sims}")
    print(f"Resume from: {args.resume if args.resume else 'Starting fresh'}")
    
    pipeline = TrainingPipeline(config)
    
    try:
        pipeline.train(resume_from=args.resume)
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving checkpoint...")
        pipeline.save_model(pipeline.current_iteration)
        print("Checkpoint saved. You can resume with --resume models/model_iter_{}.pt".format(pipeline.current_iteration))

if __name__ == "__main__":
    main()