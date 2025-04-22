import pickle
import numpy as np
from itertools import islice
from collections import Counter

# 1) Load raw replay buffer
with open("models/latest_buffer.pkl", "rb") as f:
    raw_buffer = pickle.load(f)

# 2) Inspect raw buffer
print(f"Raw buffer type : {type(raw_buffer)}")
print(f"Total examples   : {len(raw_buffer)}")
first = next(iter(raw_buffer))
print(f"Example keys     : {list(first.keys())}")
print()

# 3) Set up your pipeline (loads your normal config & placement components)
from AlphaZero.training.training_pipeline import TrainingPipeline
pipeline = TrainingPipeline(use_placement_network=False)

# 4) Extract initial‐placement examples
placement_data = pipeline.extract_placement_data(raw_buffer)
print(f"Placement examples extracted: {len(placement_data)}")

# 5) Reward stats on placement data
rewards = [ex["reward"] for ex in placement_data]
print("Placement reward range:", min(rewards), "to", max(rewards))
print("Top 10 reward counts:", Counter(rewards).most_common(10))
print()

# 6) Peek at first 5 placement examples
for i, ex in enumerate(islice(placement_data, 5)):
    state  = np.array(ex["state"])
    target = np.array(ex["target"])
    mask   = np.array(ex["valid_mask"])
    print(f"Example #{i}")
    print("  state shape :", state.shape)
    print("  target sum  :", target.sum(), "(should be 1.0)")
    print("  mask sum    :", mask.sum(), "(# valid spots)")
    print("  spot_id     :", ex["spot_id"])
    print("  phase       :", ex["phase"])
    print("  reward      :", ex["reward"])
    print()
