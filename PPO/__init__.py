from .agent import PPOAgent
from .environment import CatanEnvironment

PPO_EPSILON = 0.2  
CRITIC_DISCOUNT = 0.5 
ENTROPY_BETA = 0.01  
VALUE_LOSS_COEF = 0.5 
MAX_GRAD_NORM = 0.5 
GAMMA = 0.99  
GAE_LAMBDA = 0.95 