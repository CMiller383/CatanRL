from collections import deque
import torch
import torch.nn.functional as F
import numpy as np
from AlphaZero.model.action_mapper import ActionMapper

class NetworkTrainer:
    """
    Trainer for the neural network using self-play data
    """
    def __init__(self, network, optimizer, config):
        """
        Initialize the network trainer
        Args:
            network: Neural network to train
            optimizer: Optimizer for training
            config: Configuration dictionary
        """
        self.network = network
        self.optimizer = optimizer
        self.config = config

        # Device for tensors
        self.device = next(self.network.parameters()).device

        # Action mapper (instantiate once)
        action_dim = config.get('action_dim', 200)
        self.action_mapper = ActionMapper(action_dim)

        # Training data buffer
        self.data_buffer = deque(maxlen=config.get('buffer_size', 100000))

    def add_game_data(self, game_data):
        """
        Add game data to the buffer
        Args:
            game_data: List of state, action_probs, and reward tuples
        """
        self.data_buffer.extend(game_data)

    def train(self, epochs=None, batch_size=None):
        """
        Train the network on the current data buffer
        Returns:
            metrics: Dict of average losses
        """
        self.network.train()

        epochs = epochs or self.config.get('epochs', 10)
        batch_size = batch_size or self.config.get('batch_size', 128)

        if len(self.data_buffer) < batch_size:
            print(f"Not enough data for training: {len(self.data_buffer)} < {batch_size}")
            return {}

        total_losses, value_losses, policy_losses = [], [], []

        for epoch in range(epochs):
            # Sample a subset for speed
            sample_size = min(self.config.get('sample_size', 10000), len(self.data_buffer))
            indices = np.random.choice(len(self.data_buffer), sample_size, replace=False)
            samples = [self.data_buffer[i] for i in indices]

            epoch_loss = epoch_value = epoch_policy = 0.0
            batches = 0

            for start in range(0, sample_size, batch_size):
                batch = samples[start:start+batch_size]
                if len(batch) < batch_size:
                    break

                # Prepare batch tensors
                states = torch.stack([torch.FloatTensor(step['state']) for step in batch])
                rewards = torch.FloatTensor([step['reward'] for step in batch]).unsqueeze(1)

                # Policy targets
                policy_targets = torch.zeros(batch_size, self.config.get('action_dim', 200), device=self.device)
                for j, probs in enumerate([step['action_probs'] for step in batch]):
                    for action, prob in probs.items():
                        idx = self.action_mapper.action_to_index(action)
                        policy_targets[j, idx] = prob

                # Move data to device
                states = states.to(self.device)
                rewards = rewards.to(self.device)

                # Forward
                policy_logits, values = self.network(states)

                # Losses
                value_loss = F.mse_loss(values, rewards)
                log_probs = F.log_softmax(policy_logits, dim=1)
                policy_loss = -(policy_targets * log_probs).sum(dim=1).mean()

                loss = value_loss + policy_loss

                # Backward + optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                               self.config.get('grad_clip', 1.0))
                self.optimizer.step()

                # Accumulate
                epoch_loss += loss.item()
                epoch_value += value_loss.item()
                epoch_policy += policy_loss.item()
                batches += 1

            if batches:
                avg_loss = epoch_loss / batches
                avg_value = epoch_value / batches
                avg_policy = epoch_policy / batches
            else:
                avg_loss = avg_value = avg_policy = 0.0

            total_losses.append(avg_loss)
            value_losses.append(avg_value)
            policy_losses.append(avg_policy)

            print(f"Epoch {epoch+1}/{epochs}: Loss {avg_loss:.4f} "
                  f"(Value {avg_value:.4f}, Policy {avg_policy:.4f})")

        return {
            'total_loss': sum(total_losses) / len(total_losses),
            'value_loss': sum(value_losses) / len(value_losses),
            'policy_loss': sum(policy_losses) / len(policy_losses)
        }

    def train_one_epoch(self, batch_size=None):
        """
        Train for a single epoch; useful for external loop control
        """
        self.network.train()
        batch_size = batch_size or self.config.get('batch_size', 128)

        if len(self.data_buffer) < batch_size:
            print(f"Not enough data for training: {len(self.data_buffer)} < {batch_size}")
            return 0.0

        sample_size = min(self.config.get('sample_size', 10000), len(self.data_buffer))
        indices = np.random.choice(len(self.data_buffer), sample_size, replace=False)
        samples = [self.data_buffer[i] for i in indices]

        epoch_loss = epoch_value = epoch_policy = 0.0
        batches = 0

        for start in range(0, sample_size, batch_size):
            batch = samples[start:start+batch_size]
            if len(batch) < batch_size:
                break

            states = torch.stack([torch.FloatTensor(step['state']) for step in batch])
            rewards = torch.FloatTensor([step['reward'] for step in batch]).unsqueeze(1)
            policy_targets = torch.zeros(batch_size, self.config.get('action_dim', 200), device=self.device)
            for j, probs in enumerate([step['action_probs'] for step in batch]):
                for action, prob in probs.items():
                    idx = self.action_mapper.action_to_index(action)
                    policy_targets[j, idx] = prob

            states = states.to(self.device)
            rewards = rewards.to(self.device)

            policy_logits, values = self.network(states)
            value_loss = F.mse_loss(values, rewards)
            policy_loss = -(policy_targets * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()

            loss = value_loss + policy_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                           self.config.get('grad_clip', 1.0))
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_value += value_loss.item()
            epoch_policy += policy_loss.item()
            batches += 1

        if batches:
            avg = epoch_loss / batches
            print(f"Training: Loss {avg:.4f} "
                  f"(Value {epoch_value/batches:.4f}, Policy {epoch_policy/batches:.4f})")
            return avg
        return 0.0
