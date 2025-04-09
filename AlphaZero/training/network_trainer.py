
from collections import deque
import torch
import torch.nn.functional as F
import numpy as np

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
        
        # Training data buffer
        self.data_buffer = deque(maxlen=config.get('buffer_size', 100000))
    
    def add_game_data(self, game_data):
        """
        Add game data to the buffer
        
        Args:
            game_data: List of state, action probs, and reward tuples
        """
        self.data_buffer.extend(game_data)
    
    def train(self, epochs=None, batch_size=None):
        """
        Train the network on the current data buffer
        
        Args:
            epochs: Number of training epochs
            batch_size: Size of training batches
            
        Returns:
            losses: List of losses for each epoch
        """
        if epochs is None:
            epochs = self.config.get('epochs', 10)
        
        if batch_size is None:
            batch_size = self.config.get('batch_size', 128)
        
        if len(self.data_buffer) < batch_size:
            print(f"Not enough data for training: {len(self.data_buffer)} < {batch_size}")
            return []
        
        losses = []
        value_losses = []
        policy_losses = []
        
        for epoch in range(epochs):
            # Sample from buffer
            indices = np.random.choice(len(self.data_buffer), min(10000, len(self.data_buffer)), replace=False)
            samples = [self.data_buffer[i] for i in indices]
            
            # Train in batches
            epoch_loss = 0
            epoch_value_loss = 0
            epoch_policy_loss = 0
            batches = 0
            
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i+batch_size]
                if len(batch) < batch_size:
                    continue
                
                # Prepare batch data
                states = torch.stack([torch.FloatTensor(step['state']) for step in batch])
                action_probs = [step['action_probs'] for step in batch]
                rewards = torch.FloatTensor([step['reward'] for step in batch]).unsqueeze(1)
                
                # Convert action_probs dictionaries to tensors
                # This requires mapping from action objects to indices
                from AlphaZero.model.action_mapper import ActionMapper
                action_mapper = ActionMapper(self.config.get('action_dim', 200))
                
                policy_targets = torch.zeros(len(batch), self.config.get('action_dim', 200))
                for j, probs in enumerate(action_probs):
                    for action, prob in probs.items():
                        # Convert action to index
                        action_idx = action_mapper.action_to_index(action)
                        policy_targets[j, action_idx] = prob
                
                # Forward pass
                policy_logits, value = self.network(states)
                
                # Calculate losses
                # Value loss: MSE
                value_loss = F.mse_loss(value, rewards)
                
                # Policy loss: Cross entropy
                policy_loss = -torch.sum(policy_targets * F.log_softmax(policy_logits, dim=1)) / len(batch)
                
                # Combined loss (weighted)
                loss = value_loss + policy_loss
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Track losses
                epoch_loss += loss.item()
                epoch_value_loss += value_loss.item()
                epoch_policy_loss += policy_loss.item()
                batches += 1
            
            # Calculate average losses
            if batches > 0:
                avg_loss = epoch_loss / batches
                avg_value_loss = epoch_value_loss / batches
                avg_policy_loss = epoch_policy_loss / batches
            else:
                avg_loss = avg_value_loss = avg_policy_loss = 0
            
            losses.append(avg_loss)
            value_losses.append(avg_value_loss)
            policy_losses.append(avg_policy_loss)
            
            print(f"Epoch {epoch+1}/{epochs}: Loss: {avg_loss:.4f} "
                  f"(Value: {avg_value_loss:.4f}, Policy: {avg_policy_loss:.4f})")
        
        return {
            'total_loss': sum(losses) / len(losses) if losses else 0,
            'policy_loss': sum(policy_losses) / len(policy_losses) if policy_losses else 0,
            'value_loss': sum(value_losses) / len(value_losses) if value_losses else 0
        }
        
    def train_one_epoch(self, batch_size=None):
        """
        Train for a single epoch (alternative method if the caller wants to control epochs)
        
        Args:
            batch_size: Size of training batches
            
        Returns:
            avg_loss: Average loss for this epoch
        """
        if batch_size is None:
            batch_size = self.config.get('batch_size', 128)
        
        if len(self.data_buffer) < batch_size:
            print(f"Not enough data for training: {len(self.data_buffer)} < {batch_size}")
            return 0.0
        
        # Sample from buffer
        indices = np.random.choice(len(self.data_buffer), min(10000, len(self.data_buffer)), replace=False)
        samples = [self.data_buffer[i] for i in indices]
        
        # Train in batches
        epoch_loss = 0
        epoch_value_loss = 0
        epoch_policy_loss = 0
        batches = 0
        
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            if len(batch) < batch_size:
                continue
            
            # Prepare batch data
            states = torch.stack([torch.FloatTensor(step['state']) for step in batch])
            action_probs = [step['action_probs'] for step in batch]
            rewards = torch.FloatTensor([step['reward'] for step in batch]).unsqueeze(1)
            
            # Convert action_probs dictionaries to tensors
            # This requires mapping from action objects to indices
            from AlphaZero.model.action_mapper import ActionMapper
            action_mapper = ActionMapper(self.config.get('action_dim', 200))
            
            policy_targets = torch.zeros(len(batch), self.config.get('action_dim', 200))
            for j, probs in enumerate(action_probs):
                for action, prob in probs.items():
                    # Convert action to index
                    action_idx = action_mapper.action_to_index(action)
                    policy_targets[j, action_idx] = prob
            
            # Forward pass
            policy_logits, value = self.network(states)
            
            # Calculate losses
            # Value loss: MSE
            value_loss = F.mse_loss(value, rewards)
            
            # Policy loss: Cross entropy
            policy_loss = -torch.sum(policy_targets * F.log_softmax(policy_logits, dim=1)) / len(batch)
            
            # Combined loss (weighted)
            loss = value_loss + policy_loss
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track losses
            epoch_loss += loss.item()
            epoch_value_loss += value_loss.item()
            epoch_policy_loss += policy_loss.item()
            batches += 1
        
        # Calculate average losses
        if batches > 0:
            avg_loss = epoch_loss / batches
            avg_value_loss = epoch_value_loss / batches
            avg_policy_loss = epoch_policy_loss / batches
            
            print(f"Training: Loss: {avg_loss:.4f} "
                f"(Value: {avg_value_loss:.4f}, Policy: {avg_policy_loss:.4f})")
            return avg_loss
        else:
            return 0.0
