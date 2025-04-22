import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from game.enums import Resource, SettlementType

class InitialPlacementEncoder:
    """Encodes board state specifically for initial placement decisions"""
    
    def __init__(self):
        # Resource type mapping for one-hot encoding
        self.resource_mapping = {
            Resource.WOOD: 0,
            Resource.BRICK: 1,
            Resource.WHEAT: 2,
            Resource.SHEEP: 3,
            Resource.ORE: 4,
            Resource.DESERT: 5
        }
    
    def encode_board(self, game_state):
        """
        Create a specialized encoding for initial placement decisions
        
        Args:
            game_state: The game state to encode
            
        Returns:
            encoded_state: Numpy array of the encoded state
        """
        board = game_state.board
        
        # Features:
        # - For each hex (19 hexes):
        #   - Resource type (6 types, one-hot)
        #   - Dice number (normalized)
        #   - Is desert flag
        # - For each spot (54 spots):
        #   - Has settlement flag (binary)
        #   - Has port flag (binary)
        
        # Hex features: 19 hexes × (6 + 1 + 1) = 152 features
        hex_features = []
        for hex_id, hex_obj in board.hexes.items():
            # Resource type (one-hot)
            resource_one_hot = [0] * 6
            resource_index = self.resource_mapping[hex_obj.resource]
            resource_one_hot[resource_index] = 1
            
            # Dice number (normalized)
            dice_value = hex_obj.number / 12.0 if hex_obj.number > 0 else 0
            
            # Is desert flag
            is_desert = 1.0 if hex_obj.resource == Resource.DESERT else 0.0
            
            hex_features.extend(resource_one_hot + [dice_value, is_desert])
        
        # Spot features: 54 spots × 2 = 108 features
        spot_features = []
        for spot_id, spot in board.spots.items():
            # Is occupied flag
            is_occupied = 1.0 if spot.player_idx is not None else 0.0
            
            # Has port flag
            has_port = 1.0 if spot.has_port else 0.0
            
            spot_features.extend([is_occupied, has_port])
        
        # Combine all features and convert to numpy array
        all_features = np.array(hex_features + spot_features, dtype=np.float32)
        
        return all_features
    
    def get_valid_placement_mask(self, game_state):
        """
        Create a binary mask of valid placement spots
        
        Args:
            game_state: The game state
            
        Returns:
            valid_mask: Binary mask (1 for valid placement, 0 otherwise)
        """
        from game.setup import is_valid_initial_settlement
        
        valid_mask = np.zeros(54, dtype=np.float32)
        
        for spot_id in range(1, 55):  # Assuming 54 spots, ids 1-54
            if is_valid_initial_settlement(game_state, spot_id):
                valid_mask[spot_id-1] = 1.0  # -1 for 0-indexing
                
        return valid_mask


class InitialPlacementNetwork(nn.Module):
    """
    Neural network for evaluating initial settlement placements
    """
    def __init__(self, input_dim=260, hidden_dim=128, output_dim=54):
        super(InitialPlacementNetwork, self).__init__()
        
        # Define network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor
            
        Returns:
            output: Probability distribution over placement spots
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        
        # We don't apply softmax here as it will be applied with masking later
        return logits
    
    def predict(self, state_tensor, valid_mask=None):
        """
        Get placement probabilities with optional valid mask
        
        Args:
            state_tensor: Encoded state tensor
            valid_mask: Binary mask of valid placements
            
        Returns:
            probs: Probability distribution over spots
        """
        with torch.no_grad():
            logits = self(state_tensor)
            
            if valid_mask is not None:
                # Create mask tensor
                mask = torch.zeros_like(logits)
                mask = mask.masked_fill(valid_mask.bool(), -float('inf'))
                
                # Apply mask
                masked_logits = logits + mask
                
                # Get probabilities
                probs = F.softmax(masked_logits, dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
                
            return probs


class InitialPlacementTrainer:
    """
    Trainer for the initial placement network
    """
    def __init__(self, network, optimizer=None, lr=0.001):
        self.network = network
        
        # Create optimizer if not provided
        self.optimizer = optimizer or torch.optim.Adam(
            network.parameters(), lr=lr
        )
        
        # Data buffer
        self.data_buffer = []
        
        # Encoder
        self.encoder = InitialPlacementEncoder()
        
    def add_example(self, game_state, settlement_id, reward):
        """
        Add a training example to the buffer
        
        Args:
            game_state: Game state before placement
            settlement_id: The settlement spot chosen
            reward: Final reward from the game
        """
        # Encode the state
        encoded_state = self.encoder.encode_board(game_state)
        
        # Create target (one-hot for the chosen spot)
        target = np.zeros(54, dtype=np.float32)
        target[settlement_id-1] = 1.0  # -1 for 0-indexing
        
        # Create valid placement mask
        valid_mask = self.encoder.get_valid_placement_mask(game_state)
        
        # Store example
        self.data_buffer.append({
            'state': encoded_state,
            'target': target,
            'valid_mask': valid_mask,
            'reward': reward,
            'phase': game_state.current_phase.value
        })
    
    def extract_from_game_history(self, agent_game_history):
        """
        Extract relevant placement data from a full game history
        
        Args:
            agent_game_history: Full game history from AlphaZero agent
            
        Returns:
            placement_data: Extracted placement examples
        """
        # TODO: Implement extraction from full game history
        # This will depend on how your history is structured
        pass
        
    def train(self, epochs=10, batch_size=32):
        """
        Train the network on collected examples
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            metrics: Training metrics
        """
        if len(self.data_buffer) < batch_size:
            print(f"Not enough data for training: {len(self.data_buffer)} < {batch_size}")
            return {}
        
        self.network.train()
        
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            batches = 0
            
            # Shuffle data
            indices = np.random.permutation(len(self.data_buffer))
            
            for start in range(0, len(indices), batch_size):
                # Get batch indices
                batch_indices = indices[start:start+batch_size]
                if len(batch_indices) < batch_size:
                    # Skip incomplete batch
                    continue
                
                # Prepare batch data
                states = torch.tensor(np.stack([self.data_buffer[i]['state'] for i in batch_indices]), 
                                    dtype=torch.float32)
                targets = torch.tensor(np.stack([self.data_buffer[i]['target'] for i in batch_indices]), 
                                    dtype=torch.float32)
                masks = torch.tensor(np.stack([self.data_buffer[i]['valid_mask'] for i in batch_indices]), 
                                    dtype=torch.float32)
                rewards = torch.tensor(np.array([self.data_buffer[i]['reward'] for i in batch_indices]), 
                                    dtype=torch.float32).view(-1, 1)
                
                # Reset gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                logits = self.network(states)
                
                # Apply mask and get probabilities
                masked_logits = logits.clone()
                for i in range(len(masked_logits)):
                    # Set invalid placements to -inf
                    valid_indices = masks[i] > 0
                    invalid_indices = ~valid_indices
                    masked_logits[i][invalid_indices] = -1e9
                
                probs = F.softmax(masked_logits, dim=1)
                
                # Calculate loss - cross entropy weighted by reward
                loss = torch.mean(((rewards + 1.0) / 2.0).view(-1) * F.cross_entropy(masked_logits, torch.argmax(targets, dim=1), reduction='none'))

                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                # Calculate accuracy
                predictions = torch.argmax(probs, dim=1)
                targets_idx = torch.argmax(targets, dim=1)
                accuracy = (predictions == targets_idx).float().mean().item()
                
                # Update metrics
                epoch_loss += loss.item()
                epoch_acc += accuracy
                batches += 1
            
            # Average metrics
            if batches > 0:
                avg_loss = epoch_loss / batches
                avg_acc = epoch_acc / batches
                losses.append(avg_loss)
                accuracies.append(avg_acc)
                
                print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, Accuracy = {avg_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}: No complete batches")
        
        return {
            'loss': np.mean(losses) if losses else 0.0,
            'accuracy': np.mean(accuracies) if accuracies else 0.0
        }
    
    def save_model(self, path):
        """Save the model to a file"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'data_buffer_size': len(self.data_buffer)
        }, path)
        
    def load_model(self, path):
        """Load the model from a file"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint