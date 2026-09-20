import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class NavigateExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor for Tello Navigation Policy.
    Processes dict observations: 
      - "depth": (1, 48, 64) -> CNN -> 64 features
      - "state": (15,) -> MLP -> 64 features
    Concatenates features to form 128D vector for PPO policy.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 320):
        # Ensure we're dealing with the correct observation space
        assert isinstance(observation_space, spaces.Dict), "NavigateExtractor requires Dict observation space"
        
        super().__init__(observation_space, features_dim)
        
        # 1. CNN for depth image (1, 48, 64) -> 64 features
        # Assuming input shape (1, 48, 64)
        depth_channels = observation_space.spaces["depth"].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(depth_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 6 * 8, 256),
            nn.ReLU(),
        )
        
        # 2. MLP for state vector
        state_dim = observation_space.spaces["state"].shape[0] # Should be 15
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        
    def forward(self, observations):
        """
        observations: dict of tensors "depth" and "state"
        """
        depth = observations["depth"]
        state = observations["state"]
        
        # Process depth (batch_size, 1, 48, 64)
        cnn_features = self.cnn(depth)
        
        # Process state (batch_size, 15)
        mlp_features = self.mlp(state)
        
        # Concatenate (batch_size, 128)
        combined_features = torch.cat((cnn_features, mlp_features), dim=1)
        
        return combined_features
