import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class NavigateExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor for Tello Navigation Policy.
    Processes dict observations: 
      - "depth": (1, 48, 64) -> CNN -> cnn_output_dim features
      - "state": (60,) -> Raw Passthrough -> 60 features
    Concatenates features to form the final feature vector.
    """
    def __init__(self, observation_space: spaces.Dict, cnn_output_dim: int = 32):
        # Ensure we're dealing with the correct observation space
        assert isinstance(observation_space, spaces.Dict), "NavigateExtractor requires Dict observation space"
        
        # Dynamically calculate total features dim
        state_dim = observation_space.spaces["state"].shape[0]
        features_dim = cnn_output_dim + state_dim
        
        super().__init__(observation_space, features_dim)
        
        # 1. CNN for depth image
        depth_channels = observation_space.spaces["depth"].shape[0]
        self.cnn_features = nn.Sequential(
            nn.Conv2d(depth_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.cnn_linear = nn.Linear(64 * 6 * 8, cnn_output_dim)
        
        # [CRITICAL STABILITY FIX]
        # Initialize the CNN output to exactly zero. 
        nn.init.zeros_(self.cnn_linear.weight)
        nn.init.zeros_(self.cnn_linear.bias)
        
        self.cnn_relu = nn.ReLU()
        
        # 2. No MLP for state vector (direct passthrough to avoid parameter bloat)
        
    def forward(self, observations):
        """
        observations: dict of tensors "depth" and "state"
        """
        depth = observations["depth"]
        state = observations["state"]
        
        # Process depth
        cnn_out = self.cnn_features(depth)
        cnn_out = self.cnn_linear(cnn_out)
        cnn_features = self.cnn_relu(cnn_out)
        
        # Concatenate (batch_size, cnn_output_dim + state_dim)
        combined_features = torch.cat((cnn_features, state), dim=1)
        
        return combined_features
