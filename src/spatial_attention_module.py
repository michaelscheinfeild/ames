#!/usr/bin/env python3
"""
Spatial Attention 2D module for AMES
This is the attention mechanism used to weight local features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention2d(nn.Module):
    """
    Spatial Attention module that learns to attend to important spatial locations
    Used in AMES to weight local DINOv2 features
    """
    def __init__(self, in_channels, hidden_channels=256):
        super(SpatialAttention2d, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # Attention network
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """
        Forward pass
        Args:
            features: [B, C, H, W] input features
        Returns:
            weighted_features: [B, C, H, W] attention-weighted features
            attention_weights: [B, 1, H, W] attention weights
        """
        # Compute attention weights
        attention_weights = self.attention(features)  # [B, 1, H, W]
        
        # Apply attention to features
        weighted_features = features * attention_weights  # [B, C, H, W]
        
        return weighted_features, attention_weights
