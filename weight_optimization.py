"""
Weight Optimization Module
Implements deep learning-based weight optimization for alpha factors.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class ModelConfig:
    input_size: int
    hidden_size: int
    output_size: int
    learning_rate: float
    batch_size: int
    num_epochs: int

class AlphaWeightNet(nn.Module):
    def __init__(self, config: ModelConfig):
        """
        Initialize the two-layer neural network for weight optimization.
        
        Args:
            config (ModelConfig): Configuration for the neural network
        """
        super().__init__()
        self.layer1 = nn.Linear(config.input_size, config.hidden_size)
        self.layer2 = nn.Linear(config.hidden_size, config.output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.layer2(x)
        return x

class WeightOptimizer:
    def __init__(self, config: ModelConfig):
        """
        Initialize the weight optimization system.
        
        Args:
            config (ModelConfig): Configuration for the optimization process
        """
        self.config = config
        self.model = AlphaWeightNet(config)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()
        
    def train(self, 
              train_data: torch.Tensor, 
              train_labels: torch.Tensor,
              val_data: torch.Tensor = None,
              val_labels: torch.Tensor = None) -> Dict[str, List[float]]:
        """
        Train the weight optimization model.
        
        Args:
            train_data (torch.Tensor): Training input data
            train_labels (torch.Tensor): Training target labels
            val_data (torch.Tensor, optional): Validation input data
            val_labels (torch.Tensor, optional): Validation target labels
            
        Returns:
            Dict containing training history
        """
        history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            self.model.train()
            train_loss = self._train_epoch(train_data, train_labels)
            history["train_loss"].append(train_loss)
            
            # Validation phase
            if val_data is not None and val_labels is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss = self._evaluate(val_data, val_labels)
                    history["val_loss"].append(val_loss)
        
        return history
    
    def _train_epoch(self, train_data: torch.Tensor, train_labels: torch.Tensor) -> float:
        """Train for one epoch."""
        total_loss = 0
        num_batches = len(train_data) // self.config.batch_size
        
        for i in range(num_batches):
            start_idx = i * self.config.batch_size
            end_idx = start_idx + self.config.batch_size
            
            batch_data = train_data[start_idx:end_idx]
            batch_labels = train_labels[start_idx:end_idx]
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_data)
            loss = self.criterion(outputs, batch_labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / num_batches
    
    def _evaluate(self, val_data: torch.Tensor, val_labels: torch.Tensor) -> float:
        """Evaluate the model on validation data."""
        outputs = self.model(val_data)
        loss = self.criterion(outputs, val_labels)
        return loss.item()
    
    def predict_weights(self, alpha_data: torch.Tensor) -> torch.Tensor:
        """
        Predict optimal weights for alpha factors.
        
        Args:
            alpha_data (torch.Tensor): Input alpha factor data
            
        Returns:
            Predicted weights for each alpha factor
        """
        self.model.eval()
        with torch.no_grad():
            weights = self.model(alpha_data)
            # Normalize weights to sum to 1
            weights = torch.softmax(weights, dim=1)
        return weights
    
    def save_model(self, path: str):
        """Save the trained model."""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        """Load a trained model."""
        self.model.load_state_dict(torch.load(path))
