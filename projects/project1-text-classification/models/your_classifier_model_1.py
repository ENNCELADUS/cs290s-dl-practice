import torch
import torch.nn as nn


"""
TODO: implement the first custom model for project_1 task.
"""
# =============================================================================
# Replace with your model
class YourClassifierModel1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the custom neural network
        
        Args:
            input_size (int): Dimensionality of the input features
            hidden_size (int): Size of the hidden layer
            output_size (int): Number of output classes
        """
        super().__init__()
        # Fully connected layer from input to hidden
        self.fc1 = nn.Linear(input_size, hidden_size)
        # ReLU activation function
        self.relu = nn.ReLU()
        # Fully connected layer from hidden to output
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Define the forward propagation process
        
        Args:
            x (torch.Tensor): Input data tensor
        
        Returns:
            torch.Tensor: Output of the neural network
        """
        # Linear transformation from input to hidden
        x = self.fc1(x)
        # ReLU activation function
        x = self.relu(x)
        # Linear transformation from hidden to output  
        x = self.fc2(x)
        
        return x
# =============================================================================
