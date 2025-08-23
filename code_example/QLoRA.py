import torch.nn as nn
import torch
import torch.nn.functional as F


class LoRALayer():
    """
    A mixin class that provides the basic attributes and properties for LoRA layers.
    This class is not meant to be used directly but to be inherited by other nn.Module classes.
    """
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        """
        Initializes the LoRALayer.

        Args:
            r (int): The rank of the low-rank approximation.
            lora_alpha (int): The scaling factor for the LoRA update.
            lora_dropout (float): The dropout probability for the LoRA layers.
            merge_weights (bool): Whether to merge the weights during evaluation.
        """
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            # If dropout is 0, use an identity function
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        
        

class Linear(nn.Linear, LoRALayer):
    """
    A LoRA-adapted linear layer that inherits from both nn.Linear and LoRALayer.

    This class replaces a standard nn.Linear layer and adds LoRA functionality.
    It freezes the original weight matrix and introduces two smaller, trainable matrices
    (lora_A and lora_B) to learn the updates.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        """
        Initializes the LoRA-adapted Linear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            r (int, optional): The rank of the LoRA matrices. If r = 0, LoRA is disabled. Defaults to 0.
            lora_alpha (int, optional): The scaling factor for LoRA. Defaults to 1.
            lora_dropout (float, optional): The dropout rate for LoRA. Defaults to 0.
            fan_in_fan_out (bool, optional): Set to True if the original weight matrix is stored
                                             in (in_features, out_features) format. Defaults to False.
            merge_weights (bool, optional): If True, merges the LoRA weights with the base weights
                                            during evaluation for efficiency. Defaults to True.
            **kwargs: Additional arguments for the nn.Linear constructor.
        """
        # Initialize the parent nn.Linear layer
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        # Initialize the LoRA-specific attributes
        LoRALayer.__init__(self, r, lora_alpha, lora_dropout, merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        
        # Actual trainable parameters:
        if r > 0:
            # Create LoRA matrices A and B as trainable parameters
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            # Calculate the scaling factor
            self.scaling = self.lora_alpha / self.r
            # Freeze the original weight matrix
            self.weight.requires_grad = False
        
        # Reset parameters to initialize lora_A and lora_B
        self.reset_parameters()
        if fan_in_fan_out:
            # Transpose the weight matrix if the format is (in_features, out_features)
            # to conform to PyTorch's standard (out_features, in_features)
            self.weight.data = self.weight.data.transpose(0, 1)
            
    def train(self, mode: bool = True):
        """
        Sets the layer in training mode. If weights were merged, it un-merges them.

        Args:
            mode (bool, optional): Whether to set training mode (True) or evaluation mode (False).
                                   Defaults to True.
        """
        # A helper function to handle potential weight transposition
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        # Set the training mode for the parent nn.Linear layer
        nn.Linear.train(self, mode)
        
        # If we are in training mode
        if mode:
            # If the weights were previously merged for evaluation, we need to un-merge them
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    # Subtract the LoRA delta weight (B @ A) from the main weight.
                    # Note: The weight matrix is always in (out, in) format internally,
                    # so we don't need the T() wrapper here.
                    self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        # If we are in evaluation mode
        else:
            # If we want to merge weights and they haven't been merged yet
            if self.merge_weights and not self.merged:
                # Merge the weights
                if self.r > 0:
                    # Add the LoRA delta weight (B @ A) to the main weight.
                    self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        """
        Performs the forward pass of the LoRA-adapted linear layer.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, out_features).
        """
        # A helper function to handle potential weight transposition for F.linear
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        # If LoRA is active and weights are not merged
        if self.r > 0 and not self.merged:
            # Standard linear layer computation
            result = F.linear(x, T(self.weight), bias=self.bias)
            
            # LoRA path computation and addition
            # 1. Apply dropout to input x
            # 2. Multiply by lora_A.T: (batch, in) @ (in, r) -> (batch, r)
            # 3. Multiply by lora_B.T: (batch, r) @ (r, out) -> (batch, out)
            # 4. Apply scaling
            lora_update = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            result += lora_update
            
            return result
        # If LoRA is disabled or weights are merged for efficient inference
        else:
            # Perform a standard linear operation
            return F.linear(x, T(self.weight), bias=self.bias)

