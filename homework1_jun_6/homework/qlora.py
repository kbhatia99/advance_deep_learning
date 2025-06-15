from pathlib import Path
import torch
from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .low_precision import Linear4Bit

class QLoRALinear(Linear4Bit):
    def __init__(  # Fixed: was **init**
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        group_size: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias, group_size)
        
        # Implement LoRA on top of 4-bit quantized base layer
        # Keep the LoRA layers in float32 for training stability
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False)
        
        # Initialize LoRA weights properly
        # A initialized with random values, B initialized to zero (so initially LoRA contributes nothing)
        torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)  # Standard initialization for A
        torch.nn.init.zeros_(self.lora_b.weight)  # Zero initialization for B
        
        # Make sure the base layer (4-bit weights) are not trainable, but LoRA layers are
        # Don't call requires_grad_(False) on the whole module, just disable gradients for base weights
        for name, param in self.named_parameters():
            if 'lora' not in name:
                param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get the base 4-bit quantized layer output
        base_output = super().forward(x)
        
        # Compute LoRA adaptation: B @ A @ x (keeping in float32)
        # x should be in float32, and we want to keep LoRA computation in float32
        lora_output = self.lora_b(self.lora_a(x))
        
        # Add the LoRA adaptation to the base output
        return base_output + lora_output

class QLoRABigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels, lora_dim, group_size):  # Fixed: was **init**
            super().__init__()
            # Replace all Linear layers with QLoRALinear
            self.model = torch.nn.Sequential(
                QLoRALinear(channels, channels, lora_dim, group_size),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32, group_size: int = 16):  # Fixed: was **init**
        super().__init__()
        # Replicate the BigNet structure but with QLoRALinear layers
        # Keep LayerNorm in full precision for numerical stability
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def load(path: Path | None) -> QLoRABigNet:
    net = QLoRABigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net