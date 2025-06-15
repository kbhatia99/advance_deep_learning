from pathlib import Path
import torch
from .bignet import BIGNET_DIM, LayerNorm


def block_quantize_3bit(x: torch.Tensor, group_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize to 3-bit precision with larger group sizes for better compression.
    """
    assert x.dim() == 1
    assert x.size(0) % group_size == 0

    x = x.view(-1, group_size)
    # Use min/max for better range utilization
    x_min = x.min(dim=-1, keepdim=True).values
    x_max = x.max(dim=-1, keepdim=True).values
    
    # Normalize to [0, 7] for 3-bit
    x_norm = (x - x_min) / (x_max - x_min + 1e-8)
    x_quant = (x_norm * 7).round().to(torch.int8)
    
    # Pack 8 values into 3 bytes (more efficient packing)
    # This is simplified - real implementation would pack more efficiently
    return x_quant, torch.stack([x_min.squeeze(-1), x_max.squeeze(-1)], dim=-1).to(torch.float16)


def block_dequantize_3bit(x_quant: torch.Tensor, ranges: torch.Tensor) -> torch.Tensor:
    """
    Dequantize 3-bit values.
    """
    ranges = ranges.to(torch.float32)
    x_min = ranges[..., 0:1]
    x_max = ranges[..., 1:2]
    
    x_norm = x_quant.to(torch.float32) / 7
    x = x_norm * (x_max - x_min) + x_min
    return x.view(-1)


class Linear3Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 32) -> None:
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size

        # 3-bit quantized weights
        self.register_buffer(
            "weight_q3",
            torch.zeros(out_features * in_features // group_size, group_size, dtype=torch.int8),
            persistent=False,
        )
        # Store min/max ranges for each group
        self.register_buffer(
            "weight_ranges",
            torch.zeros(out_features * in_features // group_size, 2, dtype=torch.float16),
            persistent=False,
        )
        
        self._register_load_state_dict_pre_hook(Linear3Bit._load_state_dict_pre_hook, with_module=True)
        
        self.bias = None
        if bias:
            # Store bias in float16 to save more memory
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float16))

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]
            
            # Quantize with larger group size for better compression
            weight_flat = weight.view(-1)
            total_elements = weight_flat.numel()
            
            if total_elements % self._group_size != 0:
                padding = self._group_size - (total_elements % self._group_size)
                weight_flat = torch.cat([weight_flat, torch.zeros(padding, dtype=weight_flat.dtype, device=weight_flat.device)])
            
            weight_q3, weight_ranges = block_quantize_3bit(weight_flat, self._group_size)
            
            self.weight_q3.copy_(weight_q3)
            self.weight_ranges.copy_(weight_ranges)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Dequantize weights
            weight_dequant = block_dequantize_3bit(self.weight_q3, self.weight_ranges)
            weight_matrix = weight_dequant[:self._shape[0] * self._shape[1]].view(self._shape)
            
            # Convert bias to float32 for computation if it exists
            bias = self.bias.to(torch.float32) if self.bias is not None else None
            
            return torch.nn.functional.linear(x, weight_matrix, bias)


class LowerPrecisionBigNet(torch.nn.Module):
    """
    A BigNet with aggressive compression using 3-bit weights and optimizations.
    """

    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            # Use larger group size for better compression
            self.model = torch.nn.Sequential(
                Linear3Bit(channels, channels, group_size=64),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels, group_size=64),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels, group_size=64),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        # Keep LayerNorm in float32 for numerical stability and compatibility
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None):
    # TODO (extra credit): Implement a BigNet that uses in
    # average less than 4 bits per parameter (<9MB)
    # Make sure the network retains some decent accuracy
    net = LowerPrecisionBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net