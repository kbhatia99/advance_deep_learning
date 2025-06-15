from pathlib import Path
import torch
from .bignet import BIGNET_DIM, LayerNorm


def block_quantize_2bit(x: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Aggressive 2-bit quantization with large group sizes.
    """
    assert x.dim() == 1
    assert x.size(0) % group_size == 0

    x = x.view(-1, group_size)
    
    # Use percentile-based clipping to handle outliers better
    x_min = torch.quantile(x, 0.01, dim=-1, keepdim=True)
    x_max = torch.quantile(x, 0.99, dim=-1, keepdim=True)
    
    # Clamp values to reduce outlier impact
    x_clipped = torch.clamp(x, x_min, x_max)
    
    # Normalize to [0, 3] for 2-bit (4 values: 0, 1, 2, 3)
    x_norm = (x_clipped - x_min) / (x_max - x_min + 1e-8)
    x_quant = (x_norm * 3).round().to(torch.uint8)  # Use uint8 to pack more efficiently
    
    # Pack 4 values into 1 byte (4 * 2 bits = 8 bits)
    x_packed = (x_quant[:, 0::4] + 
                (x_quant[:, 1::4] << 2) + 
                (x_quant[:, 2::4] << 4) + 
                (x_quant[:, 3::4] << 6))
    
    # Store only min/max as single float16 values (more aggressive)
    ranges = torch.stack([x_min.squeeze(-1), x_max.squeeze(-1)], dim=-1).to(torch.float16)
    
    return x_packed.to(torch.uint8), ranges


def block_dequantize_2bit(x_packed: torch.Tensor, ranges: torch.Tensor) -> torch.Tensor:
    """
    Dequantize 2-bit packed values.
    """
    ranges = ranges.to(torch.float32)
    x_min = ranges[..., 0:1]
    x_max = ranges[..., 1:2]
    
    # Unpack 4 values from each byte
    x_quant = torch.zeros(x_packed.size(0), x_packed.size(1) * 4, dtype=torch.float32, device=x_packed.device)
    x_quant[:, 0::4] = (x_packed & 0x03).to(torch.float32)
    x_quant[:, 1::4] = ((x_packed >> 2) & 0x03).to(torch.float32)
    x_quant[:, 2::4] = ((x_packed >> 4) & 0x03).to(torch.float32)
    x_quant[:, 3::4] = ((x_packed >> 6) & 0x03).to(torch.float32)
    
    # Dequantize
    x_norm = x_quant / 3
    x = x_norm * (x_max - x_min) + x_min
    return x.view(-1)


class Linear2Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 128) -> None:
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size

        # 2-bit packed weights (4 values per byte)
        self.register_buffer(
            "weight_packed",
            torch.zeros(out_features * in_features // group_size, group_size // 4, dtype=torch.uint8),
            persistent=False,
        )
        # Minimal range storage
        self.register_buffer(
            "weight_ranges",
            torch.zeros(out_features * in_features // group_size, 2, dtype=torch.float16),
            persistent=False,
        )
        
        self._register_load_state_dict_pre_hook(Linear2Bit._load_state_dict_pre_hook, with_module=True)
        
        # Keep bias but in float16 to maintain compatibility
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float16))

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]
            
            # Quantize with very large group size
            weight_flat = weight.view(-1)
            total_elements = weight_flat.numel()
            
            if total_elements % self._group_size != 0:
                padding = self._group_size - (total_elements % self._group_size)
                weight_flat = torch.cat([weight_flat, torch.zeros(padding, dtype=weight_flat.dtype, device=weight_flat.device)])
            
            weight_packed, weight_ranges = block_quantize_2bit(weight_flat, self._group_size)
            
            self.weight_packed.copy_(weight_packed)
            self.weight_ranges.copy_(weight_ranges)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Dequantize weights
            weight_dequant = block_dequantize_2bit(self.weight_packed, self.weight_ranges)
            weight_matrix = weight_dequant[:self._shape[0] * self._shape[1]].view(self._shape)
            
            # Convert bias to float32 if it exists
            bias = self.bias.to(torch.float32) if self.bias is not None else None
            
            return torch.nn.functional.linear(x, weight_matrix, bias)


class UltraLowPrecisionBigNet(torch.nn.Module):
    """
    Ultra-aggressive compression BigNet targeting <9MB.
    """

    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            # Keep bias for compatibility, use large group size for compression
            self.model = torch.nn.Sequential(
                Linear2Bit(channels, channels, bias=True, group_size=256),
                torch.nn.ReLU(),
                Linear2Bit(channels, channels, bias=True, group_size=256),
                torch.nn.ReLU(),
                Linear2Bit(channels, channels, bias=True, group_size=256),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        # Keep the same structure as original BigNet for compatibility
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
        
        # Keep LayerNorm in original format for compatibility

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None):
    # Implement a BigNet that uses on average less than 4 bits per parameter (<9MB)
    net = UltraLowPrecisionBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net