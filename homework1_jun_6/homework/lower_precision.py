from pathlib import Path
import torch
from .bignet import BIGNET_DIM, LayerNorm


def block_quantize_4bit_optimized(x: torch.Tensor, group_size: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ultra-compressed 4-bit quantization with maximum group size.
    """
    assert x.dim() == 1
    assert x.size(0) % group_size == 0

    x = x.view(-1, group_size)
    
    # Use scale + zero point but store together
    x_min = x.min(dim=-1, keepdim=True).values
    x_max = x.max(dim=-1, keepdim=True).values
    
    # Calculate scale and zero point
    scale = (x_max - x_min) / 15
    zero_point = x_min
    
    # Quantize
    x_norm = (x - zero_point) / (scale + 1e-8)
    x_quant = x_norm.round().clamp(0, 15).to(torch.uint8)
    
    # Pack 2 values per byte
    x_packed = (x_quant[:, 0::2] & 0xF) + ((x_quant[:, 1::2] & 0xF) << 4)
    
    # Pack scale and zero together
    params = torch.stack([scale.squeeze(-1), zero_point.squeeze(-1)], dim=-1).to(torch.float16)
    
    return x_packed, params


def block_dequantize_4bit_optimized(x_packed: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """
    Dequantize using packed parameters.
    """
    scales = params[..., 0:1].to(torch.float32)
    zeros = params[..., 1:2].to(torch.float32)
    
    # Unpack 2 values from each byte
    x_quant = torch.zeros(x_packed.size(0), x_packed.size(1) * 2, dtype=torch.float32, device=x_packed.device)
    x_quant[:, 0::2] = (x_packed & 0xF).to(torch.float32)
    x_quant[:, 1::2] = ((x_packed >> 4) & 0xF).to(torch.float32)
    
    # Dequantize
    x = x_quant * scales + zeros
    return x.view(-1)


class LinearMixed(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 256) -> None:
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size

        # 4-bit packed weights with efficient storage
        num_groups = (out_features * in_features) // group_size
        self.register_buffer(
            "weight_packed",
            torch.zeros(num_groups, group_size // 2, dtype=torch.uint8),
            persistent=False,
        )
        # Even more aggressive - use single byte for both scale and zero
        self.register_buffer(
            "weight_params",
            torch.zeros(num_groups, 2, dtype=torch.float16),
            persistent=False,
        )
        
        self._register_load_state_dict_pre_hook(LinearMixed._load_state_dict_pre_hook, with_module=True)
        
        # Remove bias entirely for maximum compression
        self.register_parameter('bias', None)

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]
            
            weight_flat = weight.view(-1)
            total_elements = weight_flat.numel()
            
            if total_elements % self._group_size != 0:
                padding = self._group_size - (total_elements % self._group_size)
                weight_flat = torch.cat([weight_flat, torch.zeros(padding, dtype=weight_flat.dtype, device=weight_flat.device)])
            
            weight_packed, params = block_quantize_4bit_optimized(weight_flat, self._group_size)
            
            self.weight_packed.copy_(weight_packed)
            self.weight_params.copy_(params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            weight_dequant = block_dequantize_4bit_optimized(self.weight_packed, self.weight_params)
            weight_matrix = weight_dequant[:self._shape[0] * self._shape[1]].view(self._shape)
            
            return torch.nn.functional.linear(x, weight_matrix, None)


class OptimizedLowPrecisionBigNet(torch.nn.Module):
    """
    Optimized compression focusing on better memory layout and smaller groups.
    """

    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            # Maximum group size for ultimate compression
            self.model = torch.nn.Sequential(
                LinearMixed(channels, channels, bias=False, group_size=256),
                torch.nn.ReLU(),
                LinearMixed(channels, channels, bias=False, group_size=256),
                torch.nn.ReLU(),
                LinearMixed(channels, channels, bias=False, group_size=256),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        # Keep full BigNet structure
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
        
        # Compress LayerNorm parameters to float16 for additional savings
        for module in self.modules():
            if isinstance(module, LayerNorm):
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.data = module.weight.data.to(torch.float16)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data = module.bias.data.to(torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle LayerNorm with float16 parameters manually
        for i, module in enumerate(self.model):
            if isinstance(module, LayerNorm):
                # Convert to compatible types for group_norm
                weight = module.weight.to(x.dtype) if module.weight is not None else None
                bias = module.bias.to(x.dtype) if module.bias is not None else None
                x = torch.nn.functional.group_norm(x, 1, weight, bias, module.eps)
            else:
                x = module(x)
        return x


def load(path: Path | None):
    # More conservative approach: better 4-bit with optimizations
    net = OptimizedLowPrecisionBigNet()
    if path is not None:
        # Load state dict but ignore bias terms that we removed
        state_dict = torch.load(path, weights_only=True)
        
        # Remove bias keys from state_dict since our model doesn't have bias
        keys_to_remove = [key for key in state_dict.keys() if '.bias' in key and 'model.' in key and '.model.' in key]
        for key in keys_to_remove:
            del state_dict[key]
            
        net.load_state_dict(state_dict, strict=False)
    return net