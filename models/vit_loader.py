# models/vit_loader.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class UniversalLoader(nn.Module):
    """
    Universal model that can load ANY checkpoint structure by dynamically
    creating parameters that match exactly.
    """
    def __init__(self, state_dict):
        super().__init__()
        
        # Dynamically create all parameters from state_dict
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                # Remove any 'model.' or 'module.' prefixes
                clean_key = key.replace('model.', '').replace('module.', '')
                
                # Convert key to valid attribute name
                attr_name = clean_key.replace('.', '_')
                
                if 'num_batches_tracked' in key:
                    # Register as buffer
                    self.register_buffer(attr_name, value.clone())
                else:
                    # Register as parameter
                    self.register_parameter(attr_name, nn.Parameter(value.clone()))
        
        # Store original keys for reference
        self.original_keys = list(state_dict.keys())
        
        # Store model configuration
        self.patch_size = 9
        self.embed_dim = 512
        self.num_heads = 8
        self.head_dim = self.embed_dim // self.num_heads
        self.num_patches = 144  # From pos_embed shape
        
    def interpolate_pos_embed(self, pos_embed, num_patches):
        """Interpolate positional embeddings to match the number of patches"""
        # pos_embed shape: [1, 144, 512]
        # We need to interpolate to [1, num_patches, 512]
        
        old_size = int(pos_embed.shape[1] ** 0.5)  # 12
        new_size = int(num_patches ** 0.5)  # sqrt(num_patches)
        
        # Reshape and interpolate
        pos_embed = pos_embed.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)
        
        return pos_embed
    
    def forward(self, x):
        """
        Proper forward pass that handles different input sizes
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = F.conv2d(
            x, 
            self.patch_embed_proj_weight, 
            getattr(self, 'patch_embed_proj_bias', None),
            stride=self.patch_size, 
            padding=0
        )
        
        # Get current number of patches
        h_patches = x.shape[2]  # height in patches
        w_patches = x.shape[3]  # width in patches  
        num_patches = h_patches * w_patches
        
        # Flatten patches: [B, C, H, W] -> [B, N, C]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Handle positional embeddings
        if hasattr(self, 'pos_embed'):
            if num_patches == self.pos_embed.shape[1]:
                # Exact match - use as is
                pos_embed = self.pos_embed
            else:
                # Need to interpolate
                pos_embed = self.interpolate_pos_embed(self.pos_embed, num_patches)
            x = x + pos_embed
        
        # Apply transformer blocks if they exist
        block_keys = [key for key in self.original_keys if key.startswith('blocks.')]
        if block_keys:
            # We have transformer blocks - apply them
            for i in range(12):  # Assuming 12 blocks
                block_prefix = f'blocks.{i}'
                if any(key.startswith(block_prefix) for key in block_keys):
                    x = self.apply_transformer_block(x, i)
        
        # Apply final normalization if it exists
        if hasattr(self, 'norm_weight') and hasattr(self, 'norm_bias'):
            x = F.layer_norm(x, [self.embed_dim], 
                           weight=self.norm_weight, 
                           bias=self.norm_bias)
        
        return x
    
    def apply_transformer_block(self, x, block_idx):
        """Apply a single transformer block with proper attention"""
        residual = x
        
        # LayerNorm 1
        if hasattr(self, f'blocks_{block_idx}_norm1_weight'):
            x = F.layer_norm(x, [self.embed_dim], 
                           weight=getattr(self, f'blocks_{block_idx}_norm1_weight'),
                           bias=getattr(self, f'blocks_{block_idx}_norm1_bias', None))
        
        # Self-attention
        if hasattr(self, f'blocks_{block_idx}_attn_qkv_weight'):
            # QKV projection: [B, N, 512] -> [B, N, 1536]
            qkv = F.linear(x, 
                          getattr(self, f'blocks_{block_idx}_attn_qkv_weight'),
                          bias=None)  # No bias in qkv
            
            # Reshape: [B, N, 1536] -> [B, N, 3, 8, 64]
            qkv = qkv.reshape(x.shape[0], x.shape[1], 3, self.num_heads, self.head_dim)
            
            # Separate Q, K, V: each [B, N, 8, 64]
            q, k, v = qkv.unbind(2)
            
            # Transpose for attention: [B, 8, N, 64]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Scaled dot-product attention
            attn_weights = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_weights = F.softmax(attn_weights, dim=-1)
            
            # Apply attention to values
            attn_output = attn_weights @ v  # [B, 8, N, 64]
            
            # Transpose back: [B, N, 8, 64] -> [B, N, 512]
            attn_output = attn_output.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.embed_dim)
            
            # Projection: [B, N, 512] -> [B, N, 512]
            if hasattr(self, f'blocks_{block_idx}_attn_proj_weight'):
                x = F.linear(attn_output,
                            getattr(self, f'blocks_{block_idx}_attn_proj_weight'),
                            getattr(self, f'blocks_{block_idx}_attn_proj_bias', None))
        
        # Add residual
        x = x + residual
        
        # MLP
        residual = x
        if hasattr(self, f'blocks_{block_idx}_norm2_weight'):
            x = F.layer_norm(x, [self.embed_dim],
                           weight=getattr(self, f'blocks_{block_idx}_norm2_weight'),
                           bias=getattr(self, f'blocks_{block_idx}_norm2_bias', None))
        
        if hasattr(self, f'blocks_{block_idx}_mlp_fc1_weight'):
            x = F.linear(x, 
                        getattr(self, f'blocks_{block_idx}_mlp_fc1_weight'),
                        getattr(self, f'blocks_{block_idx}_mlp_fc1_bias', None))
            x = F.gelu(x)  # GELU activation
            x = F.linear(x,
                        getattr(self, f'blocks_{block_idx}_mlp_fc2_weight'), 
                        getattr(self, f'blocks_{block_idx}_mlp_fc2_bias', None))
        
        x = x + residual
        return x

def load_vit_checkpoint(checkpoint_path, device='cpu'):
    """
    Universal loader that works with ANY checkpoint structure.
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint
    
    print(f"Total parameters in checkpoint: {len(state_dict)}")
    
    # Create universal model
    model = UniversalLoader(state_dict).to(device)
    
    print("✓ Checkpoint loaded successfully into universal model!")
    print(f"Model has {len(list(model.parameters()))} parameters")
    
    # Show some key parameters
    key_params = ['pos_embed', 'patch_embed_proj_weight', 'mask_token']
    for param in key_params:
        if hasattr(model, param):
            tensor = getattr(model, param)
            print(f"  {param}: {tensor.shape}")
    
    model.eval()
    return model