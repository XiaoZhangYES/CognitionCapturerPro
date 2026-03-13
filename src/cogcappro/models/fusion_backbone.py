import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, CosineEmbeddingLoss
from PIL import Image
from typing import Optional, Tuple, Any, Union
from transformers import CLIPModel, GPT2LMHeadModel, AutoConfig

from .brain_backbone import PositionalEncoding, ResidualAdd



class SimpleFusionNetwork(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1024, modality_num=4, dropout_rate=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # New: independent projection layer for each modality (ensures gradient propagation)
        self.modality_proj = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(modality_num)
        ])
        # Fusion layer
        self.fusion = nn.Linear(input_dim * modality_num, output_dim)
        # Initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for proj in self.modality_proj:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)

    def forward(self, *modalities):
        # Apply trainable projection for each modality (construct gradient path)
        projected = [proj(feat) for proj, feat in zip(self.modality_proj, modalities)]
        # Concatenate and fuse
        x = torch.cat(projected, dim=1)
        # x = F.gelu(x)  # Non-linear activation, seems not used before
        x = self.fusion(x)
        return x


class CogcapFusion(nn.Module):
    """Multi-modal fusion module based on Cogcap model structure"""

    def __init__(self,
                 modal_dims=[1024, 1024, 1024, 1024],  # Input dimensions of four modalities
                 hidden_dim=512,  # Fusion intermediate dimension (consistent with Cogcap's proj_dim)
                 num_heads=1,  # Number of attention heads (following EEGAttention nhead design)
                 dropout=0.1):
        super().__init__()
        # 1. Single modality feature projection (follow Cogcap's Proj_eeg, unify modality dimensions)
        self.modal_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # Normalization consistent with Cogcap
                nn.GELU()
            ) for in_dim in modal_dims
        ])

        # 2. Modality positional encoding (follow Cogcap's PositionalEncoding, distinguish different modalities)
        self.modal_pos_encoder = PositionalEncoding(hidden_dim)  # Reuse Cogcap's positional encoding

        # 3. Cross-modal attention (follow Cogcap's EEGAttention, use Transformer encoder)
        self.cross_attention = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,  # Feedforward network dimension
            dropout=dropout,
            batch_first=True  # Set to True, input shape [batch, seq_len, dim]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.cross_attention,
            num_layers=2  # Two layers of encoder, balance capability and complexity
        )

        # 4. Fusion output projection (follow Cogcap's ResidualAdd and Proj_eeg)
        self.fusion_proj = nn.Sequential(
            ResidualAdd(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1024),  # 512→1024
            nn.LayerNorm(1024),  # Ensure output distribution stability
            nn.GELU()  # Maintain non-linear capability
        )

    def forward(self, *modal_features):
        """
        Input: Multiple modality features, e.g. (image_feat, text_feat, depth_feat, edge_feat)
        Each feature shape: [batch_size, feat_dim]
        Output: Fused features, shape: [batch_size, hidden_dim]
        """
        # Step 1: Single modality feature projection (unify dimensions and preprocess)
        projected = []
        for i, feat in enumerate(modal_features):
            # Each modality passes through independent projection layer, convert to hidden_dim
            proj_feat = self.modal_projs[i](feat)  # [batch, hidden_dim]
            projected.append(proj_feat)

        # Step 2: Stack modality features and add positional encoding (distinguish different modalities)
        # Stack as [batch, num_modals, hidden_dim] (num_modals=4)
        modal_stack = torch.stack(projected, dim=1)  # [batch, 4, hidden_dim]
        # Add modality positional encoding (follow Cogcap's temporal processing)
        modal_stack = self.modal_pos_encoder(modal_stack.permute(1, 0, 2)).permute(1, 0, 2)

        # Step 3: Cross-modal attention fusion
        attn_output = self.transformer_encoder(modal_stack)  # [batch, 4, hidden_dim]

        # Step 4: Aggregate multi-modal features (average or weighted, use average here)
        fused = attn_output.mean(dim=1)  # [batch, hidden_dim]

        # Step 5: Final projection, output fused features
        fused = self.fusion_proj(fused)  # [batch, hidden_dim]
        return fused
