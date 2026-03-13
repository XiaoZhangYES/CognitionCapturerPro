import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor
import os
import logging
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import math


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, use_channel_attn=False, channel_num=17, dropout = 0.1): # use_ori = False
        super().__init__()
        # revised from shallownet
        self.use_channel_attn = use_channel_attn
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)), # (batch, 40, 63, 226)
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        if use_channel_attn is True:
            self.channel_length = [7, 3, 4, 5, 3, 4, 3, 3, 3, 4, 3, 6, 4, 3, 8]
            self.func_area = [
                [0, 1, 2, 3, 4, 5, 6],
                [7, 8, 9], [15, 16, 17, 18],
                [10, 11, 19, 20, 21], [12, 13, 14],
                [22, 23, 24, 25], [26, 27, 28],
                [29, 30, 31], [32, 33, 34], [35, 36, 37, 38],
                [46, 47, 48], [39, 40, 41, 49, 50, 51], [42, 43, 44, 45],
                [52, 53, 54], [56, 57, 58, 60, 61, 62, 55, 59],
            ]  # 17 regions
            self.blocks = nn.ModuleList(ChannelConv(channel=channel_num) for i, channel_num in enumerate(self.channel_length))
            self.sumChannelConv = ChannelConv(channel=17)
        else:
            self.tsconv = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), (1, 1)),
                nn.AvgPool2d((1, 51), (1, 5)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Conv2d(40, 40, (channel_num, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Dropout(dropout),
            )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)  # b, 1, 63, 250 = x.shape
        if self.use_channel_attn is True:
            x = self.temporalconv(x) # batch, 40, 63, 36
            x_new = []
            for i, area in enumerate(self.func_area):
                x_new.append([])
                print(len(area))
                for j, element in enumerate(area):
                    x_new[i].append(x[:, :, self.func_area[i][j], :])
                x_new[i] = torch.stack(x_new[i], dim=2)
            del x
            for i, blk in enumerate(self.blocks):
                x_new[i] = blk(x_new[i])
            x_new = torch.cat(x_new, dim=2)
            x = self.sumChannelConv(x_new)
        else:
            x = self.tsconv(x)
        x = self.projection(x)
        return x
    
class ChannelConv(nn.Module):
    def __init__(self, channel=None, dropout = 0.1):
        super().__init__()
        # revised from shallownet
        self.channelconv = nn.Sequential(
            nn.Conv2d(40, 40, (channel, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.channelconv(x)

class EEGAttention(nn.Module):
    def __init__(self, channel, d_model, nhead):
        super(EEGAttention, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.channel = channel
        self.d_model = d_model

    def forward(self, src):
        # Or add pos_encoder after tokenization?
        src = src.permute(2, 0, 1)  # Change shape to [time_length, batch_size, channel]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output.permute(1, 2, 0)  # Change shape back to [batch_size, channel, time_length]

class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, channel_num=63, dropout=0.1):
        super().__init__(
            PatchEmbedding(emb_size, channel_num=channel_num, dropout=dropout),
            FlattenHead()
        )

class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_out=0.1, data_type='eeg'):
        # Set embedding_dim based on data type
        if data_type == 'eeg':
            embedding_dim = 1440
        elif data_type == 'meg':
            embedding_dim = 1040
            
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_out),
            )),
            nn.LayerNorm(proj_dim),
        )
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Fix: ensure div_term length correctly handles odd and even d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Fix: correctly handles odd and even dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])  # For odd dimensions, cos part has one less element

        self.register_buffer('pe', pe) # After register_buffer, not treated as parameters

    def forward(self, x):
        pe = self.pe[:x.size(0), :].unsqueeze(1).repeat(1, x.size(1), 1).to(x.device)
        x = x + pe
        return x

class Cogcap(nn.Module):
    """
    revise from ATM
    """
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=10, num_features=64, num_latents=1024,
                 num_blocks=1, dropout=0.1, data_type='eeg'):
        super(Cogcap, self).__init__()
        # self.regionmodule = ResidualAdd(
        #     nn.Sequential(
        #         EEG_GAT(),
        #         nn.Dropout(0.1),
        #     )
        # )
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)
        self.subject_wise_linear = nn.ModuleList(
            [nn.Linear(sequence_length, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg(channel_num=num_channels, dropout=dropout)
        self.proj_eeg = Proj_eeg(drop_out=dropout, data_type=data_type)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x): # todo
        # x = self.regionmodule(x)
        x = self.attention_model(x)
        x = self.subject_wise_linear[0](x) # how to deal with this
        eeg_embedding = self.enc_eeg(x)
        out = self.proj_eeg(eeg_embedding)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return  x + self.f(x)


class EEGProjectLayer_multimodal_cogcap_list(nn.Module):
    def __init__(self, z_dim, c_num, timesteps, modality_num=4, drop_proj=0.3, fusion=False, data_type='eeg'): # todo add dropout
        super(EEGProjectLayer_multimodal_cogcap_list, self).__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps
        self.modality_num = 5 if fusion else 4 # todo modality_num

        self.input_dim = self.c_num * (self.timesteps[1] - self.timesteps[0])

        self.models = nn.ModuleList([
            Cogcap(num_channels=c_num, sequence_length=timesteps[1], num_subjects=1, dropout=drop_proj, data_type=data_type) for _ in range(self.modality_num)
        ])
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, x):
        x_copies = [x.clone() for _ in range(self.modality_num)]
        outputs = [model(x_copy) for model, x_copy in zip(self.models, x_copies)]
        return outputs

class ProjMod_multimodal(nn.Module):
    def __init__(self, drop_proj=0.3, fusion=False): # todo add dropout
        super().__init__()
        self.modality_num = 5 if fusion else 4 # todo modality_num
        if fusion is True:
            self.modality_names = ['image', 'text', 'depth', 'edge', 'fusion']
        else:
            self.modality_names = ['image', 'text', 'depth', 'edge']

        self.models = nn.ModuleList([
            ProjMod(embedding_dim=1024, proj_dim=1024, drop_proj=drop_proj) for _ in range(self.modality_num)
        ])

    def forward(self, *modal_features):
        """
        Receive multiple modality features as input, each corresponding to one projection model

        Args:
            *modal_features: Variable number of modality features, order should match modality_names
                            For example: (image_feat, text_feat, depth_feat, edge_feat)

        Returns:
            outputs: Dictionary with modality names as keys (e.g. 'image') and projected features as values
        """
        # Check if the number of input modalities matches the defined number
        assert len(modal_features) == self.modality_num, \
            f"Number of input modalities ({len(modal_features)}) does not match defined number ({self.modality_num})"

        # Apply corresponding projection model to each modality feature
        outputs = {}
        for name, model, feat in zip(self.modality_names, self.models, modal_features):
            outputs[name] = model(feat)  # Each modality feature is processed through its own projection model

        return outputs

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x

class BaseModel(nn.Module):
    def __init__(self,  z_dim, c_num, timesteps, embedding_dim = 1440):
        super(BaseModel, self).__init__()

        self.backbone = None
        self.project = nn.Sequential(
            FlattenHead(),
            nn.Linear(embedding_dim, z_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(z_dim, z_dim),
                nn.Dropout(0.5))),
            nn.LayerNorm(z_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = self.project(x)
        return x

class Shallownet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), (1, 1)),
                nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.AvgPool2d((1, 51), (1, 5)),
                nn.Dropout(0.5),
            )

class Deepnet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps,embedding_dim = 1400)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 25, (1, 10), (1, 1)),
                nn.Conv2d(25, 25, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(25),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),

                nn.Conv2d(25, 50, (1, 10), (1, 1)),
                nn.BatchNorm2d(50),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),

                nn.Conv2d(50, 100, (1, 10), (1, 1)),
                nn.BatchNorm2d(100),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),

                nn.Conv2d(100, 200, (1, 10), (1, 1)),
                nn.BatchNorm2d(200),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),
            )

class EEGnet(BaseModel):
    def __init__(self,  z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps, embedding_dim = 1248)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 8, (1, 64), (1, 1)),
                nn.BatchNorm2d(8),
                nn.Conv2d(8, 16, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.AvgPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),
                nn.Conv2d(16, 16, (1, 16), (1, 1)),
                nn.BatchNorm2d(16),
                nn.ELU(),
                # nn.AvgPool2d((1, 2), (1, 2)),
                nn.Dropout2d(0.5)
            )

class TSconv(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), (1, 1)),
                nn.AvgPool2d((1, 51), (1, 5)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Dropout(0.5),
            )

class ProjMod(nn.Sequential):
    
    def __init__(self, embedding_dim=1024, proj_dim=1024, drop_proj=0.1):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x):
        return x

class ShallowNet_multimodal_list(nn.Module):
    """
    Multi-modal version of ShallowNet, output format consistent with EEGProjectLayer_multimodal_cogcap_list
    """
    def __init__(self, z_dim, c_num, timesteps, modality_num=4, drop_proj=0.3, fusion=False):
        super(ShallowNet_multimodal_list, self).__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps
        self.modality_num = 5 if fusion else 4  # 5 modalities if fusion included, otherwise 4

        # Create multiple Shallownet instances
        self.models = nn.ModuleList([
            Shallownet(z_dim, c_num, timesteps) for _ in range(self.modality_num)
        ])
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, x):
        x_copies = [x.clone() for _ in range(self.modality_num)]
        outputs = [model(x_copy) for model, x_copy in zip(self.models, x_copies)]
        return outputs

class DeepNet_multimodal_list(nn.Module):
    """
    Multi-modal version of DeepNet, output format consistent with EEGProjectLayer_multimodal_cogcap_list
    """
    def __init__(self, z_dim, c_num, timesteps, modality_num=4, drop_proj=0.3, fusion=False):
        super(DeepNet_multimodal_list, self).__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps
        self.modality_num = 5 if fusion else 4  # 5 modalities if fusion included, otherwise 4

        # Create multiple Deepnet instances
        self.models = nn.ModuleList([
            Deepnet(z_dim, c_num, timesteps) for _ in range(self.modality_num)
        ])
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, x):
        x_copies = [x.clone() for _ in range(self.modality_num)]
        outputs = [model(x_copy) for model, x_copy in zip(self.models, x_copies)]
        return outputs

class EEGNet_multimodal_list(nn.Module):
    """
    Multi-modal version of EEGNet, output format consistent with EEGProjectLayer_multimodal_cogcap_list
    """
    def __init__(self, z_dim, c_num, timesteps, modality_num=4, drop_proj=0.3, fusion=False):
        super(EEGNet_multimodal_list, self).__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps
        self.modality_num = 5 if fusion else 4  # 5 modalities if fusion included, otherwise 4

        # Create multiple EEGnet instances
        self.models = nn.ModuleList([
            EEGnet(z_dim, c_num, timesteps) for _ in range(self.modality_num)
        ])
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, x):
        x_copies = [x.clone() for _ in range(self.modality_num)]
        outputs = [model(x_copy) for model, x_copy in zip(self.models, x_copies)]
        return outputs

class TSConv_multimodal_list(nn.Module):
    """
    Multi-modal version of TSConv, output format consistent with EEGProjectLayer_multimodal_cogcap_list
    """
    def __init__(self, z_dim, c_num, timesteps, modality_num=4, drop_proj=0.3, fusion=False):
        super(TSConv_multimodal_list, self).__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps
        self.modality_num = 5 if fusion else 4  # 5 modalities if fusion included, otherwise 4

        # Create multiple TSconv instances
        self.models = nn.ModuleList([
            TSconv(z_dim, c_num, timesteps) for _ in range(self.modality_num)
        ])
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, x):
        x_copies = [x.clone() for _ in range(self.modality_num)]
        outputs = [model(x_copy) for model, x_copy in zip(self.models, x_copies)]
        return outputs

if __name__ == '__main__':
    # Test EEG data
    # eeg_model = EEGProjectLayer_multimodal_cogcap_list(z_dim=1024, c_num=63, timesteps=[0, 250], data_type='eeg').to('cuda')
    # eeg_x = torch.randn(32, 63, 250).to('cuda')
    # eeg_y = eeg_model(eeg_x)
    # print("EEG output shape:", eeg_y[0].shape)
    
    # Test MEG data
    meg_model = EEGProjectLayer_multimodal_cogcap_list(z_dim=1024, c_num=271, timesteps=[0, 201], data_type='meg').to('cuda')
    meg_x = torch.randn(1024, 271, 201).to('cuda')
    meg_y = meg_model(meg_x)
    print("MEG output shape:", meg_y[0].shape)