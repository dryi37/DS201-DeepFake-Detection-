import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class DCT_HF(nn.Module):
    def __init__(self, img_size=224, block_size=16, overlap=12, keep_ratio=24/256):
        super().__init__()
        self.img_size = img_size
        self.block = block_size
        self.stride = block_size - overlap
        self.keep = max(1, int(block_size * block_size * keep_ratio))

        # Zigzag 
        zz = np.zeros((block_size, block_size), dtype=np.int64)
        idx=0
        for s in range(2*block_size-1):
            if s%2==0:
                i=min(s,block_size-1); j=s-i
                while i>=0 and j<block_size:
                    zz[i,j]=idx; idx+=1; i-=1; j+=1
            else:
                j=min(s,block_size-1); i=s-j
                while j>=0 and i<block_size:
                    zz[i,j]=idx; idx+=1; i+=1; j-=1

        zig = torch.from_numpy(zz.reshape(-1))
        self.register_buffer("zigzag_order", zig)
        self.register_buffer("reverse_order", torch.argsort(zig))

        # HF mask 
        mask = torch.zeros(block_size*block_size)
        mask[-self.keep:] = 1.0
        self.register_buffer("hf_mask", mask.view(1,1,-1))  # (1,1,L)

    # RGB→Y 
    def _luma(self,x):
        if x.size(1)==3:
            return 0.299*x[:,0:1]+0.587*x[:,1:2]+0.114*x[:,2:3]
        return x.mean(1,keepdim=True)

    # Core Single-scale (img_size) 
    def _extract(self,x):
        B,_,H,W = x.shape
        assert H==W==self.img_size

        # unfold patches
        patches = x.unfold(2,self.block,self.stride).unfold(3,self.block,self.stride)
        B,_,Ph,Pw,_,_ = patches.shape
        patches = patches.contiguous().reshape(B,Ph*Pw,self.block,self.block)
        N = Ph*Pw ; L=self.block*self.block

        # FFT2
        freq = torch.fft.fft2(patches,norm="ortho")
        fr = freq.real.reshape(B,N,L)
        fi = freq.imag.reshape(B,N,L)

        # Zigzag reorder + HF mask
        zig = self.zigzag_order.view(1,1,L)
        fr = fr.gather(2,zig.expand(B,N,-1))*self.hf_mask
        fi = fi.gather(2,zig.expand(B,N,-1))*self.hf_mask

        # reverse reorder
        rev = self.reverse_order.view(1,1,L)
        fr = fr.gather(2,rev.expand(B,N,-1)).reshape(B,N,self.block,self.block)
        fi = fi.gather(2,rev.expand(B,N,-1)).reshape(B,N,self.block,self.block)

        # iFFT → HF patches
        res = torch.abs(torch.fft.ifft2(torch.complex(fr,fi), norm="ortho").real)

        # fold back
        res_flat = res.reshape(B,N,L).permute(0,2,1).contiguous()
        out = F.fold(
            res_flat,
            output_size=(self.img_size,self.img_size),
            kernel_size=self.block,
            stride=self.stride
        )

        # norm overlap
        ones = torch.ones_like(x)
        ones_p = ones.unfold(2,self.block,self.stride).unfold(3,self.block,self.stride)
        ones_p = ones_p.reshape(B,-1,L).permute(0,2,1).contiguous()
        norm = F.fold(
            ones_p,
            output_size=(self.img_size,self.img_size),
            kernel_size=self.block,
            stride=self.stride
        )

        out = out/(norm+1e-6)

        # L2 normalize per image
        out = out/(out.view(B,-1).norm(2,dim=1,keepdim=True).view(B,1,1,1)+1e-6)

        return out   # B,1,H,W

    # forward 
    def forward(self,x):
        B,C,H,W = x.shape
        assert H==W==self.img_size, f"input must be {self.img_size}x{self.img_size}"
        y = self._luma(x)
        return self._extract(y)    # B,1,img,img

    
class EfficientNetB0_1ch(nn.Module):
    def __init__(self):
        super().__init__()

        model = efficientnet_b0(weights=None)

        old_conv = model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        model.features[0][0] = new_conv

        self.features = model.features      # backbone
        self.avgpool  = model.avgpool       # global avg pooling
        self.out_dim  = model.classifier[1].in_features  # feature size (1280)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)     # (B, 1280, 1, 1)
        x = torch.flatten(x, 1) # (B, 1280)
        return x
    
class HF_seq(nn.Module):
    def __init__(self, img_size=224):
        super().__init__()
        self.hf = DCT_HF(img_size=img_size)
        self.enc = EfficientNetB0_1ch()

    def forward(self, video): 
        video = video.permute(0,2,1,3,4)
        B,T,C,H,W = video.shape
        video = video.reshape(B*T, C, H, W)          # merge batch & time

        hf = self.hf(video)                          # (B*T,1,H,W)
        feat = self.enc(hf)                          # (B*T,1280)

        return feat.reshape(B,T,1280)
    
class EfficientNetB0_3ch(nn.Module):
    def __init__(self):
        super().__init__()

        model = efficientnet_b0(weights=None)

        self.features = model.features       # backbone
        self.avgpool  = model.avgpool        # global avg pooling
        self.out_dim  = model.classifier[1].in_features  # 1280

    def forward(self, x):
        x = self.features(x)      # (B, 1280, H, W)
        x = self.avgpool(x)       # (B, 1280, 1, 1)
        x = torch.flatten(x, 1)   # (B, 1280)
        return x
    
class RGB_seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = EfficientNetB0_3ch()

    def forward(self, video):
        video = video.permute(0,2,1,3,4)
        B,T,C,H,W = video.shape
        video = video.reshape(B*T, C, H, W)          # merge batch & time

        feat = self.enc(video)                          # (B*T,1280)

        return feat.reshape(B,T,1280)
    
class CoAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, drop=0.1):
        super().__init__()
        self.rgb_attn = nn.MultiheadAttention(d_model, num_heads, dropout=drop, batch_first=True)
        self.hf_attn  = nn.MultiheadAttention(d_model, num_heads, dropout=drop, batch_first=True)

        self.hf_norm = nn.LayerNorm(d_model)
        self.rgb_norm = nn.LayerNorm(d_model)

    def forward(self, rgb_seq, hf_seq):
        rgb_norm = self.rgb_norm(rgb_seq)
        hf_norm  = self.hf_norm(hf_seq)

        rgb_att, _ = self.rgb_attn(query=rgb_norm, key=hf_norm, value=hf_norm)
        hf_att, _ = self.hf_attn(query=hf_norm, key=rgb_norm, value=rgb_norm)

        rgb_seq = rgb_seq + rgb_att
        hf_seq = hf_seq + hf_att

        return rgb_seq, hf_seq
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)        # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (B, T, d_model)
        """
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)
    
class DeepFake_Final_v2(nn.Module):
    def __init__(self, num_classes=2, d_model=1280, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()

        self.rgb = RGB_seq()
        self.hf = HF_seq(224)

        self.co_attn = CoAttention(1280, num_heads=num_heads, drop=dropout)

        self.pos_encoder = PositionalEncoding(d_model, max_len=500, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,      # để dùng shape (B, T, D)
        )

        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head (sau khi pool theo thời gian)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, video):
        """
        video: (B, C, T, H, W)
        """
        # Extract feature theo frame
        rgb_seq = self.rgb(video)   # (B, T, 1280)
        hf_seq  = self.hf(video)    # (B, T, 1280)

        # Co-Attention giữa 2 modality
        rgb_seq, hf_seq = self.co_attn(rgb_seq, hf_seq)  # vẫn (B, T, 1280)

        fused = (rgb_seq + hf_seq) * 0.5    # (B, T, 1280)
        x = self.pos_encoder(fused)         # (B, T, 1280)
        x = self.temporal_encoder(x)        # (B, T, 1280)
        x = x.mean(dim=1)                   # (B, 1280)

        # Classification
        logits = self.cls_head(x)           # (B, num_classes)

        return logits