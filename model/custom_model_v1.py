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


class EfficientnetB0(nn.Module):
    def __init__(self, pretrained=True, in_channels=1, out_dim=1280):
        super().__init__()
        self.model = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )

        conv = self.model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=False,
        )

        # Copy weight but convert RGB->Gray mean if pretrained
        if pretrained:
            new_conv.weight.data = conv.weight.data.mean(dim=1, keepdim=True)

        self.model.features[0][0] = new_conv

        # Remove classifier → dùng như feature extractor
        self.feature_dim = out_dim     # EfficientNet-B0 output = 1280
        self.pool = nn.AdaptiveAvgPool2d(1)  # (B,1280,1,1) → (B,1280)

    def forward(self, x):
        x = self.model.features(x)       # B,1280,7,7
        x = self.pool(x).flatten(1)      # B,1280
        return x
    
class HF_Seq(nn.Module):
    def __init__(self, img_size=224):
        super().__init__()
        self.hf = DCT_HF(img_size=img_size)
        self.enc = EfficientnetB0(pretrained=False, in_channels=1, out_dim=1280)

    def forward(self, video): 
        B,T,C,H,W = video.shape
        video = video.reshape(B*T, C, H, W)          # merge batch & time

        hf = self.hf(video)                          # (B*T,1,H,W)
        feat = self.enc(hf)                          # (B*T,1280)

        return feat.reshape(B,T,1280)               # restore sequence


class R2plus1D_Block(nn.Module):
    def __init__(self, in_c, out_c, stride=(1,2,2), use_se=True):
        super().__init__()

        # spatial 2D conv
        self.spatial = nn.Conv3d(in_c, out_c, kernel_size=(1,3,3),
                                stride=stride, padding=(0,1,1), bias=False)
        self.bn1 = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU(inplace=True)

        # temporal conv
        self.temporal = nn.Conv3d(out_c, out_c, kernel_size=(3,1,1),
                                 stride=1, padding=(1,0,0), bias=False)
        self.bn2 = nn.BatchNorm3d(out_c)

        # Squeeze-Excitation tăng tín hiệu artifact
        self.se = SEBlock(out_c) if use_se else nn.Identity()

        # residual shortcut
        self.short = nn.Sequential()
        if in_c != out_c or stride != (1,1,1):
            self.short = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_c)
            )

    def forward(self, x):
        identity = self.short(x)

        out = self.relu(self.bn1(self.spatial(x)))
        out = self.bn2(self.temporal(out))
        out = self.se(out)

        return self.relu(out + identity)
    
    
class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(ch, ch//r, 1),
            nn.ReLU(),
            nn.Conv3d(ch//r, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w
    
class S3DCNN(nn.Module):
    def __init__(self, in_channels=3, base=32):
        super().__init__()

        self.layer1 = R2plus1D_Block(in_channels, base, stride=(1,1,1))   # 32
        self.layer2 = R2plus1D_Block(base, base*2)                        # 64
        self.layer3 = R2plus1D_Block(base*2, base*4)                      # 128
        self.layer4 = R2plus1D_Block(base*4, base*8)                      # 256
        self.layer5 = R2plus1D_Block(base*8, base*16)                     # 512

        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

    def forward(self, x):
        x = self.layer1(x)        # (B,32,T,H,W)
        x = self.layer2(x)        # (B,64,T,H/2,W/2)
        x = self.layer3(x)        # (B,128,T,H/4,W/4)
        x = self.layer4(x)        # (B,256,T,H/4,W/4)
        x = self.layer5(x)        # (B,512,T,H/8,W/8)

        x = self.spatial_pool(x)  # → (B,512,T,1,1)
        x = x.squeeze(-1).squeeze(-1)  # → (B,512,T)
        x = x.permute(0,2,1)      # → (B,T,512)

        return x
    
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
    
class FusionHead(nn.Module):
    def __init__(self, rgb_dim, hf_dim, d_model, num_classes, co_layers, trans_layers, nhead, drop, max_len):
        super().__init__()

        self.rgb_proj = nn.Linear(rgb_dim, d_model)
        self.hf_proj  = nn.Linear(hf_dim, d_model)

         # learnable temporal position embedding
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        # Co-Attention stacks
        self.co_blocks = nn.ModuleList([
            CoAttention(d_model, nhead, drop)
            for _ in range(co_layers)
        ])

        # self.ln_rgb = nn.LayerNorm(d_model)
        # self.ln_hf  = nn.LayerNorm(d_model)

        # Transformer Fusion layer
        encoder = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=drop, activation="gelu",
            batch_first=True, norm_first=True   # Pre-Norm
        )
        self.transformer = nn.TransformerEncoder(
            encoder, num_layers=trans_layers
        )

        self.temp_att = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Linear(d_model//2, 1)   # score từng frame
        )

        self.norm = nn.LayerNorm(d_model)
        self.cls = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, rgb_seq, hf_seq):
        B,T,_ = rgb_seq.shape

        rgb = self.rgb_proj(rgb_seq)
        hf  = self.hf_proj(hf_seq)

        # add temporal pos encoding
        pos = self.pos[:,:T]  # (1,T,D)
        rgb, hf = rgb+pos, hf+pos

        # Co-Attention Layers
        for blk in self.co_blocks:
            rgb, hf = blk(rgb,hf)

        x = rgb + hf

        # Transformer fusion
        x = self.transformer(x)  # (B,T,D)
        
        score = self.temp_att(x)         # (B,T,1)
        att = torch.softmax(score, dim=1) # (B,T,1)
        x = torch.sum(att * x, dim=1)    # (B,D)

        x = self.norm(x)

        return self.cls(x)
    
class DeepFake_Final(nn.Module):
    def __init__(self, img_size=224, T=16, num_classes=2):
        super().__init__()
        self.rgb = S3DCNN(in_channels=3, base=32)      # → (B,T,512)
        self.hf  = HF_Seq(img_size=img_size)           # → (B,T,1280)

        self.head = FusionHead(
            rgb_dim=512, hf_dim=1280, d_model=512,
            num_classes=num_classes,
            co_layers=2, trans_layers=1,
            nhead=4, drop=0.1,
            max_len=T
        )

    def forward(self, video):     # video: (B, C, T, H, W)
        rgb_seq = self.rgb(video)                          # (B,T,512)
        hf_seq  = self.hf(video.permute(0,2,1,3,4))         # (B,T,1280)

        return self.head(rgb_seq, hf_seq)