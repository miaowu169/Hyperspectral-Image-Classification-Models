import torch
from torch import nn

from einops import rearrange, repeat
from typing import Optional
# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):  # 中间可预设
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):  # dim, 1, 2,64,1024,0.1
        super().__init__()
        inner_dim = dim_head * heads  # 512
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 做一个切分
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1):  # dim, 1, 2,64,1024,0.1
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class PatchEmbed(nn.Module):
    """
    3D  HYPERSPECTRAL Image to spectral Embedding
    """
    def __init__(self, in_channels: int,
                 out_channels: int,
                 patch_size: int,
                 stride: int = 1,
                 reduce_ratio: int = 4,
                 sigma_mapping: Optional[nn.Module] = None,
                 bias: bool = False,
                 norm_layer=False,
                 ):
        super(PatchEmbed, self).__init__()
        assert isinstance(in_channels, int) and in_channels > 0, "in channels must be a positive integer."
        # assert in_channels % groups == 0, "out_channels must be divisible by groups"
        assert isinstance(out_channels, int) and out_channels > 0, "out channels must be a positive integer."
        assert isinstance(patch_size, int), "patch_size must be an integer"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.stride = stride
        self.reduce_ratio = reduce_ratio
        # 输入通道，输出通道，卷积核大小，步长
        # C*H*W->embed_dim*grid_size*grid_size
        self.norm = nn.LayerNorm(out_channels) if norm_layer else nn.Identity()  # batch_norm or layer_norm???

        self.o_mapping = nn.AvgPool2d(kernel_size=self.stride, stride=self.stride)  # 1,1

        self.sigma_mapping = sigma_mapping if sigma_mapping is not None else nn.Sequential(
            nn.BatchNorm2d(num_features=self.out_channels // self.reduce_ratio, momentum=0.3), nn.ReLU())

        self.reduce_mapping = nn.Conv2d(in_channels=self.in_channels,
                                        out_channels=self.out_channels // self.reduce_ratio, kernel_size=(1, 1),
                                        stride=(1, 1), padding=(0, 0), bias=bias)  # 通道数做一变换

        self.span_mapping = nn.Conv2d(in_channels=self.out_channels // self.reduce_ratio,
                                      out_channels=self.out_channels,
                                      kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=bias)

    def forward(self, x):
        # x = torch.squeeze(x)  # 降维
        x = self.span_mapping(self.sigma_mapping(self.reduce_mapping(self.o_mapping(x))))  # output:B*196*11*11
        x = rearrange(x, 'B C H W -> B (H W) C')  # B 121 196
        x = self.norm(x)
        return x


class Spect_Vit(nn.Module):
    """
    in_channels = 204
    out_channels = dim = 196
    patch_Size = 11
    depth: number of attention layer
    heads: Multi-Head
    MIP_dim: 2048//1024,隐藏层
    dropout: 0.1
    CLASSES: 若不直接输出，可直接写层输出的维度
    """
    def __init__(self, in_channels, out_channels, patch_size, num_classes, depth=2, heads=16, mlp_dim=1024, dropout=0.1,
                 pool='cls', emb_dropout=0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.patch_embeding = PatchEmbed(in_channels, out_channels, patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, patch_size**2 + 1, out_channels))
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channels))
        self.pool = pool
        self.dropout = nn.Dropout(emb_dropout)
        self.to_latent = nn.Identity()
        # self.liner = nn.Sequential(nn.Linear(channels, 100, bias=False),
        #                            nn.ReLU())
        self.transformer = Transformer(out_channels, depth, heads, dim_head=64, mlp_dim=mlp_dim, dropout=dropout)
        # transformer的参数：out_channels=dim =196, depth=1 ,heads = 2
        self.flatten = self.flattened()
        self.Linear = nn.Sequential(nn.Linear(
                in_features=self.flatten,
                out_features=num_classes,
                bias=False
            ), nn.BatchNorm1d(num_classes))

    def flattened(self):
        with torch.no_grad():
            x = torch.zeros((1, self.patch_size**2, self.out_channels))
            b, n, _ = x.shape
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
            x = self.dropout(x)  # 完成位置信息的嵌入
            # 接下来要进行self_attention的嵌入********，check dim
            x = self.transformer(x)  # transformer要体现出 q 和 k 的不同，刚开始进入的时候是q和k一样的。
            # print(x.shape)
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            x = self.to_latent(x)
            b, _ = x.shape
            return b*_  # b*input_dim

    def forward(self, x):
        # x = self.liner(img)
        x = self.patch_embeding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)  # 完成位置信息的嵌入
        # 接下来要进行self_attention的嵌入********，check dim
        x = self.transformer(x)  # transformer要体现出 q 和 k 的不同，刚开始进入的时候是q和k一样的。
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        x = self.Linear(x)

        return x


if __name__ == '__main__':
    x = torch.randn((1, 3, 16, 16))
    vit = Spect_Vit(3, 64, 16, 8)  # in_channels, out_channels, patch_size, num_classes
    # model = PatchEmbed(3, 64, 16)
    # ouput = model(x)
    output = vit(x)
    print(output.shape)