import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)  
        x = self.act(x)  
        x = self.drop(x)  
        x = self.fc2(x)  
        x = self.drop(x)  

        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads=16, qkv_bias=True, attn_p=0., proj_p=0., cross=False):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.cross = cross
        if cross:
            self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):

        if self.cross:
            n_samples, n_tokens, dim = x[0].shape
            if dim != self.dim:
                raise ValueError

            n_tokens_en = n_tokens
            q = self.q_linear(x[0]).reshape(n_samples, n_tokens, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            k = self.k_linear(x[1]).reshape(n_samples, n_tokens_en, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v_linear(x[2]).reshape(n_samples, n_tokens_en, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            n_samples, n_tokens, dim = x.shape
            if dim != self.dim:
                raise ValueError

            qkv = self.qkv(x)  
            qkv = qkv.reshape(
                n_samples, n_tokens, 3, self.n_heads, self.head_dim
            )  
            qkv = qkv.permute(
                2, 0, 3, 1, 4
            )  
            q, k, v = qkv[0], qkv[1], qkv[2]

        k_t = k.transpose(-2, -1)  
        dp = (q @ k_t) * self.scale  

        if self.cross:
            # t_str = time.time()
            # dp_s = dp.softmax(dim=-1)
            # vision_features(dp_s, 'atten', 'dp_'+str(t_str))
            dp = -1 * dp
            # attn = dp.softmax(dim=-1)
            # vision_features(attn, 'atten', 'dp_v_'+str(t_str))
        attn = dp.softmax(dim=-1)  
        attn = self.attn_drop(attn)
        self.last_attn = attn
        weighted_avg = attn @ v  
        weighted_avg = weighted_avg.transpose(1, 2)  
        weighted_avg = weighted_avg.flatten(2)  

        x = self.proj(weighted_avg)  
        x = self.proj_drop(x)  

        
        return x

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0., cross=False):
        super().__init__()
        self.cross = cross
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.dim,self.n_heads, self.qkv_bias, self.p, self.attn_p =dim,n_heads, qkv_bias, p, attn_p
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p,
            cross=cross
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
        )

    def forward(self, x):
        if self.cross:
            x_ = [self.norm1(_x) for _x in x]
            out = x[2] + self.attn(x_)
            out = out + self.mlp(self.norm2(out))
            out = [x_[0], out, out]
        else:
            out = x + self.attn(self.norm1(x))
            out = out + self.mlp(self.norm2(out))
        
        return out