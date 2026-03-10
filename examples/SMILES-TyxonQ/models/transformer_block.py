import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(embed_dim=config.n_embd, 
                                          num_heads=config.n_head,
                                          dropout=config.dropout,
                                          bias=config.bias)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(nn.Linear(config.n_embd, 4*config.n_embd),
                                 nn.GELU(),
                                 nn.Linear(4*config.n_embd, config.n_embd),
                                 nn.Dropout(config.dropout),
                                 )
        
    def forward(self, x):
        x_ln1 = self.ln1(x)
        attn_output, _ = self.attn(x_ln1, x_ln1, x_ln1)
        x = x + attn_output
        x_ln2 = self.ln2(x)
        mlp_output = self.mlp(x_ln2)
        x = x + mlp_output
        return x
    
