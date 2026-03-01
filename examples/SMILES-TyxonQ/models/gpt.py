import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_block import TransformerBlock

class GPTConfig:
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout=0.1, bias=False):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None and module.bias.requires_grad:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        b, t = idx.size()
        device = idx.device
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0).expand(b, t)
        tok_emb = self.transformer['wte'](idx)
        pos_emb = self.transformer['wpe'](pos)
        x = self.transformer['drop'](tok_emb + pos_emb)
        for block in self.transformer['h']:
            x = block(x)
        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x)
        return logits
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay = set()
        no_decay = set()
        for name, param in self.named_parameters():
            if 'bias' in name or 'ln' in name:
                no_decay.add(name)
            else:
                decay.add(name)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_deday": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

class GPTQE(GPT):
    def calculate_loss(self, tokens, energies):
        current_tokens, next_tokens = tokens[:, :-1], tokens[:, 1:]
        logits = self(current_tokens)
        next_token_mask = F.one_hot(next_tokens.long(), num_classes=self.lm_head.out_features)
        next_token_logits = (logits * next_token_mask).sum(dim=2)
        cumsum_logits = torch.cumsum(next_token_logits, dim=1)
        energies = energies.view(-1, 1).expand_as(cumsum_logits)
        loss = torch.mean(torch.square(cumsum_logits - energies))
        return loss
    
    @torch.no_grad()
    def generate(self, n_sequences, max_new_tokens, temperature=1.0, device="cpu"):
        idx = torch.zeros(size=(n_sequences, 1), dtype=torch.long, device=device)
        total_logits = torch.zeros(size=(n_sequences, 1), device=device)
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.transformer['wpe'].num_embeddings else idx[:, -self.transformer['wpe'].num_embeddings:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            logits[:, 0] = -float("inf")
            probs = F.softmax(logits / temperature, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            selected_logits = torch.gather(logits, index=idx_next, dim=1)
            total_logits += selected_logits
            idx = torch.cat((idx, idx_next), dim=1)
        return idx, total_logits
    
