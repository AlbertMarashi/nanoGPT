import math
import inspect
import json
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# torch.autograd.set_detect_anomaly(True)

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        ))

        self.thinking_block = ThinkingBlock(config)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer + self.thinking_block.AVG_THINKING_STEPS_TARGET))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb
        # x = self.transformer.drop(x)
        # for block in self.transformer.h:
        #     x = block(x)

        if targets is not None:
            return self.thinking_block(x, targets)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            return logits, None

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

class ThinkingBlock(nn.Module):
    MAX_ITER = 1
    AVG_THINKING_STEPS_TARGET = 1
    THRESHOLD_SENSITIVITY = 0.0005
    # LOSS_DIFF_SCALE = THRESHOLD_SENSITIVITY
    EMBED = 128
    MIN_THINKING_STEPS = 1

    def __init__(self, config):
        super().__init__()
        # self.EMBED = config.n_embd
        new_config = GPTConfig(
            n_embd=self.EMBED,
            block_size=config.block_size,
            vocab_size=config.vocab_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            dropout=config.dropout,
            bias=config.bias,
        )
        self.block = Block(new_config)
        self.step_count = 0
        self.threshold = nn.Parameter(torch.tensor(0.0))
        self.difficulty_head = nn.Sequential(
            # nn.LayerNorm(self.EMBED, bias=config.bias),
            nn.Linear(self.EMBED, self.EMBED),  # Hidden layer 1
            nn.Linear(self.EMBED, 1), # Output layer
        )
        self.to_logits = nn.Sequential(
            # nn.LayerNorm(self.EMBED, bias=config.bias),
            nn.Linear(self.EMBED, config.vocab_size, bias=False),
            nn.LayerNorm(config.vocab_size, bias=config.bias),
        )

    def forward(self, x, targets):
        self.step_count += 1
        B, T, C = x.shape
        update_mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
        thinking_steps = torch.zeros(B, T, dtype=torch.long, device=x.device)
        predicted_difficulties = torch.zeros(B, T, dtype=torch.float, device=x.device)
        real_losses = torch.zeros(B, T, dtype=torch.float, device=x.device)

        all_predicted_difficulties = []
        all_real_losses = []

        for i in range(self.MAX_ITER):
            thinking_steps[update_mask] += 1
            # update the x tensor with the updated tokens
            x = torch.where(update_mask.unsqueeze(-1), self.block(x), x)
            # Calculate the predicted losses, only for the tokens that are being updated
            predicted_difficulties[update_mask] = self.difficulty_head(x).squeeze(-1)[update_mask]
            # Compute the logits
            logits = self.to_logits(x)

            if targets is not None:
                # Compute the real loss
                new_losses = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction='none').view(B, T)
                # real_losses[update_mask] = new_losses[update_mask]
                real_losses = torch.where(update_mask, new_losses, real_losses)
                if i == 0:
                    # initial_losses = real_losses.clone()
                    initial_losses = new_losses

                all_predicted_difficulties.append(predicted_difficulties[update_mask].view(-1))
                all_real_losses.append(real_losses[update_mask].view(-1))

            if i >= self.MIN_THINKING_STEPS - 1:
                # Determine which tokens should be updated next based on the threshold
                update_mask = update_mask & (predicted_difficulties > self.threshold)

        if targets is not None:
            thinking_steps = thinking_steps.sum()
            avg_thinking_steps = thinking_steps / (B * T)
            initial_loss = initial_losses.mean()
            real_loss = real_losses.mean()
            # loss_improvement = real_loss_total - initial_loss_total
            # real_loss_total = real_losses.sum()


            # if self.training and self.step_count % 50 == 0:
            #     initial_loss = initial_loss_total / (B * T)
            #     print(f"initial_loss_total: {initial_loss_total.item():.0f}, real_loss_total: {real_loss_total.item():.0f}")

            if self.training:
                usage_diff = avg_thinking_steps - self.AVG_THINKING_STEPS_TARGET
                self.threshold.data += usage_diff * self.THRESHOLD_SENSITIVITY

            stacked_losses = torch.stack((torch.cat(all_predicted_difficulties, dim=0), torch.cat(all_real_losses, dim=0)), dim=0)
            corr_coef = torch.corrcoef(stacked_losses)[0, 1]
            corr_loss = -0.1 * corr_coef

            # loss = (torch.cat(all_real_losses).sum() / (B * T) / thinking_steps) + corr_loss


            loss = real_loss + corr_loss + initial_loss
            # loss = real_loss + corr_loss + initial_loss_total
            if self.training and self.step_count % 10 == 0:
                print(json.dumps({
                    "step": self.step_count,
                    "loss": f"{loss.item():.4f}",
                    "real_loss": f"{real_loss.item():.4f}",
                    "corr_coef": f"{corr_coef.item():.4f}",
                    "avg_thinking_steps": f"{avg_thinking_steps.item():.2f}",
                    "predicted_difficulty": f"{torch.cat(all_predicted_difficulties).sum().item() / thinking_steps:.3f}",
                    "threshold": f"{self.threshold.item():.4f}",
                }))
            return logits, loss
        else:
            return logits, None




            # if i > 0 and self.training and self.step_count % 50 == 0:
            #     b = 0
            #     total_actual_loss = 0.0
            #     total_prev_loss = 0.0
            #     num_tokens_updated = 0
            #     for t in range(min(10, T)):
            #         prev_difficulty = predicted_difficulties[b, t].item()
            #         difficulty = new_difficulties[b, t].item()
            #         actual_token_loss = real_losses_new[b, t].item()
            #         prev_loss = prev_losses[b, t].item()
            #         token_updated = update_mask[b, t].item()
            #         if token_updated:
            #             num_tokens_updated += 1
            #             total_actual_loss += actual_token_loss
            #             total_prev_loss += prev_loss
            #             print(f"i: {i}, b: {b}, t: {t}, difficulty: {difficulty:.4f}, prev_loss: {prev_loss:.2f}, loss: {actual_token_loss:.2f}, loss_change: {prev_loss - actual_token_loss:.4f}")
            #     total_improvement = total_prev_loss - total_actual_loss
            #     avg_total_improvement = total_improvement / num_tokens_updated if num_tokens_updated > 0 else 0.0
            #     print(f"i: {i}, avg_total_improvement: {avg_total_improvement:.4f}")
            #     # print("==========================")