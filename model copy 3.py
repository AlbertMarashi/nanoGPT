import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.autograd.set_detect_anomaly(True)

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
        self.config = config
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'thinking': ThinkingBlock(config),  # The thinking block at the end
            'ln_f': LayerNorm(config.n_embd, bias=config.bias)
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # Weight tying
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

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
            {'params': nodecay_params, 'weight_decay': 0.0},
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
        pos = torch.arange(t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)  # Token embeddings (batch, token, n_embd)
        pos_emb = self.transformer.wpe(pos)  # Position embeddings (token, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x, reg_loss = self.transformer.thinking(x)

        # Get logits from the final layer normalization and linear transformation
        logits = self.lm_head(self.transformer.ln_f(x))

        # If targets are provided, compute loss
        if targets is not None:
            total_loss = main_loss(logits, targets, 0)
            # print(f"total loss: {total_loss}")
            return logits, total_loss
        else:
            return logits  # For inference, only return logits

class ThinkingBlock(nn.Module):
    MAX_ITER = 5
    INTIAL_ENERGY_BUDGET = 10.0
    ENERGY_SCALE_FACTOR = 1.0
    def __init__(self, config):
        super().__init__()
        self.block = Block(config)
        self.energy_budget = nn.Parameter(torch.tensor(self.INTIAL_ENERGY_BUDGET))

    def forward(self, x):
        B, T, C = x.shape
        update_mask = torch.ones(B, T, dtype=torch.bool, device=x.device)

        for i in range(self.MAX_ITER):
            # Compute the entropy of the current predictions
            probs = F.softmax(x, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs), dim=-1)

            if i > 0:
                # Compute the energy based on entropy improvement
                entropy_improvement = prev_entropy - entropy
                energy = entropy_improvement * self.ENERGY_SCALE_FACTOR

                # Select the top tokens within the energy budget
                energy_budget = F.relu(self.energy_budget)  # Ensure non-negative budget
                sorted_indices = torch.argsort(energy, descending=True)
                cumulative_energy = torch.cumsum(energy[sorted_indices], dim=-1)
                num_tokens_to_update = torch.sum(cumulative_energy <= energy_budget)
                update_mask = torch.zeros_like(update_mask)
                update_mask[sorted_indices[:num_tokens_to_update]] = True

            # Apply the block(s) to the tokens that require further processing
            x_to_update = x[update_mask]
            x_updated = self.block(x_to_update)
            x[update_mask] = x_updated

            prev_entropy = entropy

        return x

# class ThinkingBlock(nn.Module):
#     MAX_ITER = 5
#     INITIAL_THRESHOLD = 0.01
#     LAMBDA_REG = 0.01
#     TARGET_AVG_ITERATIONS = MAX_ITER / 2

#     def __init__(self, config):
#         super().__init__()
#         self.block = Block(config)
#         self.threshold = nn.Parameter(torch.tensor([self.INITIAL_THRESHOLD]))

#     def forward(self, x):
#         B, T, C = x.size()

#         # Initialize variables
#         iterations = torch.ones(B, T, dtype=torch.int32, device=x.device)
#         total_improvement = 0.0
#         update_mask = torch.ones(B, T, dtype=torch.bool, device=x.device)

#         # Iterate for a maximum number of iterations
#         for i in range(self.MAX_ITER - 1):
#             # Pass the input through the block
#             x_new = self.block(x)

#             # Calculate the improvement in token probabilities
#             new_x_delta = x_new.softmax(dim=-1).max(dim=-1)[0]
#             old_x_delta = x.softmax(dim=-1).max(dim=-1)[0]
#             improvement = new_x_delta - old_x_delta
#             total_improvement += improvement.mean().item()
#             # print(f"improvement: {improvement.mean():.3f}")

#             # If token predictions have increased in certainty
#             update_mask = update_mask & (new_x_delta > old_x_delta)

#             # Update the tokens that exceed the threshold
#             x = torch.where(update_mask.unsqueeze(-1), x_new, x)

#             # Increment the iteration count for updated tokens
#             iterations += update_mask.int()

#             # Check if all tokens have converged or reached the maximum iterations
#             if update_mask.sum() == 0 or i == self.MAX_ITER - 1:
#                 break

#         avg_improvement = total_improvement / (i + 1)

#         # print(f"avg improvement: {avg_improvement:.3f}")

#         # Compute the average number of iterations
#         avg_iterations = iterations.float().mean()

#         # Calculate the deviation from the target average iterations
#         iteration_deviation = self.TARGET_AVG_ITERATIONS - avg_iterations

#         # Calculate the regularization term based on the iteration deviation
#         regularization = self.LAMBDA_REG * iteration_deviation

#         # print(f"avg iterations: {avg_iterations:.2f}, threshold: {self.threshold.item():.4f}, avgregularization: {regularization.item():.3f}")

#         return x, regularization.mean()


# class AdaptiveThinkingBudget:
#     # Constants for configuration
#     TARGET_USAGE = 0.5
#     INCREMENTAL_EFFECT = 0.0005  # Controls the incremental effect of new observations.
#     INFLUENCE = 0.01

#     def __init__(self, max_thinking_steps):
#         self.max_thinking_steps = max_thinking_steps
#         self.threshold = 0.0
#         self.avg_thinking_steps = 0.0

#     def update(self, num_thinking_steps):
#         self.avg_thinking_steps = self.avg_thinking_steps * (1 - self.INFLUENCE) + num_thinking_steps * self.INFLUENCE
#         proportion_used = num_thinking_steps / self.max_thinking_steps
#         usage_diff = self.TARGET_USAGE - proportion_used

#         self.threshold += self.INCREMENTAL_EFFECT * usage_diff
#         # Clamp the threshold to ensure it remains within reasonable bounds.
#         self.threshold = max(0.0, min(1.0, self.threshold))

def main_loss(logits, targets, reg_loss):
    """
    Compute the primary loss function (cross-entropy) for the predictions
    and add regularization loss to enforce computational efficiency.

    Args:
        logits (torch.Tensor): The logits output by the model (batch size x sequence length x vocab size).
        targets (torch.Tensor): The ground truth target indices (batch size x sequence length).
        reg_loss (torch.Tensor): The regularization loss computed from the ThinkingBlock.

    Returns:
        torch.Tensor: The total loss combining both cross-entropy and regularization.
    """
    # Cross-entropy loss, assuming targets are class indices
    predictive_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

    # Combine the predictive loss with the regularization loss
    total_loss = predictive_loss + reg_loss
    return total_loss
