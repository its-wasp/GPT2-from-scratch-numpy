"""DeepCrossAttention: GRN-v3 residual mixing + cross-depth attention.

"""

from rmk.backend import xp
from rmk.tensor import Tensor
from rmk.losses import cross_entropy
from rmk.functional import relu, softmax
from rmk.nn import Module, Linear, LayerNorm, Dropout, Embedding, ModuleList
from rmk.transformer import MLP


class GRNv3(Module):
    """Input-dependent weighted combination of a stack of layer outputs.

    g = sum_i  h_i * (b_i + relu(w . h_i)),  over the stack [h_0..h_{n_stack-1}].
    """

    def __init__(self, n_stack, n_embd):
        super().__init__()
        self.w = Tensor(xp.zeros(n_embd))
        # one (C,) bias per stack entry, as separate registered params (Tensor has no indexing op)
        self.bs = []
        for i in range(n_stack):
            bi = Tensor(xp.ones(n_embd))
            setattr(self, f"b{i}", bi)
            self.bs.append(bi)

    def forward(self, stack):
        out = None
        for h, b in zip(stack, self.bs):
            gate = relu((h * self.w).sum(axis=-1, keepdims=True))  # (B,T,1) input-dependent
            weight = b + gate                                      # (B,T,C) via broadcast
            term = h * weight
            out = term if out is None else out + term
        return out


class CrossMHA(Module):
    """Causal multi-head attention whose Q/K/V come from three separate inputs."""

    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.q_proj = Linear(n_embd, n_embd)
        self.k_proj = Linear(n_embd, n_embd)
        self.v_proj = Linear(n_embd, n_embd)
        self.out_proj = Linear(n_embd, n_embd)
        self.attn_drop = Dropout(dropout)
        self.resid_drop = Dropout(dropout)

    def forward(self, q_in, k_in, v_in):
        B, T, C = q_in.data.shape
        nh, hd = self.n_head, self.head_dim

        def split(t):
            return t.reshape(B, T, nh, hd).transpose((0, 2, 1, 3))

        q = split(self.q_proj(q_in))
        k = split(self.k_proj(k_in))
        v = split(self.v_proj(v_in))

        scale = 1.0 / xp.sqrt(hd)
        scores = (q @ k.transpose((0, 1, 3, 2))) * scale
        mask = xp.triu(xp.ones((T, T)) * -1e9, k=1)
        scores = scores + mask

        weights = self.attn_drop(softmax(scores, axis=-1))
        out = weights @ v
        out = out.transpose((0, 2, 1, 3)).reshape(B, T, C)
        return self.resid_drop(self.out_proj(out))


class DCABlock(Module):
    """Three GRN-v3 compose the Q/K/V inputs from the layer stack; then attention + MLP."""

    def __init__(self, n_embd, n_head, n_stack, dropout=0.0):
        super().__init__()
        self.grn_q = GRNv3(n_stack, n_embd)
        self.grn_k = GRNv3(n_stack, n_embd)
        self.grn_v = GRNv3(n_stack, n_embd)
        self.ln_q = LayerNorm(n_embd)
        self.ln_k = LayerNorm(n_embd)
        self.ln_v = LayerNorm(n_embd)
        self.attn = CrossMHA(n_embd, n_head, dropout)
        self.ln_add = LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, stack):
        q = self.grn_q(stack)
        k = self.grn_k(stack)
        v = self.grn_v(stack)

        a = self.attn(self.ln_q(q), self.ln_k(k), self.ln_v(v))  
        h = self.ln_add(a + q)                                   
        return a + self.mlp(h)                                   


class DCAGPT(Module):
    """GPT with DeepCrossAttention blocks. Same GPTConfig + forward signature as GPT."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = Embedding(config.vocab_size, config.n_embd)
        self.wpe = Embedding(config.block_size, config.n_embd)
        self.drop = Dropout(config.dropout)
        # block i sees the embedding output + i previous block outputs -> i+1 stack entries
        self.blocks = ModuleList([
            DCABlock(config.n_embd, config.n_head, n_stack=i + 1, dropout=config.dropout)
            for i in range(config.n_layer)
        ])
        self.grn_f = GRNv3(config.n_layer + 1, config.n_embd)
        self.ln_f = LayerNorm(config.n_embd)
        self._apply_scaled_init()

    def _apply_scaled_init(self):
        std = 0.02 / xp.sqrt(2 * self.config.n_layer)
        for block in self.blocks:
            for w in (block.attn.out_proj.weight, block.mlp.fc2.weight):
                w.data[...] = xp.random.standard_normal(w.data.shape) * std

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, "sequence longer than block_size"

        pos = xp.arange(T)
        x = self.drop(self.wte(idx) + self.wpe(pos))

        stack = [x]
        for block in self.blocks:
            stack.append(block(stack))

        final = self.ln_f(self.grn_f(stack))
        logits = final @ self.wte.weight.transpose()
        loss = cross_entropy(logits, targets) if targets is not None else None
        return logits, loss