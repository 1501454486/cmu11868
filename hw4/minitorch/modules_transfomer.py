import numpy as np
from .tensor import tensor, tensor_from_numpy
from .module import Module, Parameter
from .modules_basic import (
    Embedding,
    Dropout,
    LayerNorm1d,
    Linear
)
from .tensor_ops import TensorBackend
from .nn import (
    max,
    dropout,
    GELU,
    logsumexp,
)
from typing import Any, Dict, Optional, Sequence, Tuple

datatype = np.float32


class MultiHeadAttention(Module):
    def __init__(self, n_embd: int, n_head: int, causal: bool=True, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """Implements Multi-Head Attention as described in "Attention Is All You Need"

        Args:
            n_embd: Dimensionality of embeddings and hidden states
            n_head: Number of heads
            p_dropout: Dropout ratio for dropout layer
            causal: If True, then apply a causal mask during self-attention
            bias: If True, then apply a bias in Linear layers
        
        Attributes:
            q_projection: Linear layer projecting input to Q matrix
            k_projection: Linear layer projecting input to K matrix
            v_projection: Linear layer projecting input to V matrix
            out_projection: Linear output projection layer
            dropout: Dropout layer
        """
        self.backend = backend
        self.n_embd = n_embd 
        self.n_head = n_head
        self.causal = causal
        self.attn_hidden_dim = n_embd // n_head

        ### BEGIN ASSIGN3_3
        self.q_projection = Linear(n_embd, n_embd, bias = bias, backend = backend)
        self.k_projection = Linear(n_embd, n_embd, bias = bias, backend = backend)
        self.v_projection = Linear(n_embd, n_embd, bias = bias, backend = backend)
        self.out_projection = Linear(n_embd, n_embd, bias = bias, backend = backend)
        self.dropout = Dropout(p_dropout = p_dropout)
        ### END ASSIGN3_3

    def create_causal_mask(self, seq_len):
        """
        Create a causal mask for self-attention to prevent information leakage.
        
        Generates a triangular mask where each position can only attend to previous
        positions and itself. Upper triangle contains -inf, lower triangle contains 0.

        Args:
            seq_len (int): Length of the sequence

        Returns:
            Tensor: Causal mask of shape (1, 1, seq_len, seq_len) with -inf above
                    diagonal and 0 on/below diagonal. Will be broadcasted to full
                    attention tensor shape during computation.
        """
        # Returns a 1x1xTxt triangular causal mask for Q @ K^T (You will implicitly broadcast it to BxHxTxT)
        mask = -np.finfo(datatype).max * np.triu(np.ones((1, 1, seq_len, seq_len), dtype=datatype), 1)
        return tensor_from_numpy(mask, backend=self.backend)

    def project_to_query_key_value(self, x):
        """
        Project input embeddings to Query, Key, and Value matrices for self-attention.
        
        Args:
            x (Tensor): Input embeddings of shape (batch_size, seq_len, n_embd)

        Returns:
            tuple: (q, kT, v) where:
                - q: Query matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
                - kT: Transposed key matrix of shape (batch_size, num_heads, attn_hidden_dim, seq_len)
                - v: Value matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN ASSIGN3_3
        def proj(layer):
            # (bs, seq_len, n_embd) -> (bs, n_head, seq_len, d)
            y = layer(x).view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)
            return y.permute(0, 2, 1, 3).contiguous()
        
        q = proj(self.q_projection)
        k = proj(self.k_projection)
        v = proj(self.v_projection)
        
        # Don't transpose K here - return as kT for compatibility
        kT = k.permute(0, 1, 3, 2).contiguous()
        
        ### END ASSIGN3_3
        return q, kT, v
    
    def self_attention(self, q, kT, v):
        """
        Compute self-attention: softmax((q @ kT) / sqrt(attn_hidden_dim)) @ v.
        
        Args:
            q (Tensor): Query matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
            kT (Tensor): Transposed key matrix of shape (batch_size, num_heads, attn_hidden_dim, seq_len)
            v (Tensor): Value matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)

        Returns:
            Tensor: Attention output of shape (batch_size, seq_len, n_embd)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, k_dim, _ = kT.shape  # kT is transposed
        _, _, _, v_dim = v.shape
        assert q_dim == k_dim == v_dim
        result = None
        
        ### BEGIN ASSIGN3_3
        attn_scores = q @ kT / (self.attn_hidden_dim ** 0.5)
        if self.causal:
            # if need mask
            mask = self.create_causal_mask(queries_len)
            attn_scores = attn_scores + mask
            
        # Manual softmax using logsumexp (to avoid broken softmax gradients)
        # softmax(x, dim) = exp(x - logsumexp(x, dim))
        log_sum_exp = logsumexp(attn_scores, dim=3)
        attn_weights = (attn_scores - log_sum_exp).exp()
        
        attn_weights = self.dropout(attn_weights)
        
        # (bs, h, T, T) @ (bs, h, T, d_v) -> (bs, h, T, d_v)
        attention_output = attn_weights @ v
        
        # concatenate: permute and concatenate
        permuted = attention_output.permute(0, 2, 1, 3).contiguous()
        result = permuted.view(batch_size, queries_len, self.n_embd)
            
        ### END ASSIGN3_3

        return result

    def forward(self, x):
        """
        Compute multi-head attention with optional causal masking.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN ASSIGN3_3
        q, kT, v = self.project_to_query_key_value(x)
        output = self.self_attention(q, kT, v)
        output = self.out_projection(output)
        output = self.dropout(output)
        return output
        ### END ASSIGN3_3


class FeedForward(Module):
    def __init__(self, n_embd: int, middle_dim: int=256, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """
        Initialize a feed-forward network module.
        
        Args:
            n_embd (int): Input and output dimension
            middle_dim (int): Hidden layer dimension, default 256
            p_dropout (float): Dropout probability, default 0.1
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            linear_in (Linear): First linear layer
            linear_out (Linear): Second linear layer
            dropout (Dropout): Dropout layer
        """
        ### BEGIN ASSIGN3_3
        self.linear_in = Linear(n_embd, middle_dim, bias = bias, backend = backend)
        self.linear_out = Linear(middle_dim, n_embd, bias = bias, backend = backend)
        self.dropout = Dropout(p_dropout = p_dropout)
        ### END ASSIGN3_3

    def forward(self, x):
        """
        Forward pass through feed-forward network with GELU activation and dropout.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape

        ### BEGIN ASSIGN3_3
        x = self.linear_in.forward(x)
        x = GELU(x)
        x = self.linear_out.forward(x)
        x = self.dropout(x)
        ### END ASSIGN3_3

        return x
    

class TransformerLayer(Module):
    def __init__(self, n_embd: int, n_head: int, p_dropout: float=0.1, ln_eps: float=1e-5, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """
        Initialize a transformer layer with pre-layer normalization.
        
        Args:
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
            p_dropout (float): Dropout probability, default 0.1
            ln_eps (float): Layer normalization epsilon, default 1e-5
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            ln_1 (LayerNorm1d): First layer normalization before attention
            ln_2 (LayerNorm1d): Second layer normalization after attention
            attention (MultiHeadAttention): Multi-head attention layer
            ff (FeedForward): Feed-forward network layer
        """
        ### BEGIN ASSIGN3_3
        self.ln_1 = LayerNorm1d(n_embd, eps = ln_eps, backend = backend)
        self.ln_2 = LayerNorm1d(n_embd, eps = ln_eps, backend = backend)
        self.attention = MultiHeadAttention(n_embd, n_head, p_dropout = p_dropout, bias = bias, backend = backend)
        self.ff = FeedForward(n_embd, p_dropout = p_dropout, bias = bias, backend = backend)
        ### END ASSIGN3_3

    def forward(self, x):
        """
        Forward pass through transformer layer with pre-layer normalization.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)
        
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN YOUR SOLUTION
        output = self.ln_1.forward(x.view(batch_size * seq_len, n_embd)).view(batch_size, seq_len, n_embd)
        output = self.attention.forward(output)
        x = x + output
        output = self.ln_2.forward(x.view(batch_size * seq_len, n_embd)).view(batch_size, seq_len, n_embd)
        output = self.ff.forward(output)
        x = x + output
        return x
        
        ### END YOUR SOLUTION


class DecoderLM(Module):
    def __init__(
        self, 
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        p_dropout: float=0.1,
        ln_eps: float=1e-5, 
        bias: bool=True,
        backend: TensorBackend=None
    ):
        super().__init__()
        """
        Initialize a decoder-only transformer language model.
        
        Args:
            n_vocab (int): Vocabulary size
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
            n_positions (int): Maximum sequence length
            p_dropout (float): Dropout probability, default 0.1
            ln_eps (float): Layer normalization epsilon, default 1e-5
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            token_embeddings (Embedding): Token embedding layer
            position_embeddings (Embedding): Position embedding layer
            t_layer_1 (TransformerLayer): First transformer layer
            t_layer_2 (TransformerLayer): Second transformer layer
            t_layer_3 (TransformerLayer): Third transformer layer
            t_layer_4 (TransformerLayer): Fourth transformer layer
            dropout (Dropout): Dropout layer before transformer layers
            ln (LayerNorm1d): Final layer normalization
            lm_head (Linear): Language model head for vocabulary projection
        """
        self.backend = backend
        self.n_embd = n_embd
        self.n_vocab = n_vocab
        ### BEGIN ASSIGN3_3
        self.token_embeddings = Embedding(
            num_embeddings = n_vocab,
            embedding_dim = n_embd,
            backend = backend
        )
        self.position_embeddings = Embedding(
            num_embeddings = n_positions,
            embedding_dim = n_embd,
            backend = backend
        )
        self.t_layer_1 = TransformerLayer(
            n_embd = n_embd,
            n_head = n_head,
            p_dropout = p_dropout,
            ln_eps = ln_eps,
            bias = bias,
            backend = backend
        )
        self.t_layer_2 = TransformerLayer(
            n_embd = n_embd,
            n_head = n_head,
            p_dropout = p_dropout,
            ln_eps = ln_eps,
            bias = bias,
            backend = backend
        )
        self.t_layer_3 = TransformerLayer(
            n_embd = n_embd,
            n_head = n_head,
            p_dropout = p_dropout,
            ln_eps = ln_eps,
            bias = bias,
            backend = backend
        )
        self.t_layer_4 = TransformerLayer(
            n_embd = n_embd,
            n_head = n_head,
            p_dropout = p_dropout,
            ln_eps = ln_eps,
            bias = bias,
            backend = backend
        )
        self.dropout = Dropout(p_dropout = p_dropout)
        self.ln = LayerNorm1d(
            dim = n_embd,
            eps = ln_eps,
            backend = backend
        )
        self.lm_head = Linear(
            in_size = n_embd,
            out_size = n_vocab,
            bias = bias,
            backend = backend
        )
        ### END ASSIGN3_3
    
    def forward(self, idx):
        """
        Forward pass through decoder-only transformer language model.
        
        Args:
            idx (Tensor): Input token indices of shape (batch_size, seq_len)
        
        Returns:
            Tensor: Logits of shape (batch_size, seq_len, n_vocab)
        """
        
        batch_size, seq_len = idx.shape

        ### BEGIN ASSIGN3_3
        # 1. Get token embeddings of shape (batch_size, seq_len, n_embd)
        # 2. Create positional embeddings of shape (1, seq_len, n_embd):
        #    - Create position ids tensor [0, 1, 2, ..., seq_len-1] of shape (1, seq_len)
        #    - Pass through positional embedding layer
        #    - Ensure output shape is (1, seq_len, n_embd)
        # 3. Add token and positional embeddings
        # 4. Apply dropout
        # 5. Pass through transformer layers (t_layer_1 to t_layer_4)
        # 6. Apply final layer normalization
        # 7. Project to vocabulary size using lm_head

        # Shape: (batch_size, seq_len, n_embd)
        token_embeddings = self.token_embeddings.forward(idx)

        # shape: (1, seq_len)
        pos = tensor(list(range(seq_len)), backend = self.backend, requires_grad = True).view(1, seq_len)
        pos_embeddings = self.position_embeddings.forward(pos)      # shape: (1, seq_len, n_embd)
        
        embeddings = token_embeddings + pos_embeddings
        
        output = self.dropout.forward(embeddings)
        
        output = self.t_layer_1.forward(output)
        output = self.t_layer_2.forward(output)
        output = self.t_layer_3.forward(output)
        output = self.t_layer_4.forward(output)
        
        output = self.ln.forward(output.view(batch_size * seq_len, self.n_embd)).view(batch_size, seq_len, self.n_embd)
        output = self.lm_head.forward(output)
        
        return output
        ### END ASSIGN3_3
