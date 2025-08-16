import pytest
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps

import numpy as np
import torch
import torch.nn as nn
import numba

np.random.seed(3)

datatype = np.float32

_BACKENDS = [pytest.param(
                 minitorch.TensorBackend(CudaKernelOps), 
                 marks=pytest.mark.skipif(not numba.cuda.is_available(), reason="No GPU")
             )]

@pytest.mark.a2_3
@pytest.mark.parametrize("batch_size",  [1, 64])
@pytest.mark.parametrize("queries_len", [2, 256])
@pytest.mark.parametrize("n_embd",      [64, 256])
@pytest.mark.parametrize("num_heads",   [1, 4])
@pytest.mark.parametrize("p_dropout",   [0.0])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_multihead_attention(batch_size, queries_len, n_embd, num_heads, p_dropout, backend):
    np.random.seed(10)
    torch.manual_seed(10)

    data = np.random.rand(batch_size, queries_len, n_embd)
    X    = minitorch.tensor_from_numpy(data, backend, True)
    X_   = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    layer_ = torch.nn.MultiheadAttention(n_embd, num_heads, p_dropout, bias=False, batch_first=True, dtype=torch.float32)
    layer = minitorch.MultiHeadAttention(n_embd, num_heads, True, p_dropout, bias=False, backend=backend)
    
    # Set weights of minitorch layer to torch weights
    w_qkv = layer_.in_proj_weight.detach().numpy().T.copy() # (n_embd, 3*n_embd)
    w_q_, w_k_, w_v_ = [w.copy() for w in np.split(w_qkv, 3, -1)] # 3 * (n_embd, n_embd)
    w_out_ = layer_.out_proj.weight.detach().numpy().T.copy()

    w_q = minitorch.tensor_from_numpy((w_q_), backend=backend, requires_grad=True)
    w_k = minitorch.tensor_from_numpy((w_k_), backend=backend, requires_grad=True)
    w_v = minitorch.tensor_from_numpy((w_v_), backend=backend, requires_grad=True)
    w_out = minitorch.tensor_from_numpy((w_out_), backend=backend, requires_grad=True)

    layer.q_projection.weights.value = w_q
    layer.k_projection.weights.value = w_k
    layer.v_projection.weights.value = w_v
    layer.out_projection.weights.value = w_out

    # The same mask is causal mask
    M = torch.triu(-float("inf")*torch.ones(queries_len, queries_len),1)

    result = layer(X)
    result_, _ = layer_(X_, X_, X_, attn_mask = M)
    
    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5,
        rtol=1e-5
    )

    # Check backward
    result.sum().backward()
    result_.sum().backward()
    
    np.testing.assert_allclose(
        X.grad.to_numpy(),
        X_.grad.detach().numpy(),
        atol=1e-5,
        rtol=1e-5
    )

    np.testing.assert_allclose(
        layer.out_projection.weights.value.grad.to_numpy(),
        layer_.out_proj.weight.grad.detach().numpy().T,
        atol=1e-5,
        rtol=1e-5
    )

    # Since the torch W_Q, W_K, W_V is all one matrix, we can't compare
    assert (
        (layer.q_projection.weights.value.grad is not None) and
        (layer.k_projection.weights.value.grad is not None) and
        (layer.v_projection.weights.value.grad is not None)
    )

@pytest.mark.a2_3
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [5])
@pytest.mark.parametrize("n_embd",  [9])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_feedforward_layer(batch_size, seq_len, n_embd, dropout, backend):

    np.random.seed(19943)

    x = np.random.randn(
        batch_size, seq_len, n_embd
    ).astype(datatype)

    layer = minitorch.FeedForward(
        n_embd=n_embd, p_dropout=dropout, bias=True, backend=backend)

    result = layer(
        minitorch.tensor(x.tolist(), backend=backend)
    )

    assert result is not None

@pytest.mark.a2_3
@pytest.mark.parametrize("batch_size", [2, 32])
@pytest.mark.parametrize("seq_len",   [128])
@pytest.mark.parametrize("n_embd",    [32, 64])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("causal",    [True])
@pytest.mark.parametrize("p_dropout", [0.0])
@pytest.mark.parametrize("ln_eps", [1e-5])
@pytest.mark.parametrize("bias", [False])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_transformer_layer(batch_size, seq_len, n_embd, num_heads, causal, p_dropout, ln_eps, bias, backend):
    """
    batch_size = 1, seq_len = 128, n_embd = 256, num_heads = 4, causal = True, p_dropout = 0.0, ln_eps = 1e-05, bias = False, backend = <minitorch.tensor_ops.TensorBackend object at 0x7f8d6f3d9000>
    batch_size = 16, seq_len = 32, n_embd = 64, num_heads = 4, causal = True, p_dropout = 0.0, ln_eps = 1e-05, bias = False, backend = <minitorch.tensor_ops.TensorBackend object at 0x7f8d6f3d9000>
    """
    np.random.seed(10)
    torch.manual_seed(10)

    data = np.random.randn(batch_size, seq_len, n_embd)
    X    = minitorch.tensor_from_numpy(data.copy(), backend, True)
    X_   = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    layer_ = torch.nn.TransformerEncoderLayer(
        d_model=n_embd, nhead=num_heads, dim_feedforward=256, dropout=0,
        activation=lambda x: torch.nn.functional.gelu(x, approximate='tanh'),
        # activation="relu", layer_norm_eps=ln_eps,
        batch_first=True, norm_first=True, bias=bias, dtype=torch.float32
    )

    layer = minitorch.TransformerLayer(
        n_embd=n_embd, n_head=num_heads, p_dropout=0, ln_eps=layer_.norm1.eps, 
        bias=bias, backend=backend
    )
    

    # FFN Weights
    w_ffn_in = layer_.linear1.weight.detach().numpy().T.copy()
    w_ffn_out = layer_.linear2.weight.detach().numpy().T.copy()
    
    # Transformer Weights
    w_qkv = layer_.self_attn.in_proj_weight.detach().numpy().T.copy() # (n_embd, 3*n_embd)
    w_q_, w_k_, w_v_ = [w.copy() for w in np.split(w_qkv, 3, -1)] # 3 * (n_embd, n_embd)
    w_out_ = layer_.self_attn.out_proj.weight.detach().numpy().T.copy()

    # Set weights (LN will be 0s and 1s so it's ok to ignore for now)
    layer.attention.q_projection.weights.value = minitorch.tensor_from_numpy(w_q_, backend=backend, requires_grad=True)
    layer.attention.k_projection.weights.value = minitorch.tensor_from_numpy(w_k_, backend=backend, requires_grad=True)
    layer.attention.v_projection.weights.value = minitorch.tensor_from_numpy(w_v_, backend=backend, requires_grad=True)
    layer.attention.out_projection.weights.value = minitorch.tensor_from_numpy(w_out_, backend=backend, requires_grad=True)
    layer.ff.linear_in.weights.value = minitorch.tensor_from_numpy(w_ffn_in, backend=backend, requires_grad=True)
    layer.ff.linear_out.weights.value = minitorch.tensor_from_numpy(w_ffn_out, backend=backend, requires_grad=True)

    # Mask for Torch
    M = torch.triu(-float("inf")*torch.ones(seq_len, seq_len),1)

    result = layer(X)
    result_ = layer_(X_, M)

    assert result is not None
    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5,
        rtol=1e-5
    )

    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        X.grad.to_numpy(), 
        X_.grad.detach().numpy(), 
        atol=1e-5,
        rtol=1e-5
    )

    # Let's check some weights
    # np.testing.assert_allclose(
    #     layer.attention.out_projection.weights.value.grad.to_numpy(), 
    #     layer_.self_attn.out_proj.weight.grad.detach().numpy().T,
    #     atol=1e-5, 
    #     rtol=1e-5
    # )
    # np.testing.assert_allclose(
    #     layer.ff.linear_out.weights.value.grad.to_numpy(),
    #     layer_.linear2.weight.grad.detach().numpy().T, 
    #     atol=1e-5,
    #     rtol=1e-5
    # )
    # np.testing.assert_allclose(
    #     layer.ff.linear_in.weights.value.grad.to_numpy(),
    #     layer_.linear1.weight.grad.detach().numpy().T.astype(np.float32), 
    #     atol=1e-5,
    #     rtol=1e-5
    # )


@pytest.mark.a2_3
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [5])
@pytest.mark.parametrize("n_vocab", [10])
@pytest.mark.parametrize("n_embd",  [9])
@pytest.mark.parametrize("n_head",  [3])
@pytest.mark.parametrize("n_positions", [10])
@pytest.mark.parametrize("n_layer", [1])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize("ln_eps", [1e-5])
@pytest.mark.parametrize("bias", [True])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_decoder_lm(batch_size, seq_len, n_vocab, n_embd, n_head, n_layer, n_positions, dropout, ln_eps, bias, backend):

    np.random.seed(19943)

    x = np.random.randint(low=0, high=n_vocab, size=(batch_size, seq_len))

    layer = minitorch.DecoderLM(
        n_vocab=n_vocab, n_embd=n_embd, n_head=n_head, n_positions=n_positions, 
        n_layer=n_layer, p_dropout=dropout, ln_eps=ln_eps, bias=bias, backend=backend)

    result = layer(minitorch.tensor(x.tolist(), backend=backend, requires_grad=True))

    assert result.shape == (batch_size, seq_len, n_vocab)

    result.sum().backward()

    assert layer.position_embeddings.weights.value.grad is not None
    assert layer.token_embeddings.weights.value.grad is not None