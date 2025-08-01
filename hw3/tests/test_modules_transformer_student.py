import pytest
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps

import numpy as np
import torch
import torch.nn as nn
import numba
import os

datatype = np.float32

########################################################################
# ASSIGNMENT 2 PROBLEM 4
########################################################################

_BACKENDS = [pytest.param(
                 minitorch.TensorBackend(CudaKernelOps), 
                 marks=pytest.mark.skipif(not numba.cuda.is_available(), reason="No GPU")
             )]

def load_numpy_array(arr_path):
    with open(arr_path, 'rb') as f:
        return np.load(f)

################################ MULTIHEADATTENTION ########################################

@pytest.mark.a2_4
@pytest.mark.parametrize("batch_size",  [1, 128])
@pytest.mark.parametrize("queries_len", [32, 40])
@pytest.mark.parametrize("n_embd",      [64, 256])
@pytest.mark.parametrize("num_heads",   [1, 4, 8])
@pytest.mark.parametrize("p_dropout",   [0.0])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_multihead_attention_student(batch_size, queries_len, n_embd, num_heads, p_dropout, backend):
    test_dir = f'./tests/data/multihead_attention'
    test_str = '_'.join(map(str, (batch_size, queries_len, n_embd, num_heads)))

    data = load_numpy_array(os.path.join(test_dir, f'{test_str}_data.npy'))
    w_q = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_q.npy'))
    w_k = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_k.npy'))
    w_v = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_v.npy'))
    w_out = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_out.npy'))
    result_ = load_numpy_array(os.path.join(test_dir, f'{test_str}_result.npy'))
    x_grad = load_numpy_array(os.path.join(test_dir, f'{test_str}_x_grad.npy'))
    w_q_grad = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_q_grad.npy'))
    w_k_grad = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_k_grad.npy'))
    w_v_grad = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_v_grad.npy'))
    w_out_grad = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_out_grad.npy'))

    X    = minitorch.tensor_from_numpy(data, backend, True)

    layer = minitorch.MultiHeadAttention(n_embd, num_heads, True, p_dropout, bias=False, backend=backend)
    
    layer.q_projection.weights.value   = minitorch.tensor_from_numpy((w_q), backend=backend, requires_grad=True)
    layer.k_projection.weights.value   = minitorch.tensor_from_numpy((w_k), backend=backend, requires_grad=True)
    layer.v_projection.weights.value   = minitorch.tensor_from_numpy((w_v), backend=backend, requires_grad=True)
    layer.out_projection.weights.value = minitorch.tensor_from_numpy((w_out), backend=backend, requires_grad=True)

    result = layer(X)

    np.testing.assert_allclose(result.to_numpy(), result_, atol=1e-5, rtol=1e-5)

    result.sum().backward()
    
    np.testing.assert_allclose(X.grad.to_numpy(), x_grad, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(layer.out_projection.weights.value.grad.to_numpy(), w_out_grad, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(layer.q_projection.weights.value.grad.to_numpy(), w_q_grad, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(layer.k_projection.weights.value.grad.to_numpy(), w_k_grad, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose( layer.v_projection.weights.value.grad.to_numpy(), w_v_grad, atol=1e-5, rtol=1e-5)


################################ FFN ########################################

@pytest.mark.a2_4
@pytest.mark.parametrize("batch_size", [1, 128])
@pytest.mark.parametrize("seq_len", [5, 40])
@pytest.mark.parametrize("n_embd",  [9, 256])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_feedforward_layer_student(batch_size, seq_len, n_embd, dropout, backend):

    np.random.seed(11868)

    data = np.random.randn(batch_size, seq_len, n_embd).astype(datatype)

    layer = minitorch.FeedForward(n_embd=n_embd, p_dropout=dropout, bias=True, backend=backend)

    X = minitorch.tensor(data.tolist(), backend=backend, requires_grad=True)

    result = layer(X)

    assert result is not None
    assert not np.isnan(result.to_numpy()).any()

    result.sum().backward()

    assert X.grad is not None
    assert layer.linear_in.weights.value.grad is not None
    assert layer.linear_out.weights.value.grad is not None

################################ TRANSFORMER LAYER ########################################

@pytest.mark.a2_4
@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("seq_len",   [40])
@pytest.mark.parametrize("n_embd",    [256])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("p_dropout", [0.0])
@pytest.mark.parametrize("ln_eps", [1e-5])
@pytest.mark.parametrize("bias", [False])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_transformer_layer_1_student(batch_size, seq_len, n_embd, num_heads, p_dropout, ln_eps, bias, backend):
    test_dir = f'./tests/data/transformer_layer_1'
    test_str = '_'.join(map(str, (batch_size, seq_len, n_embd, num_heads)))

    data = load_numpy_array(os.path.join(test_dir, f'{test_str}_data.npy'))
    w_ffn_in = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_ffn_in.npy'))
    w_ffn_out = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_ffn_out.npy'))
    w_q_ = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_q.npy'))
    w_k_ = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_k.npy'))
    w_v_ = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_v.npy'))
    w_out_ = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_out.npy'))
    result_ = load_numpy_array(os.path.join(test_dir, f'{test_str}_result.npy'))
    x_grad = load_numpy_array(os.path.join(test_dir, f'{test_str}_x_grad.npy'))
    
    X    = minitorch.tensor_from_numpy(data.copy(), backend, True)
    
    layer = minitorch.TransformerLayer(
        n_embd=n_embd, n_head=num_heads, p_dropout=p_dropout, ln_eps=ln_eps, 
        bias=bias, backend=backend
    )
    
    # Set weights (LN will be 0s and 1s so it's ok to ignore for now)
    layer.attention.q_projection.weights.value = minitorch.tensor_from_numpy(w_q_, backend=backend, requires_grad=True)
    layer.attention.k_projection.weights.value = minitorch.tensor_from_numpy(w_k_, backend=backend, requires_grad=True)
    layer.attention.v_projection.weights.value = minitorch.tensor_from_numpy(w_v_, backend=backend, requires_grad=True)
    layer.attention.out_projection.weights.value = minitorch.tensor_from_numpy(w_out_, backend=backend, requires_grad=True)
    layer.ff.linear_in.weights.value = minitorch.tensor_from_numpy(w_ffn_in, backend=backend, requires_grad=True)
    layer.ff.linear_out.weights.value = minitorch.tensor_from_numpy(w_ffn_out, backend=backend, requires_grad=True)

    result = layer(X)

    assert result is not None
    np.testing.assert_allclose(result.to_numpy(), result_, atol=1e-5, rtol=1e-5)

    result.sum().backward()

    np.testing.assert_allclose(X.grad.to_numpy(), x_grad, atol=1e-5, rtol=1e-5)


@pytest.mark.a2_4
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("seq_len",   [4, 32])
@pytest.mark.parametrize("n_embd",    [16, 32])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("p_dropout", [0.0])
@pytest.mark.parametrize("ln_eps", [1e-5])
@pytest.mark.parametrize("bias", [False])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_transformer_layer_2_student(batch_size, seq_len, n_embd, num_heads, p_dropout, ln_eps, bias, backend):
    test_dir = f'./tests/data/transformer_layer_2'
    test_str = '_'.join(map(str, (batch_size, seq_len, n_embd, num_heads)))

    data = load_numpy_array(os.path.join(test_dir, f'{test_str}_data.npy'))
    w_ffn_in = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_ffn_in.npy'))
    w_ffn_out = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_ffn_out.npy'))
    w_q_ = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_q.npy'))
    w_k_ = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_k.npy'))
    w_v_ = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_v.npy'))
    w_out_ = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_out.npy'))
    result_ = load_numpy_array(os.path.join(test_dir, f'{test_str}_result.npy'))
    x_grad = load_numpy_array(os.path.join(test_dir, f'{test_str}_x_grad.npy'))
    w_ffn_in_grad = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_ffn_in_grad.npy'))
    w_ffn_out_grad = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_ffn_out_grad.npy'))
    w_out_grad = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_out_grad.npy'))
    w_q_grad = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_q_grad.npy'))
    w_k_grad = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_k_grad.npy'))
    w_v_grad = load_numpy_array(os.path.join(test_dir, f'{test_str}_w_v_grad.npy'))
    
    X    = minitorch.tensor_from_numpy(data.copy(), backend, True)
    
    layer = minitorch.TransformerLayer(
        n_embd=n_embd, n_head=num_heads, p_dropout=p_dropout, ln_eps=ln_eps, 
        bias=bias, backend=backend
    )
    
    # Set weights (LN will be 0s and 1s so it's ok to ignore for now)
    layer.attention.q_projection.weights.value = minitorch.tensor_from_numpy(w_q_, backend=backend, requires_grad=True)
    layer.attention.k_projection.weights.value = minitorch.tensor_from_numpy(w_k_, backend=backend, requires_grad=True)
    layer.attention.v_projection.weights.value = minitorch.tensor_from_numpy(w_v_, backend=backend, requires_grad=True)
    layer.attention.out_projection.weights.value = minitorch.tensor_from_numpy(w_out_, backend=backend, requires_grad=True)
    layer.ff.linear_in.weights.value = minitorch.tensor_from_numpy(w_ffn_in, backend=backend, requires_grad=True)
    layer.ff.linear_out.weights.value = minitorch.tensor_from_numpy(w_ffn_out, backend=backend, requires_grad=True)

    result = layer(X)

    assert result is not None
    np.testing.assert_allclose(result.to_numpy(), result_, atol=1e-5, rtol=1e-5)

    result.sum().backward()

    np.testing.assert_allclose(X.grad.to_numpy(), x_grad, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(layer.attention.out_projection.weights.value.grad.to_numpy(), w_out_grad, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(layer.ff.linear_out.weights.value.grad.to_numpy(), w_ffn_out_grad, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(layer.ff.linear_in.weights.value.grad.to_numpy(), w_ffn_in_grad, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(layer.attention.q_projection.weights.value.grad.to_numpy(), w_q_grad, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(layer.attention.k_projection.weights.value.grad.to_numpy(), w_k_grad, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(layer.attention.v_projection.weights.value.grad.to_numpy(), w_v_grad, atol=1e-5, rtol=1e-5)

################################ DECODER LM ########################################

@pytest.mark.a2_4
@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("seq_len", [40])
@pytest.mark.parametrize("n_vocab", [1000])
@pytest.mark.parametrize("n_embd",  [256])
@pytest.mark.parametrize("n_head",  [8])
@pytest.mark.parametrize("n_positions", [40])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize("ln_eps", [1e-5])
@pytest.mark.parametrize("bias", [True])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_decoder_lm_student(batch_size, seq_len, n_vocab, n_embd, n_head, n_positions, dropout, ln_eps, bias, backend):

    np.random.seed(10)
    x = np.random.randint(low=0, high=n_vocab, size=(batch_size, seq_len))

    layer = minitorch.DecoderLM(
        n_vocab=n_vocab, n_embd=n_embd, n_head=n_head, n_positions=n_positions, 
        p_dropout=dropout, ln_eps=ln_eps, bias=bias, backend=backend)

    result = layer(minitorch.tensor(x.tolist(), backend=backend, requires_grad=True))

    assert result is not None
    assert not np.isnan(result.to_numpy()).any()
    assert result.shape == (batch_size, seq_len, n_vocab)

    result.sum().backward()

    assert layer.position_embeddings.weights.value.grad is not None
    assert layer.token_embeddings.weights.value.grad is not None
    assert not np.isnan(layer.position_embeddings.weights.value.grad.to_numpy()).any()
    assert not np.isnan(layer.token_embeddings.weights.value.grad.to_numpy()).any()