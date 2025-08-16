import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps
from minitorch.cuda_ops import CudaOps
from minitorch.fast_ops import FastOps
from minitorch import DecoderLM
import random
import numpy as np

import cProfile, pstats

datatype = np.float32
datasize = 4

def test_embedding():
    backend_cuda = minitorch.TensorBackend(CudaKernelOps)
    data = [89, 276, 381, 346, 268, 312, 297, 342, 791, 1301, 374, 80, 330, 334, 327, 449, 402, 276, 259, 336, 460, 342, 288, 295, 87, 394, 977, 342, 285, 317, 80, 1103, 290, 74, 596, 442, 389, 366, 347, 268, 1185, 487, 1110, 478, 384, 1027, 420, 472, 362, 268, 291, 437, 1326, 326, 291, 373, 363, 766, 344, 407, 261, 80, 531, 736, 270, 291, 556, 374, 73, 582, 848, 1326, 1028, 644, 266, 530, 511, 428, 259, 268, 346, 465, 318, 339, 402, 334, 495, 73, 78, 828, 86, 343, 259, 340, 487, 259, 280, 262, 330, 352, 259, 268, 346, 283, 384, 280, 292, 390, 259, 280, 78, 828, 390, 80, 342, 278, 72, 78, 787, 259, 376, 671, 69, 614, 270, 0]
    x = minitorch.tensor([data], backend=backend_cuda)

    layer = minitorch.Embedding(num_embeddings=2000, embedding_dim=36, backend=backend_cuda)

    res = layer(x)
    
    
    print("DONE")


def test_tensor():
    # backend_cuda = minitorch.TensorBackend(CudaKernelOps) 
    backend_cuda = minitorch.TensorBackend(CudaOps)
    backend_simple = minitorch.SimpleBackend

    np_data = np.arange(10).reshape(2, 5)
    # res = minitorch.tensor(np_data)
    # # res = minitorch.Tensor.make([np_data], tuple([]), backend=backend_simple)
    # print(res)
    # print(res.shape)
    # print(res.sum())

    # a = minitorch.tensor(np_data.tolist(), backend=backend_cuda)
    # b = minitorch.tensor(np_data.tolist(), backend=backend_cuda)
    # print(a + b)

    # def flatten(ls):
    #     print("flat")
    #     if isinstance(ls, (list, tuple)):
    #         return [y for x in ls for y in flatten(x)]
    #     else:
    #         return [ls]
    
    # print(flatten([i for i in range(10)]))


def test_tensor_from_numpy():
    backend_cuda = minitorch.TensorBackend(CudaKernelOps) 
    backend_simple = minitorch.SimpleBackend

    data = np.arange(10).reshape(2, 5).astype(datatype)
    a = minitorch.Tensor(
        v=minitorch.TensorData(
            storage = data.flatten(), 
            shape   = data.shape,
            strides = tuple(i // 8 for i in data.strides)
        ),
        backend=backend_cuda
    )

    b = a.view(5, 2)
    c = a.permute(1, 0)
    d = b + c

    b_ = data.reshape(5, 2)
    c_ = data.T
    d_ = b_ + c_

    tuple_div_eight = lambda x: (i // datasize for i in x)

    print(a._tensor._strides)
    print(b.shape, b._tensor._strides, b_.shape, np.array(b_.strides) // datasize)
    print(c.shape, c._tensor._strides, c_.shape, np.array(c_.strides) // datasize)
    print(d.shape, d._tensor._strides, d_.shape, np.array(d_.strides) // datasize)
    print(d.to_numpy() == d_)


def main():
    batch_size = 64
    # backend = minitorch.TensorBackend(CudaOps) 
    backend = minitorch.TensorBackend(CudaKernelOps) 
    # backend = minitorch.TensorBackend(FastOps) 

    config = {
        'n_vocab'     : 1000,  # vocab_size
        'n_embd'      : 36,   # n_embed
        'n_head'      : 3,    # n_head
        'n_positions' : 128,  # n_ctx == n_positions
        'n_layer'     : 1,    # n_layer
        'p_dropout'   : 0.1,  # x_pdrop
        'ln_eps'      : 1e-5, # layer_norm_epsilon
        # 'backend'     : minitorch.TensorBackend(CudaKernelOps) 
        'backend'     : backend
    }

    model = DecoderLM(**config)

    x = [[random.randint(0, config['n_vocab']-1) for _ in range(config['n_positions'])]
         for _ in range(batch_size)]
    x = minitorch.tensor(x, backend=backend)

    # params = model.parameters()
    # for v in params:
    #     print(type(v), v.value.shape)

    profiler = cProfile.Profile()
    profiler.enable()
    #####
    logits = model(x)
    #####
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()

    assert(logits.backend == x.backend)


if __name__ == '__main__':
    main()
    # test_tensor()
    # test_tensor_from_numpy()
    # test_embedding()