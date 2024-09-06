import cupy as cp
import torch
from torch.utils.dlpack import from_dlpack as torch_from_dlpack
from torch.utils.dlpack import to_dlpack as torch_to_dlpack


_angle_kernel = r'''
extern "C" __global__
void stable_angle(const double* X, const double* Y, double* out, const int N, int M, int D) {
    const int I = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int J = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (I >= N || J >= M) {
        return;
    }

    double norm_x = (double) 0.0;
    double norm_y = (double) 0.0;

    #pragma unroll
    for (int k = 0; k < D; k++) {
        double X_ik = X[I * D + k];
        double Y_jk = Y[J * D + k];
        norm_x = fma(X_ik, X_ik, norm_x);
        norm_y = fma(Y_jk, Y_jk, norm_y);
    }

    norm_x = sqrt(norm_x);
    norm_y = sqrt(norm_y);

    double a = (double) 0.0;
    double b = (double) 0.0;

    #pragma unroll
    for (int k = 0; k < D; k++) {
        double X_ik = X[I * D + k];
        double Y_jk = Y[J * D + k];
        double a1 = norm_y * X_ik - norm_x * Y_jk;
        double a2 = norm_y * X_ik + norm_x * Y_jk;

        a = fma(a1, a1, a);
        b = fma(a2, a2, b);
    }

    a = sqrt(a);
    b = sqrt(b);

    out[I * M + J] = 2.0 * atan2(a, b);
}
'''

_angle_backward = r'''
extern "C" __global__
void stable_angle_backward(const double* X, const double* Y, const double* grad_out, double* out_x, double* out_y, const int N, int M, int D) {
    const int I = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int J = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (I >= N || J >= M) {
        return;
    }

    double norm_x_2 = (double) 0.0;
    double norm_y_2 = (double) 0.0;
    double x_dot_y = (double) 0.0;

    #pragma unroll
    for (int k = 0; k < D; k++) {
        const double X_ik = X[I * D + k];
        const double Y_jk = Y[J * D + k];
        norm_x_2 = fma(X_ik, X_ik, norm_x_2);
        norm_y_2 = fma(Y_jk, Y_jk, norm_y_2);
        x_dot_y = fma(X_ik, Y_jk, x_dot_y);
    }

    const double norm_x = sqrt(norm_x_2);
    const double norm_y = sqrt(norm_y_2);
    const double norm_x_y = norm_x * norm_y;
    const double norm_x_over_y = norm_x / norm_y;
    const double norm_y_over_x = norm_y / norm_x;

    double a_2 = (double) 0.0;
    double b_2 = (double) 0.0;

    #pragma unroll
    for (int k = 0; k < D; k++) {
        const double X_ik = X[I * D + k];
        const double Y_jk = Y[J * D + k];
        const double a_2_k = norm_y * X_ik - norm_x * Y_jk;
        const double b_2_k = norm_y * X_ik + norm_x * Y_jk;

        a_2 = fma(a_2_k, a_2_k, a_2);
        b_2 = fma(b_2_k, b_2_k, b_2);
    }

    const double a = sqrt(a_2);
    const double b = sqrt(b_2);

    const double a_b_2 = a_2 + b_2;
    const double da_atan2 = b / a_b_2;
    const double db_atan2 = -a / a_b_2;

    double grad_output_ij = grad_out[I * M + J];

    #pragma unroll
    for (int k = 0; k < D; k++) {
        const double X_ik = X[I * D + k];
        const double Y_jk = Y[J * D + k];
        const double grad_a_X_ik = ( ((double)2.0 * norm_y_2 - norm_y_over_x * x_dot_y) * X_ik - norm_x_y * Y_jk) / a;
        const double grad_a_Y_jk = ( ((double)2.0 * norm_x_2 - norm_x_over_y * x_dot_y) * Y_jk - norm_x_y * X_ik) / a;
        const double grad_b_X_ik = ( ((double)2.0 * norm_y_2 + norm_y_over_x * x_dot_y) * X_ik + norm_x_y * Y_jk) / b;
        const double grad_b_Y_jk = ( ((double)2.0 * norm_x_2 + norm_x_over_y * x_dot_y) * Y_jk + norm_x_y * X_ik) / b;
        const double incr_x = 2.0 * (da_atan2 * grad_a_X_ik + db_atan2 * grad_b_X_ik) * grad_output_ij;
        const double incr_y = 2.0 * (da_atan2 * grad_a_Y_jk + db_atan2 * grad_b_Y_jk) * grad_output_ij; 

        atomicAdd(&out_x[I * D + k], incr_x);
        atomicAdd(&out_y[J * D + k], incr_y);
    }
}
'''


_cupy_kernel = cp.RawKernel(_angle_kernel, 'stable_angle')
_cupy_kernel.compile()

_cupy_backward_kernel = cp.RawKernel(_angle_backward, 'stable_angle_backward')
_cupy_backward_kernel.compile()


def _stable_angle_cupy(x, y):
    assert x.dtype == y.dtype

    x_cp = cp.fromDlpack(torch_to_dlpack(x))
    y_cp = cp.fromDlpack(torch_to_dlpack(y))

    out = cp.zeros((x.shape[0], y.shape[0]), dtype = x_cp.dtype)

    pt_dim = int(x.shape[1])
    dims = int(x.shape[0]), int(y.shape[0])
    threads_per_block = (16, 16)
    blocks_per_grid = tuple((dims[i] + threads_per_block[i] - 1) // threads_per_block[i] for i in range(2))

    _cupy_kernel(blocks_per_grid, threads_per_block, (x_cp, y_cp, out, dims[0], dims[1], pt_dim))

    cp.cuda.stream.get_current_stream().synchronize()   

    return torch_from_dlpack(out.toDlpack())
 
def _backward_stable_angle_cupy(x, y, grad_out):
    assert x.dtype == y.dtype == grad_out.dtype

    x_cp = cp.fromDlpack(torch_to_dlpack(x))
    y_cp = cp.fromDlpack(torch_to_dlpack(y))

    grad_out_cp = cp.fromDlpack(torch_to_dlpack(grad_out))
    out_x = cp.zeros_like(x_cp)
    out_y = cp.zeros_like(y_cp)

    pt_dim = int(x.shape[1])
    dims = int(x.shape[0]), int(y.shape[0])
    threads_per_block = (16, 16)
    blocks_per_grid = tuple((dims[i] + threads_per_block[i] - 1) // threads_per_block[i] for i in range(2))
  
    _cupy_backward_kernel(blocks_per_grid, threads_per_block, (x_cp, y_cp, grad_out_cp, out_x, out_y, dims[0], dims[1], pt_dim))
  
    cp.cuda.stream.get_current_stream().synchronize()   

    dx = torch_from_dlpack(out_x.toDlpack())
    dy = torch_from_dlpack(out_y.toDlpack())

    return dx, dy


class _StableAngle(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, y):
        if not (x.is_cuda and y.is_cuda):
            raise ValueError("stable_angle only works with CUDA tensors")
        
        ctx.save_for_backward(x, y)
        return _stable_angle_cupy(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors

        if not grad_output.is_cuda:
            raise ValueError("stable_angle backwards requires grad_out be a cuda tensor")
       
        return _backward_stable_angle_cupy(x, y, grad_output)


def stable_angle(x, y):
    return _StableAngle.apply(x, y)