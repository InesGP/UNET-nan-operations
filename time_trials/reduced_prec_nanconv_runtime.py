import numpy as np
import torch
import torch.nn.functional as F
import time
import pickle
import os
from typing import Optional
from itertools import product

# class Conv2d:
#     def __init__(
#         self,
#         apply_nan: bool = False,
#         padding: int = 0,
#         stride: int = 1,
#         kernel: torch.Tensor = None,
#         bias: torch.Tensor = None,
#         precision: torch.dtype = torch.float32,
#         threshold: Optional[float] = None
#     ):
#         self.padding = padding
#         self.stride = stride
#         self.precision = precision
#         self.apply_nan = apply_nan

#         self.kernel = kernel.to(dtype=precision)

#         if threshold is not None:
#             self.threshold = threshold

#         self.bias = bias.to(dtype=precision) if bias is not None else None

#     def get_val(self, input_padded, mean_val, n, c_in, i_h, i_w):
#         val = input_padded[n][c_in][i_h][i_w]
#         if self.apply_nan:
#             val = mean_val if torch.isnan(val) else val
#         return val

#     def conv(self, input_padded, n):
#         for c_out, h_out, w_out in product(range(self.kernel.shape[0]), range(self.output.shape[2]), range(self.output.shape[3])):
#             mean_val = None

#             if self.apply_nan:
#                 patch_vals = []
#                 nan_count = 0
#                 total_count = self.C_in * self.K_h * self.K_w

#                 for c_in, k_h, k_w in product(range(self.C_in), range(self.K_h), range(self.K_w)):
#                     i_h = h_out * self.stride + k_h
#                     i_w = w_out * self.stride + k_w
#                     val = input_padded[n][c_in][i_h][i_w]
#                     if torch.isnan(val):
#                         nan_count += 1
#                         if self.threshold and nan_count / total_count >= self.threshold:
#                             self.output[n][c_out][h_out][w_out] = torch.tensor(float('nan'), dtype=self.precision)
#                             break
#                     else:
#                         patch_vals.append(val)
#                 else:
#                     mean_val = sum(patch_vals) / len(patch_vals) if patch_vals else 0.0
#                     mean_val = mean_val.to(dtype=self.precision)
#                 if nan_count / total_count >= self.threshold:
#                     continue

#             acc = sum([
#                 self.get_val(input_padded, mean_val, n, c_in, h_out * self.stride + k_h, w_out * self.stride + k_w) *
#                 self.kernel[c_out][c_in][k_h][k_w]
#                 for c_in, k_h, k_w in product(range(self.C_in), range(self.K_h), range(self.K_w))
#             ])

#             if self.bias is not None:
#                 acc += self.bias[c_out]

#             self.output[n][c_out][h_out][w_out] = acc

#     def __call__(self, input: torch.Tensor) -> torch.Tensor:
#         N, C_in, H_in, W_in = input.shape
#         C_out, _, K_h, K_w = self.kernel.shape
#         stride = self.stride
#         padding = self.padding
#         self.C_in = C_in
#         self.K_h = K_h
#         self.K_w = K_w

#         input_padded = F.pad(input, (padding, padding, padding, padding)).to(self.precision)
#         H_pad, W_pad = input_padded.shape[2:]

#         H_out = (H_pad - K_h) // stride + 1
#         W_out = (W_pad - K_w) // stride + 1
#         self.output = torch.zeros((N, C_out, H_out, W_out), dtype=self.precision)

#         for n in range(N):
#             self.conv(input_padded, n)

#         return self.output


class Conv2d:
    def __init__(
        self,
        apply_nan: bool = False,
        padding: int = 0,
        stride: int = 1,
        kernel: torch.Tensor = None,
        bias: torch.Tensor = None,
        precision: np.dtype = torch.float32,
        threshold: Optional[float] = None
    ):
        self.padding = padding
        self.stride = stride
        self.precision = precision
        self.apply_nan = apply_nan

        self.kernel = kernel.to(dtype=precision)

        if threshold is not None:
            self.threshold = threshold
        
        if bias is not None:
            self.bias = bias.to(dtype=precision)
        else:
            self.bias = None


    def get_val(self, input_padded, mean_val, n, c_in, i_h, i_w):
        val = input_padded[n][c_in][i_h][i_w]
        if self.apply_nan:
            val = mean_val if torch.isnan(val) else val
        return val


    def conv(self, input_padded, n):
        for c_out, h_out, w_out in product(
            range(self.kernel.shape[0]), 
            range(self.output.shape[2]), 
            range(self.output.shape[3])
        ):
            mean_val = None

            if self.apply_nan:
                patch_vals = []
                nan_count = 0
                total_count = self.C_in * self.K_h * self.K_w

                for c_in, k_h, k_w in product(range(self.C_in), range(self.K_h), range(self.K_w)):
                    i_h = h_out * self.stride + k_h
                    i_w = w_out * self.stride + k_w
                    val = input_padded[n][c_in][i_h][i_w]
                    if torch.isnan(val):
                        nan_count += 1
                        if self.threshold and nan_count / total_count >= self.threshold:
                            self.output[n][c_out][h_out][w_out] = torch.tensor(float('nan'), dtype=self.precision)
                            break
                    else:
                        patch_vals.append(val)
                else:
                    mean_val = sum(patch_vals) / len(patch_vals) if patch_vals else 0.0
                    mean_val = mean_val.to(dtype=self.precision)
                # If threshold was exceeded and we broke early, skip this iteration
                if nan_count / total_count >= self.threshold:
                    continue

            # Compute the convolution
            acc = sum([
                self.get_val(input_padded, mean_val, n, c_in, h_out * self.stride + k_h, w_out * self.stride + k_w) *
                self.kernel[c_out][c_in][k_h][k_w]
                for c_in, k_h, k_w in product(range(self.C_in), range(self.K_h), range(self.K_w))
            ])

            if self.bias is not None:
                acc += self.bias[c_out]

            self.output[n][c_out][h_out][w_out] = acc



    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """
        Perform 2D convolution using scalar operations only (no np.sum or broadcasting).
        Args:
            input (torch.Tensor): shape (N, C_in, H_in, W_in)
        Returns:
            output (torch.Tensor): shape (N, C_out, H_out, W_out)
        """
        N, C_in, H_in, W_in = input.shape
        C_out, _, K_h, K_w = self.kernel.shape
        stride = self.stride
        padding = self.padding
        self.C_in = C_in
        self.K_h = K_h
        self.K_w = K_w

        input_padded = torch.nn.functional.pad(input, (self.padding, self.padding, self.padding, self.padding)).to(self.precision)
        _, _, H_pad, W_pad = input_padded.shape


        H_pad, W_pad = H_in + 2 * padding, W_in + 2 * padding

        H_out = (H_pad - K_h) // stride + 1
        W_out = (W_pad - K_w) // stride + 1
        self.output = torch.zeros((N, C_out, H_out, W_out), dtype=self.precision)

        [self.conv(input_padded, n) for n in range(N) ]
        

        # if self.apply_nan:
        #     [self.nan_conv(input_padded, n) for n in range(N) ]
        #     # [self.nan_conv(input_padded, n, c_out, h_out, w_out) for n, c_out, h_out, w_out in product(range(N), range(C_out), range(H_out), range(W_out)) ]
        # else:
        #     [self.manual_conv(input_padded, n, c_out, h_out, w_out) for n, c_out, h_out, w_out in product(range(N), range(C_out), range(H_out), range(W_out)) ]

        return self.output



def insert_nans(tensor: torch.Tensor, percentage: float):
    mask = torch.rand_like(tensor) < percentage
    tensor[mask] = torch.tensor(float('nan'), dtype=tensor.dtype)
    return tensor

def insert_border_nans_until_percentage(tensor: torch.Tensor, percentage: float):
    assert tensor.dim() == 4, "Expected tensor of shape (N, C, H, W)"
    N, C, H, W = tensor.shape
    total_elements = N * C * H * W
    num_nans_target = int(total_elements * percentage)

    nan_count = 0
    border_width = 0
    mask = torch.zeros((H, W), dtype=torch.bool)

    while nan_count < num_nans_target:

        border_width += 1

        # Create a new border mask layer
        new_mask = torch.zeros_like(mask)
        new_mask[:border_width, :] = True     # top
        new_mask[-border_width:, :] = True    # bottom
        new_mask[:, :border_width] = True     # left
        new_mask[:, -border_width:] = True    # right

        # Add new border layer (exclude already masked areas)
        to_add = new_mask & (~mask)
        new_pixels = to_add.sum().item() * N * C
        mask |= to_add
        nan_count += new_pixels

    # Apply NaNs to selected border positions for all N, C
    for n in range(N):
        for c in range(C):
            tensor[n, c][mask] = float('nan')

    return tensor


if __name__ == "__main__":
    torch.manual_seed(0)

    cputime = {}

    for nan_presence in [1.5, 2, 3, 4, 5, 7, 10]:
        cputime[nan_presence] = {'ManualConv': [], 'NaNConv': []}

        for matrix_size in [10, 20, 28, 50, 100, 200, 256, 500, 1000]:
            precision = torch.float32

            input_tensor = torch.rand(1, 1, matrix_size, matrix_size, dtype=precision)
            input_tensor = insert_nans(input_tensor, 1 - (matrix_size // nan_presence) / matrix_size)
            # input_tensor = insert_border_nans_until_percentage(input_tensor, 1 - ( matrix_size//nan_presence)/matrix_size )
            print(f'Input tensor shape: {input_tensor.shape}')

            kernel = torch.tensor([[[[1.0, 0.0], [0.0, -1.0]]]], dtype=precision)
            bias = torch.tensor([0.5], dtype=precision)
            print(f'Kernel tensor shape: {kernel.shape}')

            nanconv = Conv2d(apply_nan=True, padding=0, stride=1, kernel=kernel, bias=bias, threshold=0.5, precision=precision)
            start_time = time.perf_counter()
            output = nanconv(input_tensor)
            print(output.dtype)
            end_time = time.perf_counter()
            cputime[nan_presence]['NaNConv'].append(end_time - start_time)
            print("Elapsed time: {:.6f} seconds".format(end_time - start_time))

            manualconv = Conv2d(apply_nan=False, padding=0, stride=1, kernel=kernel, bias=bias, precision=precision)
            start_time = time.process_time()
            output = manualconv(input_tensor)
            print(output.dtype)
            end_time = time.process_time()
            cputime[nan_presence]['ManualConv'].append(end_time - start_time)
            print("Elapsed time: {:.6f} seconds".format(end_time - start_time))

    pickle.dump(cputime, open(f"/home/inesgp/nanconv_unittests/basic_runtimes/gpu_float32_random_idx_nanconv_cputime_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
