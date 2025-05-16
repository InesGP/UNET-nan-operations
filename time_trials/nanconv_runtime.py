import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import pickle
import os
from typing import Optional
from itertools import product

class ManualConv2d:
    def __init__(
        self,
        padding: int = 0,
        stride: int = 1,
        kernel: torch.Tensor = None,
        bias: torch.Tensor = None,
        precision: np.dtype = torch.float32
    ):
        self.padding = padding
        self.stride = stride
        self.precision = precision

        self.kernel = kernel.to(dtype=precision)
        
        if bias is not None:
            self.bias = bias.to(dtype=precision)
        else:
            self.bias = None

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

        input_padded = torch.nn.functional.pad(input, (self.padding, self.padding, self.padding, self.padding))
        _, _, H_pad, W_pad = input_padded.shape


        H_pad, W_pad = H_in + 2 * padding, W_in + 2 * padding
        output = torch.zeros((N, C_in, H_pad, W_pad), dtype=self.precision)


        H_out = (H_pad - K_h) // stride + 1
        W_out = (W_pad - K_w) // stride + 1
        output = torch.zeros((N, C_out, H_out, W_out), dtype=self.precision)

        for n in range(N):
            for c_out in range(C_out):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        acc = 0.0
                        for c_in in range(C_in):
                            for k_h in range(K_h):
                                for k_w in range(K_w):
                                    i_h = h_out * stride + k_h
                                    i_w = w_out * stride + k_w
                                    acc += input_padded[n][c_in][i_h][i_w] * self.kernel[c_out][c_in][k_h][k_w]
                        if self.bias is not None:
                            acc += self.bias[c_out]
                        output[n][c_out][h_out][w_out] = acc

        return output


class NaNConv2d(nn.Module):
    def __init__(
        self,
        padding: int = 0,
        stride: int = 1,
        threshold: float = 0.5,
        kernel: torch.Tensor = None,
        bias: torch.Tensor = None,
        precision: torch.dtype = torch.float32
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.threshold = threshold
        self.precision = precision

        self.kernel = kernel.to(dtype=precision)
        self.bias = bias.to(dtype=precision) if bias is not None else None

        self.out_channels, self.in_channels, self.kh, self.kw = self.kernel.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        B, C, H, W = x.shape

        oh = (H - self.kh) // self.stride + 1
        ow = (W - self.kw) // self.stride + 1

        output = torch.empty(B, self.out_channels, oh, ow, dtype=self.precision)

        for b in range(B):
            for oc in range(self.out_channels):
                for i in range(oh):
                    for j in range(ow):
                        x_start = i * self.stride
                        y_start = j * self.stride

                        nan_count = 0
                        total = 0
                        val_sum = 0.0
                        patch_vals = []  # buffer for mean-substitution pass

                        # First pass: gather patch and compute stats
                        for ic in range(self.in_channels):
                            for m in range(self.kh):
                                for n in range(self.kw):
                                    val = x[b][ic][x_start + m][y_start + n]
                                    patch_vals.append((ic, m, n, val))
                                    total += 1
                                    if torch.isnan(val):
                                        nan_count += 1
                                    else:
                                        val_sum += float(val)

                        nan_ratio = nan_count / total
                        if nan_ratio >= self.threshold or total == nan_count:
                            output[b][oc][i][j] = float('nan')
                            continue

                        patch_mean = val_sum / (total - nan_count)
                        conv_sum = 0.0

                        # Convolution with mean-substitution
                        for ic, m, n, val in patch_vals:
                            val = patch_mean if torch.isnan(val) else float(val)
                            kval = float(self.kernel[oc][ic][m][n])
                            conv_sum += val * kval

                        if self.bias is not None:
                            conv_sum += float(self.bias[oc])

                        output[b][oc][i][j] = conv_sum

        return output



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

    # def manual_conv(self, input_padded, n, c_out, h_out, w_out):

    #     acc = 0.0
    #     for c_in, k_h, k_w in product(range(self.C_in), range(self.K_h), range(self.K_w)):
    #         i_h = h_out * self.stride + k_h
    #         i_w = w_out * self.stride + k_w
    #         acc += input_padded[n][c_in][i_h][i_w] * self.kernel[c_out][c_in][k_h][k_w]
    #     if self.bias is not None:
    #         acc += self.bias[c_out]

    #     self.output[n][c_out][h_out][w_out] = acc

    
    # def nan_conv(self, input_padded, n, c_out, h_out, w_out):

    #     # Calculate NaN ratio
    #     nan_ratio = torch.isnan(input_padded[n]).sum() / (input_padded[n].size(0) * input_padded[n].size(1) * input_padded[n].size(2))
    #     if nan_ratio >= self.threshold: 
    #         self.count += 1
    #         self.output[n][c_out][h_out][w_out] = float('nan')
    #     else:
    #         acc = 0.0
    #         for c_in in range(self.C_in):
    #             #Calling torch operations to calculate mean and substitution
    #             insert = input_padded[n][c_in][~torch.isnan(input_padded[n][c_in])].flatten()
    #             image_patch = torch.nan_to_num(input_padded[n][c_in], nan=insert.mean())

    #             for k_h in range(self.K_h):
    #                 for k_w in range(self.K_w):
    #                     i_h = h_out * self.stride + k_h
    #                     i_w = w_out * self.stride + k_w
                        
    #                     acc += image_patch[i_h][i_w] * self.kernel[c_out][c_in][k_h][k_w]
    #         if self.bias is not None:
    #             acc += self.bias[c_out]

    #         self.output[n][c_out][h_out][w_out] = acc


    def nan_conv(self, input_padded, n):
        for c_out, h_out, w_out in product(
            range(self.kernel.shape[0]),
            range(self.output.shape[2]),
            range(self.output.shape[3])
        ):
            acc = 0.0
            nan_count = 0
            total_count = 0
            patch_vals = []

            for c_in, k_h, k_w in product(range(self.C_in), range(self.K_h), range(self.K_w)):
                i_h = h_out * self.stride + k_h
                i_w = w_out * self.stride + k_w
                val = input_padded[n][c_in][i_h][i_w]
                total_count += 1
                if torch.isnan(val):
                    nan_count += 1
                else:
                    patch_vals.append(val)

            nan_ratio = nan_count / total_count
            if self.threshold is not None and nan_ratio >= self.threshold:
                self.output[n][c_out][h_out][w_out] = float('nan')
                continue


            mean_val = torch.stack(patch_vals).mean() if patch_vals else torch.tensor(0.0, dtype=self.precision)

            for c_in, k_h, k_w in product(range(self.C_in), range(self.K_h), range(self.K_w)):
                i_h = h_out * self.stride + k_h
                i_w = w_out * self.stride + k_w
                val = input_padded[n][c_in][i_h][i_w]
                val = mean_val if torch.isnan(val) else val
                acc += val * self.kernel[c_out][c_in][k_h][k_w]

            if self.bias is not None:
                acc += self.bias[c_out]

            self.output[n][c_out][h_out][w_out] = acc



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
                            self.output[n][c_out][h_out][w_out] = float('nan')
                            break
                    else:
                        patch_vals.append(val)
                else:
                    mean_val = sum(patch_vals) / len(patch_vals) if patch_vals else 0.0
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

        input_padded = torch.nn.functional.pad(input, (self.padding, self.padding, self.padding, self.padding))
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

def insert_nans(tensor: torch.Tensor, percentage: float):
    """
    Randomly inserts NaNs into the tensor with each element having 
    `percentage` chance of becoming NaN.

    Args:
        tensor (torch.Tensor): Input tensor.
        percentage (float): Probability (between 0 and 1) that each element becomes NaN.

    Returns:
        torch.Tensor: Tensor with some elements set to NaN.
    """
    mask = torch.rand_like(tensor) < percentage
    tensor[mask] = float('nan')
    return tensor


if __name__ == "__main__":

    torch.manual_seed(0)
    walltime = {}

    # for nan_presence in [2,3,4,5,6,7,8,9,10]:
    for nan_presence in [1.5,2,3,4,5,7,10]:
        walltime[nan_presence] = {'ManualConv': [], 'NaNConv': []}


        for matrix_size in [10,20,28,50,100,200,256,500,1000]:

            input_tensor = torch.rand(1,1,matrix_size,matrix_size)
            # input_tensor = insert_border_nans_until_percentage(input_tensor, 1 - ( matrix_size//nan_presence)/matrix_size )
            input_tensor = insert_nans(input_tensor, 1 - ( matrix_size//nan_presence)/matrix_size )
    
            # input_tensor[:,:, :(matrix_size - matrix_size//nan_presence), :(matrix_size - matrix_size//nan_presence)] = float('nan')

            print(f'Input tensor shape: {input_tensor.shape}')

            # Define a simple kernel (out_channels=1, in_channels=1, 2x2 kernel)
            kernel = torch.tensor(
                [[[[1.0, 0.0],
                [0.0, -1.0]]]],
                dtype=torch.float32
            )
            print(f'Kernel tensor shape: {kernel.shape}')

            # Optional bias
            bias = torch.tensor([0.5], dtype=torch.float32)

            # Instantiate the NaN-aware convolution
            nanconv = Conv2d(
                apply_nan=True,
                padding=0,
                stride=1,
                kernel=kernel,
                bias=bias,
                threshold=0.5,
            )

            # Time the execution
            start_time = time.perf_counter()
            output = nanconv(input_tensor)
            end_time = time.perf_counter()

            # Print the output and timing
            # print("Output:\n", output)
            print("Elapsed time: {:.6f} seconds".format(end_time - start_time))
            walltime[nan_presence]['NaNConv'].append(end_time - start_time)



            # Instantiate the NaN-aware convolution
            manualconv = Conv2d(
                apply_nan=False,
                padding=0,
                stride=1,
                kernel=kernel,
                bias=bias,
            )

            # Time the execution
            start_time = time.perf_counter()
            output = manualconv(input_tensor)
            end_time = time.perf_counter()

            # Print the output and timing
            # print("Output:\n", output)
            print("Elapsed time: {:.6f} seconds".format(end_time - start_time))
            walltime[nan_presence]['ManualConv'].append(end_time - start_time)

    pickle.dump(walltime, open(f"/home/inesgp/nanconv_unittests/basic_runtimes/random_idx_nanconv_walltime_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

    cputime = {}

    # for nan_presence in [2,3,4,5,6,7,8,9,10]:
    for nan_presence in [1.5,2,3,4,5,7,10]:
        cputime[nan_presence] = {'ManualConv': [], 'NaNConv': []}


        for matrix_size in [10,20,28,50,100,200,256,500,1000]:

            input_tensor = torch.rand(1,1,matrix_size,matrix_size)
            # input_tensor = insert_border_nans_until_percentage(input_tensor, 1 - ( matrix_size//nan_presence)/matrix_size )
            input_tensor = insert_nans(input_tensor, 1 - ( matrix_size//nan_presence)/matrix_size )

            # input_tensor[:,:, :(matrix_size - matrix_size//nan_presence), :(matrix_size - matrix_size//nan_presence)] = float('nan')

            print(f'Input tensor shape: {input_tensor.shape}')

            # Define a simple kernel (out_channels=1, in_channels=1, 2x2 kernel)
            kernel = torch.tensor(
                [[[[1.0, 0.0],
                [0.0, -1.0]]]],
                dtype=torch.float32
            )
            print(f'Kernel tensor shape: {kernel.shape}')

            # Optional bias
            bias = torch.tensor([0.5], dtype=torch.float32)

            # Instantiate the NaN-aware convolution
            nanconv = Conv2d(
                apply_nan=True,
                padding=0,
                stride=1,
                kernel=kernel,
                bias=bias,
                threshold=0.5,
            )

            # Time the execution
            start_time = time.process_time()
            output = nanconv(input_tensor)
            end_time = time.process_time()

            # Print the output and timing
            # print("Output:\n", output)
            print("Elapsed time: {:.6f} seconds".format(end_time - start_time))
            cputime[nan_presence]['NaNConv'].append(end_time - start_time)



            # Instantiate the NaN-aware convolution
            manualconv = Conv2d(
                apply_nan=False,
                padding=0,
                stride=1,
                kernel=kernel,
                bias=bias,
            )

            # Time the execution
            start_time = time.process_time()
            output = manualconv(input_tensor)
            end_time = time.process_time()

            # Print the output and timing
            # print("Output:\n", output)
            print("Elapsed time: {:.6f} seconds".format(end_time - start_time))
            cputime[nan_presence]['ManualConv'].append(end_time - start_time)

    pickle.dump(cputime, open(f"/home/inesgp/nanconv_unittests/basic_runtimes/random_idx_nanconv_cputime_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
