import random
import numpy as np
import torch
import torch.nn.init as init
from torch import nn
from typing import Tuple
from math import prod  
from torch.nn import functional as F
from typing import Tuple, Optional
from itertools import product

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


class NaNConv2d(nn.Module):
    def __init__(
        self,
        train: bool = False,
        bias_presence: bool = True,
        padding: int = 0,
        stride: int = 1,
        threshold: float = 0.5,
        in_channels: int = None,
        out_channels: int = None,
        kernel_size: Tuple = None,
        kernel: torch.Tensor = None,
        bias: torch.Tensor = None,
    ):
        """
        Initializes the NaNConv2d layer.

        Args:
            train (bool): Whether to train the model.
            bias_presence (bool): Whether bias is present in the layer.
            padding (int): Padding size.
            stride (int): Stride size.
            threshold (float): Threshold for NaN ratio.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (tuple): Size of the convolutional kernel.
            kernel (torch.Tensor): Initial kernel weights.
            bias (torch.Tensor): Initial bias values.
        """

        super().__init__()
        # kernel = torch.flipud(torch.fliplr(kernel))
        self.stride = int(stride)
        self.padding = int(padding)
        self.threshold = threshold

        if train is False:
            self.inference(kernel, bias)
        else:
            self.trainer(in_channels, out_channels, kernel_size, bias_presence)

        # print(f'Kernel {self.kernel.shape} Stride {self.stride}, padding {self.padding}, bias {self.bias.shape}, in_channels {self.image_padded.shape}, out_channels {self.output.shape}')

    def inference(self, kernel: torch.Tensor, bias: torch.Tensor) -> None:
        """
        Initializes the NaNConv2d layer at inference with the provided kernel and bias from the original model.

        Args:
            kernel (torch.Tensor): 4D tensor representing the convolutional kernel.
            bias (torch.Tensor): 1D tensor representing the bias values.

        Returns:
            None
        """
        self.bias = bias
        self.kernel = kernel
        self.out_channels, _, self.kernel_height, self.kernel_width = kernel.shape
        # Force values to int
        self.out_channels = int(self.out_channels)
        self.kernel_height = int(self.kernel_height)
        self.kernel_width = int(self.kernel_width)

    # UNDER WORK
    def trainer(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], bias_presence: bool) -> None:
        """
        Initialize the NaNConv2d layer for training a model.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (tuple[int, int]): Size of the convolutional kernel (height, width).
            bias_presence (bool): Whether bias term should be included.

        Returns:
            None
        """

        self.out_channels = int(out_channels)
        self.kernel_height, self.kernel_width = kernel_size
        # Force values to int
        self.kernel_height = int(self.kernel_height)
        self.kernel_width = int(self.kernel_width)

        # Initialize kernel parameters
        self.kernel = nn.Parameter(
            init.xavier_normal_(torch.zeros((self.out_channels, in_channels, self.kernel_height, self.kernel_width)))
        )
        # Initialize bias parameters if bias is present
        if bias_presence:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.bias = None

    # NO LONGER IN USE
    def choose_probability_large(
        self, input: torch.Tensor, kernel: torch.Tensor, min_val: float, max_val: float
    ) -> torch.Tensor:
        """
        Choose probabilities for replacing NaN values with random values for window across channels and batch.

        Args:
            input (torch.Tensor): Input tensor.
            kernel (torch.Tensor): Kernel tensor.
            min_val (float): Minimum value.
            max_val (float): Maximum value.

        Returns:
            torch.Tensor: Processed tensor with NaN values replaced.
        """
        # If all NaN, return input as is
        if torch.isnan(input).all():
            return input
        elif not torch.isnan(input).any():
            return input

        # Flatten input and kernel
        input_flat = input.ravel()  # view(-1)
        kernel_flat = kernel.ravel()  # view(-1)

        # Calculate probabilities for NaN values
        distance_to_max = max_val - kernel_flat
        total_distance = max_val - min_val
        probabilities = 1 - distance_to_max / total_distance

        # Create a mask for NaN values
        nan_mask = torch.isnan(input_flat)

        # Indices of NaN values
        nan_indices = torch.nonzero(nan_mask).squeeze()

        # Generate random values from the distribution
        hist, bin_edges = torch.histogram(input_flat[~nan_mask], bins=5, density=True)
        random_values = torch.tensor(
            random.choices(bin_edges[:-1], weights=(hist / torch.sum(hist)), k=nan_indices.numel())
        )

        # Replace NaN values with random values
        # input_flat[nan_indices] = random_values

        probabilities = torch.mean(probabilities.view(kernel.shape), axis=0).ravel()

        result = torch.where(
            torch.isnan(input_flat) & (probabilities > random.random()),
            random_values[torch.randint(len(random_values), size=input_flat.shape)],
            input_flat,
        )

        return result.view(input.shape)

    def get_window(self, i: int, j: int) -> torch.Tensor:
        """
        Get the receptive field window at position (i, j).

        Args:
            i (int): Horizontal index.
            j (int): Vertical index.

        Returns:
            torch.Tensor: Receptive field window.
        """
        x_start = i * self.stride
        y_start = j * self.stride

        # Calculate the ending point of the receptive field
        x_end = min(x_start + self.kernel_height, self.in_height + 2 * self.padding)
        y_end = min(y_start + self.kernel_width, self.in_width + 2 * self.padding)

        # Extract the image patch
        return self.image_padded[:, :, x_start:x_end, y_start:y_end]

    def apply_threshold(self, kernel: torch.Tensor, i: int, j: int) -> None:
        """
        Apply NaN thresholding to the output tensor window at position (i, j).

        Args:
            kernel (torch.Tensor): Convolution kernel.
            i (int): Horizontal index.
            j (int): Vertical index.
        """

        image_patch = self.get_window(i, j)
        # image_patch = choose_probability_large(image_patch, kernel, torch.min(kernel), torch.max(kernel))

        # Calculate NaN ratio
        # nan_ratio = torch.sum(torch.isnan(image_patch)).item() / image_patch.numel()
        nan_ratio = torch.isnan(image_patch).sum(dim=(1, 2, 3)) / (image_patch.size(1) * image_patch.size(2) * image_patch.size(3))

        # # Use NaN ratio to determine whether NaNs are ignored or convolution output
        # for num, nan_batch in enumerate(image_patch):
        #     print(nan_ratio[num], self.threshold)
        #     if nan_ratio[num].item() >= self.threshold:
        #         self.output[num, :, i, j] = float("nan")
        #     else:
        #         self.output[num, :, i, j] = F.conv2d(torch.nan_to_num(nan_batch), kernel, stride=1, padding=0).squeeze(dim=-1).squeeze(dim=-1)

        nan_mask = nan_ratio >= self.threshold

        insert = image_patch[~torch.isnan(image_patch)].flatten()
        if len(insert) < 1: insert=torch.tensor([0]).float()
        # print(insert.mean())
        # Apply convolution to the batches that do not meet the NaN threshold
        # Practically speaking we apply a convolution on everything and then replace the NaNs for faster computation at the moment
        # NaNs are replaced with the mean of the entire 4D patch
        conv_result = F.conv2d(torch.nan_to_num(image_patch, nan=insert.mean()), kernel, stride=1, padding=0).squeeze(dim=-1).squeeze(dim=-1)

        # Set output to NaN where the threshold is met, otherwise set it to the convolution result
        self.output[:, :, i, j] = torch.where(nan_mask.unsqueeze(1), torch.tensor(float('nan')), conv_result)


        # # Use NaN ratio to determine whether NaNs are ignored or convolution output
        # if nan_ratio >= self.threshold:
        #     self.output[:, :, i, j] = float("nan")
        # else:
        #     # # Identify non-NaN values
        #     # nonnan_values = image_patch[~torch.isnan(image_patch)]

        #     # # Calculate mean and standard deviation of non-NaN values
        #     # mean = torch.mean(nonnan_values)
        #     # std = torch.std(nonnan_values)

        #     # # Generate random values to replace NaNs, drawn from a normal distribution
        #     # nan_mask = torch.isnan(image_patch)
        #     # random_values = torch.normal(mean, std, size=image_patch.shape)

        #     # # Replace NaNs in the original tensor with the generated random values
        #     # image_patch[nan_mask] = random_values[nan_mask]
        #     insert = image_patch[~torch.isnan(image_patch)].flatten()
        #     print('NaN Replacement', insert[0])

        #     # self.output[:, :, i, j] = torch.nansum(image_patch.unsqueeze(1) * kernel.unsqueeze(0), dim=(2, 3, 4))
        #     self.output[:, :, i, j] = F.conv2d(torch.nan_to_num(image_patch, nan=insert[0]), kernel, stride=1, padding=0).squeeze(dim=-1).squeeze(dim=-1)
        #     # self.output[:, :, i, j] = F.conv2d(torch.nan_to_num(image_patch), kernel, stride=1, padding=0).squeeze(dim=-1).squeeze(dim=-1)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NaNConv module.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Output tensor window after applying NaNConv operation.
        """

        # Pad the input image
        self.image_padded = torch.nn.functional.pad(image, (self.padding, self.padding, self.padding, self.padding))

        # Get dimensions
        self.batch_size, _, self.in_height, self.in_width = self.image_padded.shape

        self.x_img, self.y_img = image.shape[-2:]

        self.out_height = int(((self.x_img + 2 * self.padding - 1 * (self.kernel_height - 1)) - 1) / self.stride) + 1
        self.out_width = int(((self.y_img + 2 * self.padding - 1 * (self.kernel_width - 1)) - 1) / self.stride) + 1

        # Initialize output tensor
        self.output = torch.zeros(self.batch_size, self.out_channels, self.out_height, self.out_width)

        _ = [self.apply_threshold(self.kernel, i, j) for i in range(self.out_height) for j in range(self.out_width)]
        # self.output = torch.Tensor([self.apply_threshold(self.kernel, i, j) for i in range(self.out_height) for j in range(self.out_width)])

        if self.bias is not None:
            self.output += self.bias.view(1, -1, 1, 1)

        return self.output

class NormalConv2d(nn.Module):
    def __init__(
        self,
        train: bool = False,
        bias_presence: bool = True,
        padding: int = 0,
        stride: int = 1,
        threshold: float = 0.5,
        in_channels: int = None,
        out_channels: int = None,
        kernel_size: Tuple = None,
        kernel: torch.Tensor = None,
        bias: torch.Tensor = None,
    ):
        """
        Initializes the NaNConv2d layer.

        Args:
            train (bool): Whether to train the model.
            bias_presence (bool): Whether bias is present in the layer.
            padding (int): Padding size.
            stride (int): Stride size.
            threshold (float): Threshold for NaN ratio.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (tuple): Size of the convolutional kernel.
            kernel (torch.Tensor): Initial kernel weights.
            bias (torch.Tensor): Initial bias values.
        """

        super().__init__()
        # kernel = torch.flipud(torch.fliplr(kernel))
        self.stride = int(stride)
        self.padding = int(padding)
        self.threshold = threshold

        if train is False:
            self.inference(kernel, bias)
        else:
            self.trainer(in_channels, out_channels, kernel_size, bias_presence)

        # print(f'Kernel {self.kernel.shape} Stride {self.stride}, padding {self.padding}, bias {self.bias.shape}, in_channels {self.image_padded.shape}, out_channels {self.output.shape}')

    def inference(self, kernel: torch.Tensor, bias: torch.Tensor) -> None:
        """
        Initializes the NaNConv2d layer at inference with the provided kernel and bias from the original model.

        Args:
            kernel (torch.Tensor): 4D tensor representing the convolutional kernel.
            bias (torch.Tensor): 1D tensor representing the bias values.

        Returns:
            None
        """
        self.bias = bias
        self.kernel = kernel
        self.out_channels, _, self.kernel_height, self.kernel_width = kernel.shape
        # Force values to int
        self.out_channels = int(self.out_channels)
        self.kernel_height = int(self.kernel_height)
        self.kernel_width = int(self.kernel_width)

    # UNDER WORK
    def trainer(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], bias_presence: bool) -> None:
        """
        Initialize the NaNConv2d layer for training a model.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (tuple[int, int]): Size of the convolutional kernel (height, width).
            bias_presence (bool): Whether bias term should be included.

        Returns:
            None
        """

        self.out_channels = int(out_channels)
        self.kernel_height, self.kernel_width = kernel_size
        # Force values to int
        self.kernel_height = int(self.kernel_height)
        self.kernel_width = int(self.kernel_width)

        # Initialize kernel parameters
        self.kernel = nn.Parameter(
            init.xavier_normal_(torch.zeros((self.out_channels, in_channels, self.kernel_height, self.kernel_width)))
        )
        # Initialize bias parameters if bias is present
        if bias_presence:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.bias = None

    def get_window(self, i: int, j: int) -> torch.Tensor:
        """
        Get the receptive field window at position (i, j).

        Args:
            i (int): Horizontal index.
            j (int): Vertical index.

        Returns:
            torch.Tensor: Receptive field window.
        """
        x_start = i * self.stride
        y_start = j * self.stride

        # Calculate the ending point of the receptive field
        x_end = min(x_start + self.kernel_height, self.in_height + 2 * self.padding)
        y_end = min(y_start + self.kernel_width, self.in_width + 2 * self.padding)

        # Extract the image patch
        return self.image_padded[:, :, x_start:x_end, y_start:y_end]

    def apply_threshold(self, kernel: torch.Tensor, i: int, j: int) -> None:
        """
        Apply NaN thresholding to the output tensor window at position (i, j).

        Args:
            kernel (torch.Tensor): Convolution kernel.
            i (int): Horizontal index.
            j (int): Vertical index.
        """

        image_patch = self.get_window(i, j)
        # image_patch = choose_probability_large(image_patch, kernel, torch.min(kernel), torch.max(kernel))

        # self.output[:, :, i, j] = torch.sum(image_patch.unsqueeze(1) * kernel.unsqueeze(0), dim=(2, 3, 4))
        self.output[:, :, i, j] = F.conv2d(image_patch, kernel, stride=1, padding=0).squeeze(dim=-1).squeeze(dim=-1)


    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NaNConv module.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Output tensor window after applying NaNConv operation.
        """

        # Pad the input image
        self.image_padded = torch.nn.functional.pad(image, (self.padding, self.padding, self.padding, self.padding))

        # Get dimensions
        self.batch_size, _, self.in_height, self.in_width = self.image_padded.shape

        self.x_img, self.y_img = image.shape[-2:]

        self.out_height = int(((self.x_img + 2 * self.padding - 1 * (self.kernel_height - 1)) - 1) / self.stride) + 1
        self.out_width = int(((self.y_img + 2 * self.padding - 1 * (self.kernel_width - 1)) - 1) / self.stride) + 1

        # Initialize output tensor
        self.output = torch.zeros(self.batch_size, self.out_channels, self.out_height, self.out_width)

        _ = [self.apply_threshold(self.kernel, i, j) for i in range(self.out_height) for j in range(self.out_width)]
        # self.output = torch.Tensor([self.apply_threshold(self.kernel, i, j) for i in range(self.out_height) for j in range(self.out_width)])

        if self.bias is not None:
            self.output += self.bias.view(1, -1, 1, 1)

        return self.output


class NoArgMaxIndices(BaseException):
    def __init__(self):
        super(NoArgMaxIndices, self).__init__(
            "no argmax indices: batch_argmax requires non-batch shape to be non-empty")


class NaNPool2d:

    def __init__(self, max_threshold: int = 1, probabilistic: bool = False, rtol_epsilon: float = 1e-7, nan_probability: Optional[float] = 0.8):
        """
        Initializes the NaNPool2d object.

        Args:
            max_threshold (float, optional): Max threshold value for determining multiple max value occurrence ratio. Defaults to 0.5.
        """
        torch.manual_seed(0)
        self.max_threshold = max_threshold
        self.nan_probability = nan_probability
        self.probabilistic = probabilistic 
        self.rtol_epsilon = rtol_epsilon

    # CURRENTLY UNUSED
    ## unravel_index and batch_argmax are from https://stackoverflow.com/questions/39458193/using-list-tuple-etc-from-typing-vs-directly-referring-type-as-list-tuple-etc
    ## they generalize well to multi-dimensions in theory but are they necessary? Window is always 3D
    def unravel_index(self, 
        indices: torch.LongTensor,
        shape: Tuple[int, ...],
    ) -> torch.LongTensor:
        r"""Converts flat indices into unraveled coordinates in a target shape.

        This is a `torch` implementation of `numpy.unravel_index`.

        Args:
            indices: A tensor of (flat) indices, (*, N).
            shape: The targeted shape, (D,).

        Returns:
            The unraveled coordinates, (*, N, D).
        """

        coord = []

        for dim in reversed(shape):
            coord.append(indices % dim)
            indices = indices // dim

        coord = torch.stack(coord[::-1], dim=-1)

        return coord

    # CURRENTLY UNUSED
    def batch_argmax(self, tensor, batch_dim=1):
        """
        Assumes that dimensions of tensor up to batch_dim are "batch dimensions"
        and returns the indices of the max element of each "batch row".
        More precisely, returns tensor `a` such that, for each index v of tensor.shape[:batch_dim], a[v] is
        the indices of the max element of tensor[v].
        """
        if batch_dim >= len(tensor.shape):
            raise NoArgMaxIndices()
        batch_shape = tensor.shape[:batch_dim]
        non_batch_shape = tensor.shape[batch_dim:]
        flat_non_batch_size = prod(non_batch_shape)
        tensor_with_flat_non_batch_portion = tensor.reshape(*batch_shape, flat_non_batch_size)

        dimension_of_indices = len(non_batch_shape)

        # We now have each batch row flattened in the last dimension of tensor_with_flat_non_batch_portion,
        # so we can invoke its argmax(dim=-1) method. However, that method throws an exception if the tensor
        # is empty. We cover that case first.
        if tensor_with_flat_non_batch_portion.numel() == 0:
            # If empty, either the batch dimensions or the non-batch dimensions are empty
            batch_size = prod(batch_shape)
            if batch_size == 0:  # if batch dimensions are empty
                # return empty tensor of appropriate shape
                batch_of_unraveled_indices = torch.ones(*batch_shape, dimension_of_indices).long()  # 'ones' is irrelevant as it will be empty
            else:  # non-batch dimensions are empty, so argmax indices are undefined
                raise NoArgMaxIndices()
        else:   # We actually have elements to maximize, so we search for them
            indices_of_non_batch_portion = tensor_with_flat_non_batch_portion.argmax(dim=-1)
            batch_of_unraveled_indices = self.unravel_index(indices_of_non_batch_portion, non_batch_shape)

        if dimension_of_indices == 1:
            # above function makes each unraveled index of a n-D tensor a n-long tensor
            # however indices of 1D tensors are typically represented by scalars, so we squeeze them in this case.
            batch_of_unraveled_indices = batch_of_unraveled_indices.squeeze(dim=-1)
        return batch_of_unraveled_indices

    def check_for_nans(self, c, i, j, window, maxval, max_index):

        # Handle NaNs across the batch in a vectorized way
        nan_mask = torch.isnan(window)  # Find NaNs in maxval
        window = window.masked_fill(torch.isnan(window), float("-inf"))  # Replace NaNs in window with -inf
        maxval = torch.max(window.reshape(self.batch_size, -1), dim=1)[0]  # Recompute max after replacing NaNs

        # Update max_index in a vectorized way
        # We check if maxval was recomputed when NaNs are removed
        valid_indices = (window == maxval[:, None, None]) & (~nan_mask[:, None, None])  # Filter valid max indices
        if valid_indices.any():
            # max_index = torch.max(window.masked_fill(nan_mask[:, None, None], float('-inf')).reshape(self.batch_size, -1), dim=1)[1]
            max_index = torch.argmax(window.reshape(self.batch_size, -1), dim=1)
            max_index = torch.stack((max_index // self.pool_width, max_index % self.pool_width), dim=1)

            # # Ensure indices are within bounds by clamping
            # max_index[:, 0] = torch.clamp(max_index[:, 0], 0, self.pool_height - 1)
            # max_index[:, 1] = torch.clamp(max_index[:, 1], 0, self.pool_width - 1)


        # Check for multiple max values in a vectorized way
        check_multi_max = torch.sum(torch.isclose(window, maxval[:, None, None], rtol=self.rtol_epsilon, equal_nan=True), axis=(1, 2))

        # Apply the max threshold to the entire batch
        exceed_threshold = check_multi_max > self.max_threshold

        if exceed_threshold.any():
            if self.probabilistic:
                # Generate random mask with nan_probability
                random_mask = torch.rand(self.batch_size) < self.nan_probability
                # Apply both threshold and random mask
                maxval = torch.where(exceed_threshold & random_mask, torch.tensor(float('nan')), maxval)
            else:
                maxval = torch.where(exceed_threshold, torch.tensor(float('nan')), maxval)
                max_index = torch.zeros((self.batch_size, 2), dtype=torch.long)  # Default to index [0,0] for NaNs


        # if entire window is NaN and maxval=NaN then max_index is 1D instead of 2D
        if torch.isnan(maxval).all() or torch.isinf(maxval).all():
          max_index = torch.zeros((self.batch_size, 2), dtype=torch.long)  # Default to index [0,0] for NaNs


        # Compute 1D indices for output
        try:
            max_index_1d = (i * self.stride_height + max_index[:, 0]) * self.input_width + (j * self.stride_width + max_index[:, 1])
        except:
          print(maxval, maxval.shape, max_index, max_index.shape, i,j )

        # Assign the values and indices
        self.output_array[:, c, i, j] = maxval
        self.index_array[:, c, i, j] = max_index_1d


    def __call__(self, input_array: torch.Tensor, pool_size: Tuple, strides: Tuple = None) -> Tuple:
        """
        Perform NaN-aware max pooling on the input array.

        Args:
            input_array (torch.Tensor): Input tensor of shape (batch_size, channels, input_height, input_width).
            pool_size (tuple): Size of the pooling window (pool_height, pool_width).
            strides (tuple, optional): Strides for pooling (stride_height, stride_width). Defaults to None.

        Returns:
            tuple: A tuple containing output array and index array after pooling.
        """

        batch_size, channels, input_height, input_width = input_array.shape
        # Force values to int
        self.batch_size = int(batch_size)
        channels = int(channels)
        self.input_height = int(input_height)
        self.input_width = int(input_width)

        pool_height, pool_width = pool_size
        # Force values to int
        self.pool_height = int(pool_height)
        self.pool_width = int(pool_width)

        if strides:
            stride_height, stride_width = strides

        else:
            stride_height, stride_width = pool_size

        # Force values to int
        self.stride_height = int(stride_height)
        self.stride_width = int(stride_width)

        # Calculate simplified intensity distribution of the layer
        self.min_intensity = torch.min(input_array)
        self.max_intensity = torch.max(input_array)

        # Calculate the output dimensions
        output_height = int((input_height - pool_height) // stride_height + 1)
        output_width = int((input_width - pool_width) // stride_width + 1)

        # Initialize output arrays for pooled values and indices
        self.output_array = torch.zeros((self.batch_size, channels, output_height, output_width))
        self.index_array = torch.zeros((self.batch_size, channels, output_height, output_width), dtype=torch.int64)


        # Perform max pooling with list comprehensions
        for c in range(channels):

            # Create a list of tuples with pooled values and indices
            values_and_indices = [
                self.check_for_nans(c, i, j, window, torch.max(window.reshape(self.batch_size, -1), dim=1)[0], torch.max(window.reshape(self.batch_size, -1), dim=1)[1])
                for i in range(output_height)
                for j in range(output_width)
                for window in [
                    input_array[
                        :,
                        c,
                        i * stride_height : i * stride_height + pool_height,
                        j * stride_width : j * stride_width + pool_width,
                    ]
                ]
            ]

        return torch.Tensor(self.output_array), torch.Tensor(self.index_array).type(torch.int64)


class NormalPool2d:

    def __init__(self, max_threshold: float = 1):
        """
        Initializes the NaNPool2d object.

        Args:
            max_threshold (float, optional): Max threshold value for determining multiple max value occurrence ratio. Defaults to 0.5.
        """
        self.max_threshold = max_threshold

    def __call__(self, input_array: torch.Tensor, pool_size: Tuple, strides: Tuple = None) -> Tuple:
        """
        Perform NaN-aware max pooling on the input array.

        Args:
            input_array (torch.Tensor): Input tensor of shape (batch_size, channels, input_height, input_width).
            pool_size (tuple): Size of the pooling window (pool_height, pool_width).
            strides (tuple, optional): Strides for pooling (stride_height, stride_width). Defaults to None.

        Returns:
            tuple: A tuple containing output array and index array after pooling.
        """

        batch_size, channels, input_height, input_width = input_array.shape
        # Force values to int
        batch_size = int(batch_size)
        channels = int(channels)
        input_height = int(input_height)
        input_width = int(input_width)

        pool_height, pool_width = pool_size
        # Force values to int
        pool_height = int(pool_height)
        pool_width = int(pool_width)

        if strides:
            stride_height, stride_width = strides

        else:
            stride_height, stride_width = pool_size

        # Force values to int
        stride_height = int(stride_height)
        stride_width = int(stride_width)

        # Calculate simplified intensity distribution of the layer
        self.min_intensity = torch.min(input_array)
        self.max_intensity = torch.max(input_array)

        # Calculate the output dimensions
        output_height = int((input_height - pool_height) // stride_height + 1)
        output_width = int((input_width - pool_width) // stride_width + 1)

        # Initialize output arrays for pooled values and indices
        output_array = torch.zeros((batch_size, channels, output_height, output_width))
        index_array = torch.zeros((batch_size, channels, output_height, output_width), dtype=torch.int64)

        # Perform max pooling with list comprehensions
        for c in range(channels):

            # Create a list of tuples with pooled values and indices
            values_and_indices = [
                (torch.max(window.reshape(batch_size, -1), dim=1)[0], torch.argmax(window, dim=1))
                for i in range(output_height)
                for j in range(output_width)
                for window in [
                    input_array[
                        :,
                        c,
                        i * stride_height : i * stride_height + pool_height,
                        j * stride_width : j * stride_width + pool_width,
                    ]
                ]
            ]

            # Handle NaNs and probabilities for choosing max value
            for k, (maxval, max_index) in enumerate(values_and_indices):
                # Re-initialize the window within the second for-loop
                i = k // output_width
                j = k % output_width
                window = input_array[
                    :,
                    c,
                    i * stride_height : i * stride_height + pool_height,
                    j * stride_width : j * stride_width + pool_width,
                ]

                # if torch.isnan(maxval).any():
                #     window = window.masked_fill(torch.isnan(window), float("-inf"))
                #     maxval = torch.max(window).reshape(1)

                # # Strict approach to identifying multiple max values
                # # check_multi_max = torch.sum(window == maxval[:, None, None], axis=(1, 2))
                # # Less restrictive more theoretically stable approach
                # check_multi_max = torch.sum(
                #     torch.isclose(window, maxval[:, None, None], rtol=1e-7, equal_nan=True), axis=(1, 2)
                # )

                # # Reduce multiple max value counts to ratios in order to use passed max threshold value
                # # check_multi_max = check_multi_max / (window.shape[-1] * window.shape[-2])

                # if (check_multi_max > self.max_threshold).any():
                #     maxval = torch.where(check_multi_max > self.max_threshold, np.nan, maxval)

                # Calculate the indices for 1D representation
                max_index_1d = (i * stride_height + (max_index[:, 0] // pool_width)) * input_width + (
                    j * stride_width + (max_index[:, 1] % pool_width)
                )

                # Update output arrays
                output_array[:, c, i, j] = maxval
                index_array[:, c, i, j] = max_index_1d
            # break

        return torch.Tensor(output_array), torch.Tensor(index_array).type(torch.int64)


class NaNPool2d_v2:

    def __init__(self, max_threshold: int = 1, rtol_epsilon: float = 1e-7,):
        """
        Initializes the NaNPool2d object.

        Args:
            max_threshold (float, optional): Max threshold value for determining multiple max value occurrence ratio. Defaults to 0.5.
        """
        torch.manual_seed(0)
        self.max_threshold = max_threshold
        self.rtol_epsilon = rtol_epsilon


    def check_for_nans(self, c, i, j, window, maxval, max_index):

        # print(window)
        # Handle NaNs across the batch in a vectorized way
        nan_mask = torch.isnan(window)  # Find NaNs in maxval
        window = window.masked_fill(torch.isnan(window), float("-inf"))  # Replace NaNs in window with -inf
        maxval = torch.max(window.reshape(self.batch_size, -1), dim=1)[0]  # Recompute max after replacing NaNs

        # Update max_index in a vectorized way
        # We check if maxval was recomputed when NaNs are removed
        valid_indices = (window == maxval[:, None, None]) & (~nan_mask[:, None, None])  # Filter valid max indices
        if valid_indices.any():
            # max_index = torch.max(window.masked_fill(nan_mask[:, None, None], float('-inf')).reshape(self.batch_size, -1), dim=1)[1]
            max_index = torch.argmax(window.reshape(self.batch_size, -1), dim=1)
            max_index = torch.stack((max_index // self.pool_width, max_index % self.pool_width), dim=1)

            

        # Check for multiple max values in a vectorized way
        check_multi_max = torch.sum(torch.isclose(window, maxval[:, None, None], rtol=self.rtol_epsilon, equal_nan=True), axis=(1, 2))

        # Apply the max threshold to the entire batch
        exceed_threshold = check_multi_max > self.max_threshold
        # print('Max index shape and original index', max_index.shape, max_index)
        # print('original maxval', maxval)
        if exceed_threshold.any():
            
            # print('exceeded')
            # max_index = torch.zeros((self.batch_size, 2), dtype=torch.long)  # Default to index [0,0] for NaNs

            # Get the indices where the value equals the maximum value
            # We directly compare the window to the max values
            # matching_indices = (window == maxval.view(-1, 1, 1))  # Broadcasting maxval across the window
            matching_indices = torch.isclose(window, maxval.view(-1, 1, 1), rtol=self.rtol_epsilon, equal_nan=True )  # Broadcasting maxval across the window

            # Convert the boolean mask to indices
            indices = matching_indices.nonzero(as_tuple=False)

            # Group the indices by batch
            max_index = [indices[indices[:, 0] == batch, 1:] for batch in indices[:, 0].unique()]

        # else:
        #     print('here')
        #     max_index = max_index.unsqueeze(0)
            
        # print('new shape', max_index.shape)
        # print('maxval and new index', maxval, max_index)
       
        max_index_1d = []
        # print(type(max_index), max_index[:, 0])
        # Compute 1D indices for output
        if type(max_index) == list:
            # print(max_index)
            for batch_idx in max_index:
                tmp = (i * self.stride_height + batch_idx[:, 0]) * self.input_width + (j * self.stride_width + batch_idx[:, 1])
                if len(tmp) > 1:
                    max_index_1d.append( tmp )
                else:
                    max_index_1d.append( tmp[0] )
                
            # print(max_index_1d)
        else:
            max_index_1d.append( (i * self.stride_height + max_index[:, 0]) * self.input_width + (j * self.stride_width + max_index[:, 1]) )

        # print('max_index_1d', max_index_1d)

        # Assign the values and indices
        self.output_array[:, c, i, j] = maxval
        # print(max_index_1d)
        # self.index_array[:, c, i, j, :, :] = torch.stack(max_index_1d)
        self.index_array[(c, i, j)] = max_index_1d



    def __call__(self, input_array: torch.Tensor, pool_size, strides = None, padding=None) -> Tuple:
        """
        Perform NaN-aware max pooling on the input array.

        Args:
            input_array (torch.Tensor): Input tensor of shape (batch_size, channels, input_height, input_width).
            pool_size (int or tuple): Size of the pooling window (pool_height, pool_width).
            strides (int or tuple, optional): Strides for pooling (stride_height, stride_width). Defaults to None.

        Returns:
            tuple: A tuple containing output array and index array after pooling.
        """

        batch_size, channels, input_height, input_width = input_array.shape
        # Force values to int
        self.batch_size = int(batch_size)
        channels = int(channels)
        self.input_height = int(input_height)
        self.input_width = int(input_width)

        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding) #IMPLEMENT PROPERLY

        # pool_height, pool_width = self.pool_size
        # Force values to int
        self.pool_height = int(self.pool_size[0])
        self.pool_width = int(self.pool_size[1])

        if self.strides:
            stride_height, stride_width = self.strides

        else:
            stride_height, stride_width = self.pool_size

        # Force values to int
        self.stride_height = int(stride_height)
        self.stride_width = int(stride_width)

        # Calculate simplified intensity distribution of the layer
        self.min_intensity = torch.min(input_array)
        self.max_intensity = torch.max(input_array)

        # Calculate the output dimensions
        output_height = int((input_height - self.pool_height) // stride_height + 1)
        output_width = int((input_width - self.pool_width) // stride_width + 1)

        # Initialize output arrays for pooled values and indices
        self.output_array = torch.zeros((self.batch_size, channels, output_height, output_width))
        # self.index_array = torch.zeros((self.batch_size, channels, output_height, output_width), dtype=torch.object)
        self.index_array = torch.zeros((self.batch_size, channels, output_height, output_width, self.pool_height, self.pool_width),)
        self.index_array = {}


        # Perform max pooling with list comprehensions
        for c in range(channels):

            # Create a list of tuples with pooled values and indices
            values_and_indices = [
                self.check_for_nans(c, i, j, window, torch.max(window.reshape(self.batch_size, -1), dim=1)[0], torch.max(window.reshape(self.batch_size, -1), dim=1)[1])
                for i in range(output_height)
                for j in range(output_width)
                for window in [
                    input_array[
                        :,
                        c,
                        i * stride_height : i * stride_height + self.pool_height,
                        j * stride_width : j * stride_width + self.pool_width,
                    ]
                ]
            ]

        return torch.Tensor(self.output_array), self.index_array #torch.Tensor(self.index_array).type(torch.int64)


class NaNUnpool2d:
    def __init__(self, kernel_size, stride, padding, output_size=None):
        """
        Initializes NaN unpooling for a 4D tensor using 1D flattened indices.

        Args:
            kernel_size (int or tuple): The kernel size used in pooling.
            stride (int or tuple): The stride used in pooling.
            padding (int or tuple): The padding used in pooling.
            output_size (tuple, optional): The output size (H_out, W_out). If None, it will be calculated.
        """
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_size = output_size  # Optional argument for output size

    def _convert_1d_to_2d(self, flat_idx, width, stride_height, stride_width, padding, output_height, output_width):
        """
        Converts a 1D index to 2D coordinates, ensuring the positions are within bounds.

        Args:
            flat_idx (int): The 1D index to convert.
            width (int): The width of the input tensor.
            stride_height (int): The stride in height.
            stride_width (int): The stride in width.
            padding (int): The padding.
            output_height (int): The height of the output tensor.
            output_width (int): The width of the output tensor.

        Returns:
            tuple: The (row, col) 2D coordinates.
        """
        out_row = (flat_idx // width) * stride_height - padding
        out_col = (flat_idx % width) * stride_width - padding

        if 0 <= out_row < output_height and 0 <= out_col < output_width:
            return out_row, out_col
        return None, None  # Invalid position

    def __call__(self, input_tensor, indices, ):
        """
        Implements NaN unpooling for a 4D tensor using 1D flattened indices.

        Args:
            input_tensor (torch.Tensor): The input tensor of shape (B, C, H, W) (from max pooling).
            indices (torch.Tensor): The indices of max pooling of shape (B, C, H, W).

        Returns:
            torch.Tensor: The unpooled tensor of shape (B, C, H_out, W_out).
        """
        # Extract input dimensions
        batch_size, channels, height, width = input_tensor.size()

        # Calculate output dimensions if output_size is not provided
        if self.output_size is None:
            output_height = (height - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0]
            output_width = (width - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1]
            self.output_size = (batch_size, channels, output_height, output_width)
       
        # Initialize the unpooled tensor with zeros
        output = torch.zeros(self.output_size, dtype=input_tensor.dtype, device=input_tensor.device)
        insert_nan = False

        # Use PyTorch's scatter_add to assign values based on the indices
        for b in range(batch_size):
            for c in range(channels):
            # for c, i in zip(np.arange(channels), indices.keys()):

                # modified indices for nan unpooling are in the form of a dictionary
                if type(indices) == dict:

                    try: 
                        #attempt to process normal indices that do not have multiple max values 
                        flat_indices = torch.stack([item[0] for key, item in indices.items() if key[:len((c,))] == (c,)])[:, b] #.view(-1)
                        flat_input =  input_tensor[b, c].view(-1)
                    except:
                        # we go here when a ragged array is detected above aka presence of multiple max values
                        # print( list({key: value for key, value in indices.items() if key[:len((c,))] == (c,)}.values()) )

                        # sort through the multiple max values indices vs the normal max values indices to extract the appropriate 
                        # indices for the present coordinates
                        tmp_indices = []
                        for key,value in indices.items():
                            if key[:len((c,))] == (c,):
                                total_elements = sum(t.numel() for t in value)

                                if total_elements > batch_size:
                                    tmp_indices.append(value[b])
                                else: 
                                    tmp_indices.append(value[0][b])

                        flat_indices = [tmp_indices[i:i + batch_size] for i in range(0, len(tmp_indices), batch_size)]

                        # Calculate padding size
                        total_elements = input_tensor[b, c].numel()  # Total elements = 81
                        remainder = total_elements % batch_size
                        if remainder != 0: # total elements is an odd number which may be a problem if batch size is even or vice versa
                            padding_size = batch_size - remainder
                            flat_input = F.pad(input_tensor[b, c].view(-1), (0, padding_size), value=float('inf')).view(-1, batch_size)
                        else: #assume window is even numbered
                            flat_input =  input_tensor[b, c].view(-1, batch_size)

                        # Turn flag on to ensure NaNs are inserted later
                        insert_nan = True
                        # second_dim = flat_input.shape[1]
                   
                    # print('final',  flat_indices )

                else: # if its normal type of indices array
                    flat_indices = indices[b, c].view(-1)
                    flat_input =  input_tensor[b, c].view(-1)

                flat_output = output[b, c].view(-1)
                # flat_indices = indices[b, c].view(-1)
                # flat_input = input_tensor[b, c].view(-1)


                # print(flat_indices, flat_input)

                if insert_nan:
                    # for idx_batch, inp in zip(flat_indices, flat_input[:, :second_dim]):
                    for idx_batch, inp in zip(flat_indices, flat_input):
                        inp = inp[~torch.isinf(inp)] # stripping away inf potentially added in earlier for padding
                        for flat_idx, flat_inp in zip(idx_batch, inp):
                            # print(flat_idx, flat_idx.shape )
                            if flat_idx.shape:
                                # print(flat_idx, 'nan')
                                flat_output.scatter_(0, flat_idx, float('nan'))
                            else:
                                # print(flat_idx, flat_inp)
                                flat_output.scatter_(0, flat_idx, flat_inp)

                    # Reset flag to prevent unwanted NaN insertion
                    insert_nan = False

                else:     
                    # Normal unpooling
                    flat_output.scatter_(0, flat_indices, flat_input)

        return output


class NormalUnpool2d:
    def __init__(self, kernel_size, stride, padding, output_size=None):
        """
        Initializes NaN unpooling for a 4D tensor using 1D flattened indices.

        Args:
            kernel_size (int or tuple): The kernel size used in pooling.
            stride (int or tuple): The stride used in pooling.
            padding (int or tuple): The padding used in pooling.
            output_size (tuple, optional): The output size (H_out, W_out). If None, it will be calculated.
        """
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_size = output_size  # Optional argument for output size

    def _convert_1d_to_2d(self, flat_idx, width, stride_height, stride_width, padding, output_height, output_width):
        """
        Converts a 1D index to 2D coordinates, ensuring the positions are within bounds.

        Args:
            flat_idx (int): The 1D index to convert.
            width (int): The width of the input tensor.
            stride_height (int): The stride in height.
            stride_width (int): The stride in width.
            padding (int): The padding.
            output_height (int): The height of the output tensor.
            output_width (int): The width of the output tensor.

        Returns:
            tuple: The (row, col) 2D coordinates.
        """
        out_row = (flat_idx // width) * stride_height - padding
        out_col = (flat_idx % width) * stride_width - padding

        if 0 <= out_row < output_height and 0 <= out_col < output_width:
            return out_row, out_col
        return None, None  # Invalid position

    def __call__(self, input_tensor, indices):
        """
        Implements NaN unpooling for a 4D tensor using 1D flattened indices.

        Args:
            input_tensor (torch.Tensor): The input tensor of shape (B, C, H, W) (from max pooling).
            indices (torch.Tensor): The indices of max pooling of shape (B, C, H, W).

        Returns:
            torch.Tensor: The unpooled tensor of shape (B, C, H_out, W_out).
        """
        # Extract input dimensions
        batch_size, channels, height, width = input_tensor.size()

        # Calculate output dimensions if output_size is not provided
        if self.output_size is None:
            output_height = (height - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0]
            output_width = (width - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1]
            self.output_size = (batch_size, channels, output_height, output_width)
        # else:
        #     output_size = (*self.output_size)  # Use the provided output size

        # Initialize the unpooled tensor with zeros
        output = torch.zeros(self.output_size, dtype=input_tensor.dtype, device=input_tensor.device)
        # print('output', output.shape)

        # Use PyTorch's scatter_add to assign values based on the indices
        for b in range(batch_size):
            for c in range(channels):
                flat_output = output[b, c].view(-1)
                flat_indices = indices[b, c].view(-1)
                flat_input = input_tensor[b, c].view(-1)

                print(flat_indices, flat_input)

                flat_output.scatter_(0, flat_indices, flat_input)

        return output


# Conv2d Skipped Operation Counter -- not to be implemented in CPP
def count_skip_conv2d(image, kernel, padding=0, stride=1, threshold=0.5):

    # kernel = torch.flipud(torch.fliplr(kernel))

    # Pad the input image
    image_padded = torch.nn.functional.pad(image, (padding, padding, padding, padding))

    # Get dimensions
    batch_size, in_channels, in_height, in_width = image_padded.shape
    out_channels, _, kernel_height, kernel_width = kernel.shape

    x_img, y_img = image.shape[-2:]

    out_height = int((( x_img + 2*padding - 1*(kernel_height - 1) ) -1)/stride ) + 1
    out_width = int((( y_img + 2*padding - 1*(kernel_width - 1) ) -1)/stride ) + 1

    # print(out_height, out_width)
    skip = 0
    total=0

    # Perform convolution
    for b in range(batch_size):
        for i in range(0, out_height):

            for j in range(0, out_width):

                #PUT IN FUNCTION FOR LIST COMPREHENSION
                x_start = i * stride
                y_start = j * stride
                # Calculate the ending point of the receptive field
                x_end = min(x_start + kernel_height, in_height + 2 * padding)
                y_end = min(y_start + kernel_width, in_width + 2 * padding)

                # Extract the image patch
                image_patch = image_padded[b, :, x_start:x_end, y_start:y_end]
                # print(image_patch.shape)
                
                try:
                    # print(torch.sum(torch.isnan(image_patch)).item(), torch.sum(~torch.isnan(image_patch)).item())
                    # nan_ratio = torch.sum(torch.isnan(image_patch)).item() / torch.sum(~torch.isnan(image_patch)).item() 
                    # nan_ratio = torch.sum(torch.isnan(image_patch)).item() / image_patch.numel() 
                    # print( image_patch.size(1) * image_patch.size(2) )
                    nan_ratio = torch.isnan(image_patch).sum(dim=(0, 1, 2)) / (image_patch.size(0) * image_patch.size(1) * image_patch.size(2) )

                except ZeroDivisionError:
                    nan_ratio = 1
                
                if nan_ratio >= threshold: 
                    skip += 1
                total+=1

                
        
    return skip, total
