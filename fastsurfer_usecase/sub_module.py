# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Tuple

# IMPORTS
import torch
print(torch.__version__)
from torch import Tensor, nn
import pickle
import os
import random
import numpy as np
import torch.nn.init as init
import time
# import torch._dynamo as dynamo

#for nan pooling
# from nan_ops import NaNConv2d, NaNPool2d, count_skip_conv2d
#for nan unpooling
from nan_ops import NaNConv2d, NaNUnpool2d, count_skip_conv2d
from nan_ops import NaNPool2d_v2 as NaNPool2d


# Building Blocks
class InputDenseBlock(nn.Module):
    """
    Input Dense Block.

    Attributes
    ----------
    conv[0-3]
        Convolution layers.
    bn0
        Batch Normalization.
    gn[1-4]
        Batch Normalizations.
    prelu
        Learnable ReLU Parameter.

    Methods
    -------
    forward
        Feedforward through graph.
    """

    def __init__(self, params: Dict):
        """
        Construct InputDenseBlock object.

        Parameters
        ----------
        params : Dict
            Parameters in dictionary format.
        """
        super(InputDenseBlock, self).__init__()

        self.params = params

        # Padding to get output tensor of same dimensions
        padding_h = int((params["kernel_h"] - 1) / 2)
        padding_w = int((params["kernel_w"] - 1) / 2)

        # Sub-layer output sizes for BN; and
        conv0_in_size = params["num_channels"]
        conv1_in_size = params["num_filters_interpol"]
        conv2_in_size = params["num_filters_interpol"]
        out_size = (
            params["num_filters_interpol_last"]
            if "num_filters_interpol_last" in params
            else params["num_filters_interpol"]
        )
        self.conv0_in_size = conv0_in_size
        self.conv1_in_size = conv1_in_size
        self.conv2_in_size = conv2_in_size
        self.out_size = out_size


        # learnable layers
        self.conv0 = nn.Conv2d(
            in_channels=conv0_in_size,
            out_channels=params["num_filters_interpol"],
            kernel_size=(params["kernel_h"], params["kernel_w"]),
            stride=params["stride_conv"],
            padding=(padding_h, padding_w),
        )

        print(f'INPUT Conv0:\nOut channels: {params["num_filters_interpol"]}\nIn_channels: {conv0_in_size}\nKernel size: {params["kernel_h"], params["kernel_w"]}\nStride: {params["stride_conv"]}\nPadding: {padding_h, padding_w}')


        self.conv1 = nn.Conv2d(
            in_channels=conv1_in_size,
            out_channels=params["num_filters_interpol"],
            kernel_size=(params["kernel_h"], params["kernel_w"]),
            stride=params["stride_conv"],
            padding=(padding_h, padding_w),
        )

        print(f'INPUT Conv1:\nOut channels: {params["num_filters_interpol"]}\nIn_channels: {conv0_in_size}\nKernel size: {params["kernel_h"], params["kernel_w"]}\nStride: {params["stride_conv"]}\nPadding: {padding_h, padding_w}')

        self.conv2 = nn.Conv2d(
            in_channels=conv1_in_size,
            out_channels=params["num_filters_interpol"],
            kernel_size=(params["kernel_h"], params["kernel_w"]),
            stride=params["stride_conv"],
            padding=(padding_h, padding_w),
        )
        print(f'INPUT Conv2:\nOut channels: {params["num_filters_interpol"]}\nIn_channels: {conv0_in_size}\nKernel size: {params["kernel_h"], params["kernel_w"]}\nStride: {params["stride_conv"]}\nPadding: {padding_h, padding_w}')

        # D \times D convolution for the last block --> with maxout this is redundant unless we want to reduce
        # the number of filter maps here compared to conv1
        self.conv3 = nn.Conv2d(
            in_channels=conv2_in_size,
            out_channels=out_size,
            kernel_size=(params["kernel_h"], params["kernel_w"]),
            stride=params["stride_conv"],
            padding=(padding_h, padding_w),
        )
        print(f'INPUT Conv3:\nOut channels: {params["num_filters_interpol"]}\nIn_channels: {conv0_in_size}\nKernel size: {params["kernel_h"], params["kernel_w"]}\nStride: {params["stride_conv"]}\nPadding: {padding_h, padding_w}')

        self.bn0 = nn.BatchNorm2d(params["num_channels"])
        print('INPUT', self.bn0.weight, self.bn0.bias)
        self.gn1 = nn.BatchNorm2d(conv1_in_size)
        self.gn2 = nn.BatchNorm2d(conv2_in_size)
        self.gn3 = nn.BatchNorm2d(conv2_in_size)
        self.gn4 = nn.BatchNorm2d(out_size)

        self.prelu = nn.PReLU()  # Learnable ReLU Parameter



    def forward(self, x: Tensor) -> Tensor:
        """
        Feedforward through graph.

        Parameters
        ----------
        x : Tensor
            Input image [N, C, H, W] representing the input data.

        Returns
        -------
        out : Tensor
            [MISSING].
        """
        # if torch.isnan(x).any():
        #     bn = ManualBatchNorm2d(num_features=self.conv0_in_size)

        # Input batch normalization
        x0_bn = self.bn0(x)
        # pickle.dump(x0_bn, open(f"/output/inp_block/{os.environ['NANCONV_THRESHOLD']}_outp_bn0_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))


        # Convolution block1 (RF: 3x3)
        start_time = time.time()
        if os.environ['NAN_ACTIVE'] == "true":
            conv0 = NaNConv2d(train=False, kernel=self.conv0.weight, bias=self.conv0.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
            print(self.conv0.stride, self.conv0.padding)
            print(self.params["stride_conv"], int((self.params["kernel_h"] - 1) / 2))
            x0 = conv0(x0_bn)
        else:
            x0 = self.conv0(x0_bn)
                
        print(f'Took: {time.time()-start_time} seconds')
        # conv0 = ConvCustom(x0_bn, self.conv0.weight, self.conv0.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
        # x0 = conv0.forward()
        print('SKIP count', count_skip_conv2d(x0_bn, self.conv0.weight, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD'])))
        # pickle.dump(x0, open(f"/output/inp_block/{os.environ['NANCONV_THRESHOLD']}_outp_conv0_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))


        x1_gn = self.gn1(x0)
        # pickle.dump(x1_gn, open(f"/output/inp_block/{os.environ['NANCONV_THRESHOLD']}_outp_bn1_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))


        x1 = self.prelu(x1_gn)
        # pickle.dump(x1, open(f"/output/inp_block/{os.environ['NANCONV_THRESHOLD']}_outp_prelu0_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        # Convolution block2 (RF: 5x5)
        start_time = time.time()
        if os.environ['NAN_ACTIVE'] == "true":
            conv1 = NaNConv2d(train=False, kernel=self.conv1.weight, bias=self.conv1.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
            print(self.conv1.stride, self.conv1.padding)
            print(self.params["stride_conv"], int((self.params["kernel_h"] - 1) / 2))        
            x1 = conv1(x1)
        else:
            x1 = self.conv1(x1)
        
        print(f'Took: {time.time()-start_time} seconds')
        # conv1 = ConvCustom(x1, self.conv1.weight, self.conv1.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
        # x1 = conv1.forward()
        print('SKIP count', count_skip_conv2d(x1, self.conv1.weight, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD'])))
        # pickle.dump(x1, open(f"/output/inp_block/{os.environ['NANCONV_THRESHOLD']}_outp_conv1_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        x2_gn = self.gn2(x1)
        # pickle.dump(x2_gn, open(f"/output/inp_block/{os.environ['NANCONV_THRESHOLD']}_outp_bn2_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))



        # First Maxout
        x2_max = torch.maximum(x2_gn, x1_gn)
        # pickle.dump(x2_max, open(f"/output/inp_block/{os.environ['NANCONV_THRESHOLD']}_outp_maxout0_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        x2 = self.prelu(x2_max)
        # pickle.dump(x2, open(f"/output/inp_block/{os.environ['NANCONV_THRESHOLD']}_outp_prelu1_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        # Convolution block 3 (RF: 7x7)
        start_time = time.time()
        if os.environ['NAN_ACTIVE'] == "true":
            conv2 = NaNConv2d(train=False, kernel=self.conv2.weight, bias=self.conv2.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
            print(self.conv2.stride, self.conv2.padding)
            print(self.params["stride_conv"], int((self.params["kernel_h"] - 1) / 2))
            x2 = conv2(x2)
        else:
            x2 = self.conv2(x2)
        
        print(f'Took: {time.time()-start_time} seconds')
        # conv2 = ConvCustom(x2, self.conv2.weight, self.conv2.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
        # x2 = conv2.forward()
        print('SKIP count', count_skip_conv2d(x2, self.conv2.weight, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD'])))
        # pickle.dump(x2, open(f"/output/inp_block/{os.environ['NANCONV_THRESHOLD']}_outp_conv2_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        x3_gn = self.gn3(x2)
        # pickle.dump(x3_gn, open(f"/output/inp_block/{os.environ['NANCONV_THRESHOLD']}_outp_bn3_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))



        # Second Maxout

        x3_max = torch.maximum(x3_gn, x2_max)
        # pickle.dump(x3_max, open(f"/output/inp_block/{os.environ['NANCONV_THRESHOLD']}_outp_maxout1_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        x3 = self.prelu(x3_max)
        # pickle.dump(x3, open(f"/output/inp_block/{os.environ['NANCONV_THRESHOLD']}_outp_prelu2_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        # Convolution block 4 (RF: 9x9)
        start_time = time.time()
        if os.environ['NAN_ACTIVE'] == "true":
            conv3 = NaNConv2d(train=False, kernel=self.conv3.weight, bias=self.conv3.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
            print(self.conv3.stride, self.conv3.padding)
            print(self.params["stride_conv"], int((self.params["kernel_h"] - 1) / 2))
            x3 = conv3(x3)
        else:
            x3 = self.conv3(x3)

        print(f'Took: {time.time()-start_time} seconds')
        # conv3 = ConvCustom(x3, self.conv3.weight, self.conv3.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
        # x3 = conv3.forward()
        print('SKIP count', count_skip_conv2d(x3, self.conv3.weight, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD'])))
        # pickle.dump(x3, open(f"/output/inp_block/{os.environ['NANCONV_THRESHOLD']}_outp_conv3_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))


        out = self.gn4(x3)
        # pickle.dump(out, open(f"/output/inp_block/{os.environ['NANCONV_THRESHOLD']}_outp_bn4_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))



        return out


# def count_skip_conv2d(image, kernel, padding=0, stride=1, threshold=0.5):

#     # kernel = torch.flipud(torch.fliplr(kernel))

#     # Pad the input image
#     image_padded = torch.nn.functional.pad(image, (padding, padding, padding, padding))

#     # Get dimensions
#     batch_size, in_channels, in_height, in_width = image_padded.shape
#     out_channels, _, kernel_height, kernel_width = kernel.shape

#     x_img, y_img = image.shape[-2:]

#     out_height = int((( x_img + 2*padding - 1*(kernel_height - 1) ) -1)/stride ) + 1
#     out_width = int((( y_img + 2*padding - 1*(kernel_width - 1) ) -1)/stride ) + 1

#     print(out_height, out_width)
#     skip = 0
#     total=0

#     # Perform convolution
#     for c in range(in_channels):
#         for i in range(0, out_height):

#             for j in range(0, out_width):

#                 #PUT IN FUNCTION FOR LIST COMPREHENSION
#                 x_start = i * stride
#                 y_start = j * stride
#                 # Calculate the ending point of the receptive field
#                 x_end = min(x_start + kernel_height, in_height + 2 * padding)
#                 y_end = min(y_start + kernel_width, in_width + 2 * padding)

#                 # Extract the image patch
#                 image_patch = image_padded[:, c, x_start:x_end, y_start:y_end]
                
#                 try:
#                     # print(torch.sum(torch.isnan(image_patch)).item(), torch.sum(~torch.isnan(image_patch)).item())
#                     # nan_ratio = torch.sum(torch.isnan(image_patch)).item() / torch.sum(~torch.isnan(image_patch)).item() 
#                     nan_ratio = torch.sum(torch.isnan(image_patch)).item() / image_patch.numel() 
#                 except ZeroDivisionError:
#                     nan_ratio = 1
                
#                 if nan_ratio >= threshold: 
#                     skip += 1
#                 total+=1
                
        
#     return skip, total

# # def choose_probability_large(input, kernel, min_val, max_val):
# #     # If all NaN, return input as is
# #     if torch.isnan(input).all(): return input
# #     elif not torch.isnan(input).any(): return input


# #     # Flatten input and kernel
# #     input_flat = input.ravel() #view(-1)
# #     kernel_flat = kernel.ravel() #view(-1)
# #     # print(input_flat)

# #     # Calculate probabilities for NaN values
# #     distance_to_max = max_val - kernel_flat
# #     total_distance = max_val - min_val
# #     probabilities = 1 - distance_to_max / total_distance

# #     # Create a mask for NaN values
# #     nan_mask = torch.isnan(input_flat)

# #     # Indices of NaN values
# #     nan_indices = torch.nonzero(nan_mask).squeeze()
# #     # print('NAN INDICES', nan_indices)
# #     # print(nan_indices.numel())

# #     # Generate random values from the distribution
# #     # print('NO NAN INPUT', input_flat[~nan_mask])
# #     hist, bin_edges = torch.histogram(input_flat[~nan_mask], bins=5, density=True)
# #     # print('HIST', hist, bin_edges)
# #     random_values = torch.tensor(random.choices(bin_edges[:-1], weights=(hist / torch.sum(hist)), k=nan_indices.numel()))

# #     # Replace NaN values with random values
# #     # input_flat[nan_indices] = random_values

# #     # print(probabilities.shape, kernel.shape)
# #     probabilities = torch.mean(probabilities.view(kernel.shape), axis=0).ravel()
# #     # print(probabilities.shape)
# #     result = torch.where(
# #         torch.isnan(input_flat) & (probabilities > random.random()),
# #         random_values[torch.randint(len(random_values), size=input_flat.shape)],
# #         input_flat
# #     )
# #     # print(input_flat)
# #     # print(result)
# #     # print()

# #     return result.view(input.shape)
# # class ConvCustom:
# #     def __init__(self, image, kernel, bias=None, padding=0, stride=1, threshold=0.5):

# #         # kernel = torch.flipud(torch.fliplr(kernel))

# #         self.stride = stride
# #         self.padding = padding
# #         self.bias = bias
# #         self.threshold = threshold
# #         self.kernel = kernel

# #         # Pad the input image
# #         self.image_padded = torch.nn.functional.pad(image, (padding, padding, padding, padding))

# #         # Get dimensions
# #         self.batch_size, self.in_channels, self.in_height, self.in_width = self.image_padded.shape
# #         self.out_channels, _, self.kernel_height, self.kernel_width = kernel.shape

# #         self.x_img, self.y_img = image.shape[-2:]

# #         self.out_height = int((( self.x_img + 2*padding - 1*(self.kernel_height - 1) ) -1)/stride ) + 1
# #         self.out_width = int((( self.y_img + 2*padding - 1*(self.kernel_width - 1) ) -1)/stride ) + 1


# #         # Initialize output tensor
# #         self.output = torch.zeros(self.batch_size, self.out_channels, self.out_height, self.out_width)

# #         # print(f'Kernel {self.kernel.shape} Stride {self.stride}, padding {self.padding}, bias {self.bias.shape}, in_channels {self.image_padded.shape}, out_channels {self.output.shape}')
        
# #     def choose_probability_large(self, input, kernel, min_val, max_val):
# #         # If all NaN, return input as is
# #         if torch.isnan(input).all(): return input
# #         elif not torch.isnan(input).any(): return input


# #         # Flatten input and kernel
# #         input_flat = input.ravel() #view(-1)
# #         kernel_flat = kernel.ravel() #view(-1)

# #         # Calculate probabilities for NaN values
# #         distance_to_max = max_val - kernel_flat
# #         total_distance = max_val - min_val
# #         probabilities = 1 - distance_to_max / total_distance

# #         # Create a mask for NaN values
# #         nan_mask = torch.isnan(input_flat)

# #         # Indices of NaN values
# #         nan_indices = torch.nonzero(nan_mask).squeeze()

# #         # Generate random values from the distribution
# #         hist, bin_edges = torch.histogram(input_flat[~nan_mask], bins=5, density=True)
# #         random_values = torch.tensor(random.choices(bin_edges[:-1], weights=(hist / torch.sum(hist)), k=nan_indices.numel()))

# #         # Replace NaN values with random values
# #         # input_flat[nan_indices] = random_values

# #         probabilities = torch.mean(probabilities.view(kernel.shape), axis=0).ravel()

# #         result = torch.where(
# #             torch.isnan(input_flat) & (probabilities > random.random()),
# #             random_values[torch.randint(len(random_values), size=input_flat.shape)],
# #             input_flat
# #         )

# #         return result.view(input.shape)

# #     def get_window(self, i, j):
# #         x_start = i * self.stride
# #         y_start = j * self.stride
# #         # Calculate the ending point of the receptive field
# #         x_end = min(x_start + self.kernel_height, self.in_height + 2 * self.padding)
# #         y_end = min(y_start + self.kernel_width, self.in_width + 2 * self.padding)

# #         # Extract the image patch
# #         return self.image_padded[:, :, x_start:x_end, y_start:y_end]

    
# #     def apply_threshold(self, kernel, i, j):

# #         image_patch = self.get_window(i, j)
# #         # image_patch = choose_probability_large(image_patch, kernel, torch.min(kernel), torch.max(kernel))


# #         try:
# #             # print(torch.sum(torch.isnan(image_patch)).item(), torch.sum(~torch.isnan(image_patch)).item())
# #             # nan_ratio = torch.sum(torch.isnan(image_patch)).item() / torch.sum(~torch.isnan(image_patch)).item() 
# #             nan_ratio = torch.sum(torch.isnan(image_patch)).item() / image_patch.numel()
# #         except ZeroDivisionError:
# #             nan_ratio = 1
        
# #         if nan_ratio >= self.threshold: #change to and equal
# #             #CHANGE TO SET TO NAN
# #             self.output[:, :, i, j] = float('nan') #torch.full(self.output[:, :, i, j].shape, float('nan')) #torch.sum(image_patch.unsqueeze(1) * kernel.unsqueeze(0), dim=(2, 3, 4))
# #             # self.output[:, :, i, j] = torch.sum(image_patch.unsqueeze(1) * kernel.unsqueeze(0), dim=(2, 3, 4))
# #         else:
# #             self.output[:, :, i, j] = torch.nansum(image_patch.unsqueeze(1) * kernel.unsqueeze(0), dim=(2, 3, 4))
            

# #     def forward(self):

# #         _ = [self.apply_threshold(self.kernel, i, j) for i in range(self.out_height) for j in range(self.out_width)]
# #         # print(type(a) , (a))
# #         # self.output = torch.Tensor([self.apply_threshold(self.kernel, i, j) for i in range(self.out_height) for j in range(self.out_width)])


# #         if self.bias is not None:
# #             self.output += self.bias.view(1, -1, 1, 1)

# #         return self.output


# class NaNConv2d(nn.Module):
#     # def __init__(self, image, kernel, bias=None, padding=0, stride=1, threshold=0.5, train=False):
#     def __init__(self, train=False, bias_presence=True, padding=0, stride=1, threshold=0.5, in_channels=None, out_channels=None, kernel_size=None, kernel=None, bias=None):
#         super().__init__()
#         # kernel = torch.flipud(torch.fliplr(kernel))
#         self.stride = stride
#         self.padding = padding
#         self.threshold = threshold
        
#         if train is False:
#             self.inference(kernel, bias)
#         else:
#             self.trainer(in_channels, out_channels, kernel_size, bias_presence)
         
#         # print(f'Kernel {self.kernel.shape} Stride {self.stride}, padding {self.padding}, bias {self.bias.shape}, in_channels {self.image_padded.shape}, out_channels {self.output.shape}')

#     def inference(self, kernel, bias):
#         self.bias = bias
#         self.kernel = kernel
#         self.out_channels, _, self.kernel_height, self.kernel_width = kernel.shape

#     def trainer(self, in_channels, out_channels, kernel_size, bias_presence):
#         self.out_channels = out_channels
#         self.kernel_height, self.kernel_width = kernel_size
#         self.kernel = nn.Parameter(init.xavier_normal_(torch.zeros((self.out_channels, in_channels, self.kernel_height, self.kernel_width))))
#         if bias_presence:
#             self.bias = nn.Parameter(torch.zeros(self.out_channels))
#         else:
#             self.bias = None
    
#     def choose_probability_large(self, input, kernel, min_val, max_val):
#         # If all NaN, return input as is
#         if torch.isnan(input).all(): return input
#         elif not torch.isnan(input).any(): return input


#         # Flatten input and kernel
#         input_flat = input.ravel() #view(-1)
#         kernel_flat = kernel.ravel() #view(-1)

#         # Calculate probabilities for NaN values
#         distance_to_max = max_val - kernel_flat
#         total_distance = max_val - min_val
#         probabilities = 1 - distance_to_max / total_distance

#         # Create a mask for NaN values
#         nan_mask = torch.isnan(input_flat)

#         # Indices of NaN values
#         nan_indices = torch.nonzero(nan_mask).squeeze()

#         # Generate random values from the distribution
#         hist, bin_edges = torch.histogram(input_flat[~nan_mask], bins=5, density=True)
#         random_values = torch.tensor(random.choices(bin_edges[:-1], weights=(hist / torch.sum(hist)), k=nan_indices.numel()))

#         # Replace NaN values with random values
#         # input_flat[nan_indices] = random_values

#         probabilities = torch.mean(probabilities.view(kernel.shape), axis=0).ravel()

#         result = torch.where(
#             torch.isnan(input_flat) & (probabilities > random.random()),
#             random_values[torch.randint(len(random_values), size=input_flat.shape)],
#             input_flat
#         )

#         return result.view(input.shape)

#     def get_window(self, i, j):
#         x_start = i * self.stride
#         y_start = j * self.stride
#         # Calculate the ending point of the receptive field
#         x_end = min(x_start + self.kernel_height, self.in_height + 2 * self.padding)
#         y_end = min(y_start + self.kernel_width, self.in_width + 2 * self.padding)

#         # Extract the image patch
#         return self.image_padded[:, :, x_start:x_end, y_start:y_end]
  
#     def apply_threshold(self, kernel, i, j):

#         image_patch = self.get_window(i, j)
#         # image_patch = choose_probability_large(image_patch, kernel, torch.min(kernel), torch.max(kernel))


#         try:
#             # print(torch.sum(torch.isnan(image_patch)).item(), torch.sum(~torch.isnan(image_patch)).item())
#             # nan_ratio = torch.sum(torch.isnan(image_patch)).item() / torch.sum(~torch.isnan(image_patch)).item() 
#             nan_ratio = torch.sum(torch.isnan(image_patch)).item() / image_patch.numel()
#         except ZeroDivisionError:
#             nan_ratio = 1
        
#         if nan_ratio >= self.threshold: #change to and equal
#             #CHANGE TO SET TO NAN
#             self.output[:, :, i, j] = float('nan') #torch.full(self.output[:, :, i, j].shape, float('nan')) #torch.sum(image_patch.unsqueeze(1) * kernel.unsqueeze(0), dim=(2, 3, 4))
#             # self.output[:, :, i, j] = torch.sum(image_patch.unsqueeze(1) * kernel.unsqueeze(0), dim=(2, 3, 4))
#         else:
#             self.output[:, :, i, j] = torch.nansum(image_patch.unsqueeze(1) * kernel.unsqueeze(0), dim=(2, 3, 4))
            
#     def __call__(self, image):

#         # Pad the input image
#         self.image_padded = torch.nn.functional.pad(image, (self.padding, self.padding, self.padding, self.padding))

#         # Get dimensions
#         self.batch_size, _, self.in_height, self.in_width = self.image_padded.shape

#         self.x_img, self.y_img = image.shape[-2:]

#         self.out_height = int((( self.x_img + 2*self.padding - 1*(self.kernel_height - 1) ) -1)/self.stride ) + 1
#         self.out_width = int((( self.y_img + 2*self.padding - 1*(self.kernel_width - 1) ) -1)/self.stride ) + 1


#         # Initialize output tensor
#         self.output = torch.zeros(self.batch_size, self.out_channels, self.out_height, self.out_width)


#         _ = [self.apply_threshold(self.kernel, i, j) for i in range(self.out_height) for j in range(self.out_width)]
#         # print(type(a) , (a))
#         # self.output = torch.Tensor([self.apply_threshold(self.kernel, i, j) for i in range(self.out_height) for j in range(self.out_width)])


#         if self.bias is not None:
#             self.output += self.bias.view(1, -1, 1, 1)

#         return self.output 

# class NoArgMaxIndices(BaseException):
#     def __init__(self):
#         super(NoArgMaxIndices, self).__init__(
#             "no argmax indices: batch_argmax requires non-batch shape to be non-empty")


# class NaNPool2d:

#     def __init__(self, threshold: float = 0.25):
#         """
#         Initializes the NaNPool2d object.

#         Args:
#             threshold (float, optional): Threshold value for determining multiple max value occurrence ratio. Defaults to 0.5.
#         """
#         self.threshold = threshold

#     # CURRENTLY UNUSED
#     ## unravel_index and batch_argmax are from https://stackoverflow.com/questions/39458193/using-list-tuple-etc-from-typing-vs-directly-referring-type-as-list-tuple-etc
#     ## they generalize well to multi-dimensions in theory but are they necessary? Window is always 3D
#     def unravel_index(self, 
#         indices: torch.LongTensor,
#         shape: Tuple[int, ...],
#     ) -> torch.LongTensor:
#         r"""Converts flat indices into unraveled coordinates in a target shape.

#         This is a `torch` implementation of `numpy.unravel_index`.

#         Args:
#             indices: A tensor of (flat) indices, (*, N).
#             shape: The targeted shape, (D,).

#         Returns:
#             The unraveled coordinates, (*, N, D).
#         """

#         coord = []

#         for dim in reversed(shape):
#             coord.append(indices % dim)
#             indices = indices // dim

#         coord = torch.stack(coord[::-1], dim=-1)

#         return coord

#     # CURRENTLY UNUSED
#     def batch_argmax(self, tensor, batch_dim=1):
#         """
#         Assumes that dimensions of tensor up to batch_dim are "batch dimensions"
#         and returns the indices of the max element of each "batch row".
#         More precisely, returns tensor `a` such that, for each index v of tensor.shape[:batch_dim], a[v] is
#         the indices of the max element of tensor[v].
#         """
#         if batch_dim >= len(tensor.shape):
#             raise NoArgMaxIndices()
#         batch_shape = tensor.shape[:batch_dim]
#         non_batch_shape = tensor.shape[batch_dim:]
#         flat_non_batch_size = prod(non_batch_shape)
#         tensor_with_flat_non_batch_portion = tensor.reshape(*batch_shape, flat_non_batch_size)

#         dimension_of_indices = len(non_batch_shape)

#         # We now have each batch row flattened in the last dimension of tensor_with_flat_non_batch_portion,
#         # so we can invoke its argmax(dim=-1) method. However, that method throws an exception if the tensor
#         # is empty. We cover that case first.
#         if tensor_with_flat_non_batch_portion.numel() == 0:
#             # If empty, either the batch dimensions or the non-batch dimensions are empty
#             batch_size = prod(batch_shape)
#             if batch_size == 0:  # if batch dimensions are empty
#                 # return empty tensor of appropriate shape
#                 batch_of_unraveled_indices = torch.ones(*batch_shape, dimension_of_indices).long()  # 'ones' is irrelevant as it will be empty
#             else:  # non-batch dimensions are empty, so argmax indices are undefined
#                 raise NoArgMaxIndices()
#         else:   # We actually have elements to maximize, so we search for them
#             indices_of_non_batch_portion = tensor_with_flat_non_batch_portion.argmax(dim=-1)
#             batch_of_unraveled_indices = self.unravel_index(indices_of_non_batch_portion, non_batch_shape)

#         if dimension_of_indices == 1:
#             # above function makes each unraveled index of a n-D tensor a n-long tensor
#             # however indices of 1D tensors are typically represented by scalars, so we squeeze them in this case.
#             batch_of_unraveled_indices = batch_of_unraveled_indices.squeeze(dim=-1)
#         return batch_of_unraveled_indices

#     def check_for_nans(self, c, i, j, window, maxval, max_index):

#         # Converting 1d indices from pool window to 2d indices
#         max_index = torch.stack((max_index // self.pool_width, max_index % self.pool_width), dim=1)

#         if torch.isnan(maxval).any():
#             window = window.masked_fill(torch.isnan(window), float("-inf"))
#             maxval = torch.max(window.reshape(self.batch_size, -1), dim=1)[0] 


#         # Strict approach to identifying multiple max values
#         # check_multi_max = torch.sum(window == maxval[:, None, None], axis=(1, 2))
#         # Less restrictive more theoretically stable approach
#         check_multi_max = torch.sum(
#             torch.isclose(window, maxval[:, None, None], rtol=1e-7, equal_nan=True), axis=(1, 2)
#         )

#         # Reduce multiple max value counts to ratios in order to use passed threshold value
#         check_multi_max = check_multi_max / (window.shape[-1] * window.shape[-2])

#         if (check_multi_max > self.threshold).any():
#             maxval = torch.where(check_multi_max > self.threshold, np.nan, maxval)

#         # Find new index of max value if it has changed and is not NaN
#         if torch.where(window == maxval)[0].numel() != 0: 
#             max_index = torch.max(window.masked_fill(torch.isnan(window), float('-inf')).reshape(self.batch_size, -1), dim=1)[1]
#             max_index = torch.stack((max_index // 2, max_index % 2), dim=1)

#         # Calculate the indices for 1D representation
#         max_index_1d = (i * self.stride_height + (max_index[:, 0])) * self.input_width + (
#             j * self.stride_width + (max_index[:, 1])
#         )


#         self.output_array[:, c, i, j] = maxval
#         self.index_array[:, c, i, j] = max_index_1d


#     def __call__(self, input_array: torch.Tensor, pool_size: tuple, strides: tuple = None) -> tuple:
#         """
#         Perform NaN-aware max pooling on the input array.

#         Args:
#             input_array (torch.Tensor): Input tensor of shape (batch_size, channels, input_height, input_width).
#             pool_size (tuple): Size of the pooling window (pool_height, pool_width).
#             strides (tuple, optional): Strides for pooling (stride_height, stride_width). Defaults to None.

#         Returns:
#             tuple: A tuple containing output array and index array after pooling.
#         """

#         batch_size, channels, input_height, input_width = input_array.shape
#         # Force values to int
#         self.batch_size = int(batch_size)
#         channels = int(channels)
#         self.input_height = int(input_height)
#         self.input_width = int(input_width)

#         pool_height, pool_width = pool_size
#         # Force values to int
#         self.pool_height = int(pool_height)
#         self.pool_width = int(pool_width)

#         if strides:
#             stride_height, stride_width = strides

#         else:
#             stride_height, stride_width = pool_size

#         # Force values to int
#         self.stride_height = int(stride_height)
#         self.stride_width = int(stride_width)

#         # Calculate simplified intensity distribution of the layer
#         self.min_intensity = torch.min(input_array)
#         self.max_intensity = torch.max(input_array)

#         # Calculate the output dimensions
#         output_height = int((input_height - pool_height) // stride_height + 1)
#         output_width = int((input_width - pool_width) // stride_width + 1)

#         # Initialize output arrays for pooled values and indices
#         self.output_array = torch.zeros((self.batch_size, channels, output_height, output_width))
#         self.index_array = torch.zeros((self.batch_size, channels, output_height, output_width), dtype=torch.int64)


#         # Perform max pooling with list comprehensions
#         for c in range(channels):

#             # Create a list of tuples with pooled values and indices
#             values_and_indices = [
#                 self.check_for_nans(c, i, j, window, torch.max(window.reshape(self.batch_size, -1), dim=1)[0], torch.max(window.reshape(self.batch_size, -1), dim=1)[1])
#                 for i in range(output_height)
#                 for j in range(output_width)
#                 for window in [
#                     input_array[
#                         :,
#                         c,
#                         i * stride_height : i * stride_height + pool_height,
#                         j * stride_width : j * stride_width + pool_width,
#                     ]
#                 ]
#             ]

#         return torch.Tensor(self.output_array), torch.Tensor(self.index_array).type(torch.int64)

class CompetitiveDenseBlock(nn.Module):
    """
    Define a competitive dense block comprising 3 convolutional layers, with BN/ReLU.

    Attributes
    ----------
     params = {'num_channels': 1,
               'num_filters': 64,
               'kernel_h': 5,
               'kernel_w': 5,
               'stride_conv': 1,
               'pool': 2,
               'stride_pool': 2,
               'num_classes': 44
               'kernel_c':1
               'input':True
               }

    Methods
    -------
    forward
        Feedforward through graph.
    """

    def __init__(self, params: Dict, outblock: bool = False):
        """
        Construct CompetitiveDenseBlock object.

        Parameters
        ----------
        params : Dict
            Dictionary with parameters specifying block architecture.
        outblock : bool
            Flag indicating if last block (Default value = False).
        """
        super(CompetitiveDenseBlock, self).__init__()

        self.params = params

        # Padding to get output tensor of same dimensions
        padding_h = int((params["kernel_h"] - 1) / 2)
        padding_w = int((params["kernel_w"] - 1) / 2)

        # Sub-layer output sizes for BN; and
        conv0_in_size = int(params["num_channels"])  # num_channels
        conv1_in_size = int(params["num_filters"])
        conv2_in_size = int(params["num_filters"])
        out_size = (
            params["num_filters_last"]
            if "num_filters_last" in params
            else params["num_filters"]
        )

        self.conv0_in_size = conv0_in_size
        self.conv1_in_size = conv1_in_size
        self.conv2_in_size = conv2_in_size
        self.out_size = out_size

        # Define the learnable layers
        # Standard conv layers
        self.conv0 = nn.Conv2d(
            in_channels=conv0_in_size,
            out_channels=params["num_filters"],
            kernel_size=(params["kernel_h"], params["kernel_w"]),
            stride=params["stride_conv"],
            padding=(padding_h, padding_w),
        )
        print(f'Conv0:\nOut channels: {params["num_filters"]}\nIn_channels: {conv0_in_size}\nKernel size: {params["kernel_h"], params["kernel_w"]}\nStride: {params["stride_conv"]}\nPadding: {padding_h, padding_w}')
        

        self.conv1 = nn.Conv2d(
            in_channels=conv1_in_size,
            out_channels=params["num_filters"],
            kernel_size=(params["kernel_h"], params["kernel_w"]),
            stride=params["stride_conv"],
            padding=(padding_h, padding_w),
        )
        print(f'Conv1:\nOut channels: {params["num_filters"]}\nIn_channels: {conv0_in_size}\nKernel size: {params["kernel_h"], params["kernel_w"]}\nStride: {params["stride_conv"]}\nPadding: {padding_h, padding_w}')
        

        self.conv2 = nn.Conv2d(
            in_channels=conv1_in_size,
            out_channels=params["num_filters"],
            kernel_size=(params["kernel_h"], params["kernel_w"]),
            stride=params["stride_conv"],
            padding=(padding_h, padding_w),
        )
        print(f'Conv2:\nOut channels: {params["num_filters"]}\nIn_channels: {conv0_in_size}\nKernel size: {params["kernel_h"], params["kernel_w"]}\nStride: {params["stride_conv"]}\nPadding: {padding_h, padding_w}')
        

        # D \times D convolution for the last block
        self.conv3 = nn.Conv2d(
            in_channels=conv2_in_size,
            out_channels=out_size,
            kernel_size=(params["kernel_h"], params["kernel_w"]),
            stride=params["stride_conv"],
            padding=(padding_h, padding_w),
        )
        print(f'Conv3:\nOut channels: {params["num_filters"]}\nIn_channels: {conv0_in_size}\nKernel size: {params["kernel_h"], params["kernel_w"]}\nStride: {params["stride_conv"]}\nPadding: {padding_h, padding_w}')
        
        self.bn1 = nn.BatchNorm2d(num_features=conv1_in_size)
        self.bn2 = nn.BatchNorm2d(num_features=conv2_in_size)
        self.bn3 = nn.BatchNorm2d(num_features=conv2_in_size)
        self.bn4 = nn.BatchNorm2d(num_features=out_size)

        self.prelu = nn.PReLU()  # Learnable ReLU Parameter
        self.outblock = outblock


    def forward(self, x: Tensor) -> Tensor:
        """
        Feedforward through CompetitiveDenseBlock.

        {in (Conv - BN from prev. block) -> PReLU} -> {Conv -> BN -> Maxout -> PReLU} x 2 -> {Conv -> BN} -> out
        end with batch-normed output to allow maxout across skip-connections.

        Parameters
        ----------
        x : Tensor
            Input tensor (image or feature map).

        Returns
        -------
        out
            Output tensor (processed feature map).
        """

        # if torch.isnan(x).any():
        #     bn = ManualBatchNorm2d(num_features=self.conv0_in_size)

        # print('Before Competitive Dense Block', torch.set_num_threads(1))
        # print('Before Competitive Dense Block', torch.get_num_threads())

        # comment all this out
        print('PreLu0')
        # Activation from pooled input
        x0 = self.prelu(x)
        # pickle.dump(x0, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_prelu0_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        print('Conv 0')
        print('Conv 0', self.conv0.weight.shape)
        start_time=time.time()
        # Convolution block 1 (RF: 3x3)
        if os.environ['NAN_ACTIVE'] == "true":
            conv0 = NaNConv2d(train=False, kernel=self.conv0.weight, bias=self.conv0.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
            print(self.conv0.stride, self.conv0.padding)
            print(self.params["stride_conv"], int((self.params["kernel_h"] - 1) / 2))
            x0 = conv0(x0)
        else:
            x0 = self.conv0(x0)

        print(f'Took: {time.time()-start_time} seconds')
        # conv0 = ConvCustom(x0, self.conv0.weight, self.conv0.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
        # x0 = conv0.forward()
        print('SKIP count', count_skip_conv2d(x0, self.conv0.weight, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD'])))

        # pickle.dump(x0, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_conv0_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        # pickle.dump(self.conv0.weight, open(f"/output/outp_conv0weight_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        # pickle.dump(self.conv0.bias, open(f"/output/outp_conv0bias_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        print('BN1')
        # bn1 = ManualBatchNorm2d(num_features=self.conv1_in_size)
        # bn1.train()
        # if torch.isnan(x0).any(): x0, nan_index = bn.replace(x0)
        x1_bn = self.bn1(x0)
        # if torch.isnan(x1_bn).any(): x1_bn = bn.return_nan(x1_bn, nan_index)
        # x1_bn = bn1(x, threshold=float(os.environ['NANCONV_THRESHOLD']))
        # pickle.dump(x1_bn, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn1_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        # pickle.dump(bn1.running_mean, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn1runningmean_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        # pickle.dump(bn1.running_var, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn1runningvar_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        print('maxadd0')
        # First Maxout/Addition
        x1_max = torch.maximum(x, x1_bn)
        # pickle.dump(x1_max, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_maxadd0_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        print('Prelu 1')
        x1 = self.prelu(x1_max)
        # pickle.dump(x1, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_prelu1_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        
        print('Conv 1')
        start_time=time.time()
        # print(self.conv1)
        print('Conv 1', self.conv1.weight.shape)
        # Convolution block 2
        if os.environ['NAN_ACTIVE'] == "true":
            conv1 = NaNConv2d(train=False, kernel=self.conv1.weight, bias=self.conv1.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
            print(self.conv1.stride, self.conv1.padding)
            print(self.params["stride_conv"], int((self.params["kernel_h"] - 1) / 2))
            x1 = conv1(x1)
        else:
            x1 = self.conv1(x1)

        print(f'Took: {time.time()-start_time} seconds')
        # conv1 = ConvCustom(x1, self.conv1.weight, self.conv1.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))        
        # x1 = conv1.forward()
        print('SKIP count', count_skip_conv2d(x0, self.conv1.weight, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD'])))
        # pickle.dump(x1, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_conv1_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        # pickle.dump(self.conv1.weight, open(f"/output/outp_conv1weight_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        # pickle.dump(self.conv1.bias, open(f"/output/outp_conv1bias_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        print('BN2')
        # bn2 = ManualBatchNorm2d(num_features=self.conv2_in_size)
        # bn2.train()
        # if torch.isnan(x1).any(): x1, nan_index = bn.replace(x1)
        x2_bn = self.bn2(x1)
        # if torch.isnan(x2_bn).any(): x2_bn = bn.return_nan(x2_bn, nan_index)
        # x2_bn = bn2(x, threshold=float(os.environ['NANCONV_THRESHOLD']))
        # pickle.dump(x2_bn, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn2_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        # pickle.dump(bn2.running_mean, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn2runningmean_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        # pickle.dump(bn2.running_var, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn2runningvar_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        print('maxadd1')
        # Second Maxout/Addition
        x2_max = torch.maximum(x2_bn, x1_max)
        # pickle.dump(x2_max, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_maxadd1_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        print('Prelu 2')
        x2 = self.prelu(x2_max)
        # pickle.dump(x2, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_prelu2_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        print('Conv 2')
        print('Conv 2', self.conv2.weight.shape)
        start_time=time.time()
        # Convolution block 3
        if os.environ['NAN_ACTIVE'] == "true":
            conv2 = NaNConv2d(train=False, kernel=self.conv2.weight, bias=self.conv2.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
            print(self.conv2.stride, self.conv2.padding)
            print(self.params["stride_conv"], int((self.params["kernel_h"] - 1) / 2))
            x2 = conv2(x2)
        else:
            x2 = self.conv2(x2)

        print(f'Took: {time.time()-start_time} seconds')
        # conv2 = ConvCustom(x2, self.conv2.weight, self.conv2.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
        # x2 = conv2.forward()
        print('SKIP count', count_skip_conv2d(x2, self.conv2.weight, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD'])))

        # pickle.dump(x2, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_conv2_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        # pickle.dump(self.conv2.weight, open(f"/output/outp_conv2weight_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        # pickle.dump(self.conv2.bias, open(f"/output/outp_conv2bias_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        print('BN3')
        # bn3 = ManualBatchNorm2d(num_features=self.conv2_in_size)
        # bn3.train()
        # if torch.isnan(x2).any(): x2, nan_index = bn.replace(x2)
        x3_bn = self.bn3(x2)
        # if torch.isnan(x3_bn).any(): x3_bn = bn.return_nan(x3_bn, nan_index)
        # x3_bn = bn3(x, threshold=float(os.environ['NANCONV_THRESHOLD']))
        # pickle.dump(x3_bn, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn3_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        # pickle.dump(bn3.running_mean, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn3runningmean_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        # pickle.dump(bn3.running_var, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn3runningvar_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        print('maxadd2')
        # Third Maxout/Addition
        x3_max = torch.maximum(x3_bn, x2_max)
        # pickle.dump(x3_max, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_maxadd2_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        print('Prelu 3')
        x3 = self.prelu(x3_max)
        # pickle.dump(x3, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_prelu3_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        print('Conv 3')
        print('Conv 3', self.conv3.weight.shape)
        start_time = time.time()
        # Convolution block 4 (end with batch-normed output to allow maxout across skip-connections)
        if os.environ['NAN_ACTIVE'] == "true":
            conv3 = NaNConv2d(train=False, kernel=self.conv3.weight, bias=self.conv3.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
            print(self.conv3.stride, self.conv3.padding)
            print(self.params["stride_conv"], int((self.params["kernel_h"] - 1) / 2))
            out = conv3(x3)
        else:
            out = self.conv3(x3)

        print(f'Took: {time.time()-start_time} seconds')
        # conv3 = ConvCustom(x3, self.conv3.weight, self.conv3.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
        # out = conv3.forward()
        print('SKIP count', count_skip_conv2d(x3, self.conv3.weight, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD'])))

        # pickle.dump(out, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_conv3_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        # pickle.dump(self.conv3.weight, open(f"/output/outp_conv3weight_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        # pickle.dump(self.conv3.bias, open(f"/output/outp_conv3bias_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))


        if not self.outblock:
            print('BN4')
            # bn4 = ManualBatchNorm2d(num_features=self.out_size)
            # bn4.train()
            # if torch.isnan(out).any(): out, nan_index = bn.replace(out)
            out = self.bn4(out)
            # if torch.isnan(out).any(): out = bn.return_nan(out, nan_index)
            # out = bn4(x, threshold=float(os.environ['NANCONV_THRESHOLD']))
            # pickle.dump(out, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn4_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
            # pickle.dump(bn4.running_mean, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn4runningmean_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
            # pickle.dump(bn4.running_var, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn4runningvar_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        # print('After Competitive Dense Block', torch.set_num_threads(1))
        # print('After Competitive Dense Block', torch.get_num_threads())

        return out


class ManualBatchNorm2d(torch.nn.Module):
    def __init__(self, num_features, epsilon=1e-5, momentum=0.1, affine=True):
        super(ManualBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        self.affine = affine
        
        if self.affine:
            # Learnable parameters for scaling (gamma) and bias (beta)
            self.gamma = torch.nn.Parameter(torch.ones(num_features))
            self.beta = torch.nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
        
        # Running mean and variance for inference
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x, threshold):
        
        # nan_ratio = 0
        # try:
        #     print(torch.sum(torch.isnan(x)).item(), torch.sum(~torch.isnan(x)).item(), x.numel())
        #     # nan_ratio = torch.sum(torch.isnan(x)).item() / torch.sum(~torch.isnan(x)).item() 
        #     nan_ratio = torch.sum(torch.isnan(x)).item() / x.numel() 
        # except ZeroDivisionError:
        #     print('zero division error')
        #     nan_ratio = 1
        # print('NAN RATIO', nan_ratio)
        # if nan_ratio < threshold: 
        #     x = torch.nan_to_num(x)
        print(x.shape)
        try:
            # Get the indices of NaN values
            nan_indices = torch.nonzero(torch.isnan(x))[0]
            x = torch.nan_to_num(x)
        except: print('No NaNs here')

        if self.training:
            batch_mean = x.mean([0, 2, 3], keepdim=True)
            batch_var = x.var([0, 2, 3], unbiased=False, keepdim=True)
            print(batch_mean.shape)

            # Update running mean and variance
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean.squeeze()
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var.squeeze()

            mean = batch_mean
            var = batch_var
        else:
            # Use running mean and variance during inference
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)

        # Normalize the input
        x_norm = (x - mean) / torch.sqrt(var + self.epsilon)

        if self.affine:
            # Apply learnable scaling and bias
            x_norm = x_norm * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        
        try:
        # print(x_norm.shape)
            print('INDICES', nan_indices)
            x_norm[ x_norm == x_norm[nan_indices[0], nan_indices[1], nan_indices[2], nan_indices[3] ]] = float('nan')
            print('NaN after batch', torch.sum(torch.isnan(x_norm)).item(), x_norm.numel())
        except: pass

        return x_norm
    
    def replace(self, x):
        try:
            # Get the indices of NaN values
            nan_indices = torch.nonzero(torch.isnan(x))[0]
            x = torch.nan_to_num(x)
            return x, nan_indices        
        except: 
            print('No NaNs here')

    def return_nan(self, x_norm, nan_indices):
        try:
        # print(x_norm.shape)
            print('INDICES', nan_indices)
            x_norm[ x_norm == x_norm[nan_indices[0], nan_indices[1], nan_indices[2], nan_indices[3] ]] = float('nan')
            print('NaN after batch', torch.sum(torch.isnan(x_norm)).item(), x_norm.numel())
        except: pass
        
        return x_norm
        

    
class CompetitiveDenseBlockInput(nn.Module):
    """
    Define a competitive dense block comprising 3 convolutional layers, with BN/ReLU for input.

    Attributes
    ----------
     params (dict): {'num_channels': 1,
                    'num_filters': 64,
                    'kernel_h': 5,
                    'kernel_w': 5,
                    'stride_conv': 1,
                    'pool': 2,
                    'stride_pool': 2,
                    'num_classes': 44
                    'kernel_c':1
                    'input':True}
     Methods
     -------
     forward
        Feedforward through graph.
    """

    def __init__(self, params: Dict):
        """
        Construct CompetitiveDenseBlockInput object.

        Parameters
        ----------
        params : Dict
            Dictionary with parameters specifying block architecture.
        """
        super(CompetitiveDenseBlockInput, self).__init__()

        self.params = params

        # Padding to get output tensor of same dimensions
        padding_h = int((params["kernel_h"] - 1) / 2)
        padding_w = int((params["kernel_w"] - 1) / 2)

        # Sub-layer output sizes for BN; and
        conv0_in_size = int(params["num_channels"])
        conv1_in_size = int(params["num_filters"])
        conv2_in_size = int(params["num_filters"])

        self.conv0_in_size = conv0_in_size
        self.conv1_in_size = conv1_in_size
        self.conv2_in_size = conv2_in_size

        # Define the learnable layers
        self.conv0 = nn.Conv2d(
            in_channels=conv0_in_size,
            out_channels=params["num_filters"],
            kernel_size=(params["kernel_h"], params["kernel_w"]),
            stride=params["stride_conv"],
            padding=(padding_h, padding_w),
        )

        self.conv1 = nn.Conv2d(
            in_channels=conv1_in_size,
            out_channels=params["num_filters"],
            kernel_size=(params["kernel_h"], params["kernel_w"]),
            stride=params["stride_conv"],
            padding=(padding_h, padding_w),
        )


        self.conv2 = nn.Conv2d(
            in_channels=conv1_in_size,
            out_channels=params["num_filters"],
            kernel_size=(params["kernel_h"], params["kernel_w"]),
            stride=params["stride_conv"],
            padding=(padding_h, padding_w),
        )

        # 1 \times 1 convolution for the last block
        self.conv3 = nn.Conv2d(
            in_channels=conv2_in_size,
            out_channels=params["num_filters"],
            kernel_size=(params["kernel_h"], params["kernel_w"]),
            stride=params["stride_conv"],
            padding=(padding_h, padding_w),
        )

        print('NUM FEATURES', conv0_in_size)
        self.bn0 = nn.BatchNorm2d(num_features=conv0_in_size)

        self.bn1 = nn.BatchNorm2d(num_features=conv1_in_size)

        self.bn2 = nn.BatchNorm2d(num_features=conv2_in_size)

        self.bn3 = nn.BatchNorm2d(num_features=conv2_in_size)

        self.bn4 = nn.BatchNorm2d(num_features=conv2_in_size)

        self.prelu = nn.PReLU()  # Learnable ReLU Parameter

    def forward(self, x: Tensor) -> Tensor:
        """
        Feed forward trough CompetitiveDenseBlockInput.

        in -> BN -> {Conv -> BN -> PReLU} -> {Conv -> BN -> Maxout -> PReLU} -> {Conv -> BN} -> out

        Parameters
        ----------
        x : Tensor
            Input tensor (image or feature map).

        Returns
        -------
        out
            Output tensor (processed feature map).
        """

        # if torch.isnan(x).any():
        #     bn = ManualBatchNorm2d(num_features=self.conv0_in_size)


        print('Batch')
        # Input batch normalization
        print(self.params["num_channels"])
        x0_bn = self.bn0(x)
        # pickle.dump(x0_bn, open(f"/output/encode1/{os.environ['NANCONV_THRESHOLD']}_outp_bn0_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        # pickle.dump(bn0.running_mean, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn0runningmean_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        # pickle.dump(bn0.running_var, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn0runningvar_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))



        print('Conv', x0_bn.shape)
        print('Conv 0', self.conv0.weight.shape)
        # Convolution block1 (RF: 3x3)
        start_time = time.time()
        if os.environ['NAN_ACTIVE'] == "true":
            conv0 = NaNConv2d(train=False, kernel=self.conv0.weight, bias=self.conv0.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
            # dynamo_conv0 = dynamo.optimize("inductor")(conv0)
            # dynamo_conv0 = torch.compile(conv0, mode='reduce-overhead') 
            # print(f'Compilation took: {time.time()-start_time} seconds')
            # start_time = time.time()
            print(self.conv0.stride, self.conv0.padding)
            print(self.params["stride_conv"], int((self.params["kernel_h"] - 1) / 2))
            x0 = conv0(x0_bn)
        else:
            x0 = self.conv0(x0_bn)

        print(f'Took: {time.time()-start_time} seconds')
        # x0 = conv2d_custom(x0_bn, self.conv0.weight, self.conv0.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
        # conv0 = ConvCustom(x0_bn, self.conv0.weight, self.conv0.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
        # x0 = conv0.forward()
        print('SKIP count', count_skip_conv2d(x0_bn, self.conv0.weight, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD'])))

        # pickle.dump(x0, open(f"/output/encode1/{os.environ['NANCONV_THRESHOLD']}_outp_conv0_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        print('Batch')
        x1_bn = self.bn1(x0)
        # pickle.dump(x1_bn, open(f"/output/encode1/{os.environ['NANCONV_THRESHOLD']}_outp_bn1_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        # pickle.dump(bn1.running_mean, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn1runningmean_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        # pickle.dump(bn1.running_var, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn1runningvar_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        print('Prelu')
        x1 = self.prelu(x1_bn)
        # pickle.dump(x1, open(f"/output/encode1/{os.environ['NANCONV_THRESHOLD']}_outp_prelu0_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        # Convolution block2 (RF: 5x5)
        start_time = time.time()
        print('Conv 1', self.conv1.weight.shape)
        if os.environ['NAN_ACTIVE'] == "true":
            conv1 = NaNConv2d(train=False, kernel=self.conv1.weight, bias=self.conv1.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
            # dynamo_conv1 = dynamo.optimize("inductor")(conv1)
            # dynamo_conv1= torch.compile(conv1, mode='reduce-overhead') 
            # print(f'Compilation took: {time.time()-start_time} seconds')
            # start_time = time.time()
            print(self.conv1.stride, self.conv1.padding)
            print(self.params["stride_conv"], int((self.params["kernel_h"] - 1) / 2))
            x1 = conv1(x1) 
        else:
            x1 = self.conv1(x1)
       
        print(f'Took: {time.time()-start_time} seconds')

        # x1 = conv2d_custom(x1, self.conv1.weight, self.conv1.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
        # conv1 = ConvCustom(x1, self.conv1.weight, self.conv1.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
        # x1 = conv1.forward()
        print('SKIP count', count_skip_conv2d(x1, self.conv1.weight, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD'])))
        
        # pickle.dump(x1, open(f"/output/encode1/{os.environ['NANCONV_THRESHOLD']}_outp_conv1_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        print('Batch')
        x2_bn = self.bn2(x1)
        # pickle.dump(x2_bn, open(f"/output/encode1/{os.environ['NANCONV_THRESHOLD']}_outp_bn2_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        # pickle.dump(bn2.running_mean, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn2runningmean_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        # pickle.dump(bn2.running_var, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn2runningvar_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))


        print('Max')
        # First Maxout
        x2_max = torch.maximum(x2_bn, x1_bn)
        # pickle.dump(x2_max, open(f"/output/encode1/{os.environ['NANCONV_THRESHOLD']}_outp_maxout0_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        print('Prelu')
        x2 = self.prelu(x2_max)
        # pickle.dump(x2, open(f"/output/encode1/{os.environ['NANCONV_THRESHOLD']}_outp_prelu1_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        # Convolution block3 (RF: 7x7)
        print('Conv', x2.shape)
        print('Conv 2', self.conv2.weight.shape)
        start_time = time.time()
        if os.environ['NAN_ACTIVE'] == "true":
            conv2 = NaNConv2d(train=False, kernel=self.conv2.weight, bias=self.conv2.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
            # dynamo_conv2 = dynamo.optimize("inductor")(conv2)
            # dynamo_conv2 = torch.compile(conv2, mode='reduce-overhead') 
            # print(f'Compilation took: {time.time()-start_time} seconds')
            # start_time = time.time()
            print(self.conv2.stride, self.conv2.padding)
            print(self.params["stride_conv"], int((self.params["kernel_h"] - 1) / 2))
            x2 = conv2(x2)
        else:
            x2 = self.conv2(x2)

        print(f'Took: {time.time()-start_time} seconds')
        # x2 = conv2d_custom(x2, self.conv2.weight, self.conv2.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
        # conv2 = ConvCustom(x2, self.conv2.weight, self.conv2.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
        # x2 = conv2.forward()
        print('SKIP count', count_skip_conv2d(x2, self.conv2.weight, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD'])))
        
        # pickle.dump(x2, open(f"/output/encode1/{os.environ['NANCONV_THRESHOLD']}_outp_conv2_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        print('Batch')
        x3_bn = self.bn3(x2)
        # pickle.dump(x3_bn, open(f"/output/encode1/{os.environ['NANCONV_THRESHOLD']}_outp_bn3_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        # pickle.dump(bn3.running_mean, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn3runningmean_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        # pickle.dump(bn3.running_var, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn3runningvar_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))


        print('Max')
        # Second Maxout
        x3_max = torch.maximum(x3_bn, x2_max)
        # pickle.dump(x3_max, open(f"/output/encode1/{os.environ['NANCONV_THRESHOLD']}_outp_maxout1_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        print('Prelu')
        x3 = self.prelu(x3_max)
        # pickle.dump(x3, open(f"/output/encode1/{os.environ['NANCONV_THRESHOLD']}_outp_prelu2_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        print('Conv', x3.shape)
        print('Conv 3', self.conv3.weight.shape)
        # Convolution block 4 (RF: 9x9)
        start_time = time.time()
        if os.environ['NAN_ACTIVE'] == "true":
            conv3 = NaNConv2d(train=False, kernel=self.conv3.weight, bias=self.conv3.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
            # dynamo_conv3 = dynamo.optimize("inductor")(conv3)
            # dynamo_conv3 = torch.compile(conv3, mode='reduce-overhead') 
            # print(f'Compilation took: {time.time()-start_time} seconds')
            # start_time = time.time()
            print(self.conv3.stride, self.conv3.padding)
            print(self.params["stride_conv"], int((self.params["kernel_h"] - 1) / 2))
            out = conv3(x3)
        else:
            out = self.conv3(x3)

        print(f'Took: {time.time()-start_time} seconds')
        # out = conv2d_custom(x3, self.conv3.weight, self.conv3.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
        # conv3 = ConvCustom(x3, self.conv3.weight, self.conv3.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
        # out = conv3.forward()
        print('SKIP count', count_skip_conv2d(x3, self.conv3.weight, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD'])))

        # pickle.dump(out, open(f"/output/encode1/{os.environ['NANCONV_THRESHOLD']}_outp_conv3_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        print('Batch')
        out = self.bn4(out)
        # pickle.dump(out, open(f"/output/encode1/{os.environ['NANCONV_THRESHOLD']}_outp_bn4_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        # pickle.dump(bn4.running_mean, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn4runningmean_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        # pickle.dump(bn4.running_var, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn4runningvar_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))


        return out


class GaussianNoise(nn.Module):
    """
    Define a Gaussian Noise Block.

    Methods
    -------
    forward
        Feedforward through graph.
    """

    def __init__(self, sigma: float = 0.1, device: str = "cuda"):
        """
        Construct GaussianNoise object.

        Parameters
        ----------
        sigma : float
             [MISSING] (Default value = 0.1).
        device : str
             [MISSING] (Default value = "cuda").
        """
        super().__init__()
        self.sigma = sigma
        self.noise = torch.tensor(0).to(device)
        self.register_buffer("noise", torch.tensor(0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Feedforward through graph.

        Parameters
        ----------
        x : Tensor
            Input Tensor.

        Returns
        -------
        x : Tensor
            Output tensor (processed feature map).
        """
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach()
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x


def probability_function(x, min_val, max_val):
        distance_to_max = max_val - x
        total_distance = max_val - min_val
        probability = 1 - distance_to_max / total_distance
        
        return probability

def choose_probability(data, min_val, max_val):
    random.seed(0)
    
    # Create a mask to handle NaN values
    mask = torch.isnan(data)
    masked_values = data.masked_fill(mask, float('-inf'))

    # Dictionary comprehension to create probabilities only for NaNs and maximum values
    probs = {x: probability_function(x, min_val, max_val) for x in data if torch.isnan(x) or x == torch.max(masked_values)}
    
    choice = random.random()
    
    # List comprehension to find the first item meeting the probability condition
    selected_value = next((x for x, prob in probs.items() if choice <= prob), np.nan)
    
    return selected_value

def max_pooling2d(input_array, pool_size, strides):

    _, channels, input_height, input_width = input_array.shape
    pool_height, pool_width = pool_size
    stride_height, stride_width = strides

    # Calculate simplified intensity distribution of the layer
    min_intensity = torch.min(input_array)
    max_intensity = torch.max(input_array)

    # Calculate the output dimensions
    output_height = (input_height - pool_height) // stride_height + 1
    output_width = (input_width - pool_width) // stride_width + 1

    # Initialize output arrays for pooled values and indices
    output_array = torch.zeros((1, channels, output_height, output_width))
    index_array = torch.zeros((1, channels, output_height, output_width), dtype=torch.int64)

    # Perform max pooling with list comprehensions
    for c in range(channels):
        # Create a list of tuples with pooled values and indices
        values_and_indices = [
            (torch.max(window), torch.argmax(window))
            for i in range(output_height)
            for j in range(output_width)
            for window in [
                input_array[0, c, i * stride_height:i * stride_height + pool_height, 
                            j * stride_width:j * stride_width + pool_width]
            ]
        ]

        # Handle NaNs and probabilities for choosing max value
        for k, (maxval, max_index) in enumerate(values_and_indices):
            # Re-initialize the window within the second for-loop
            i = k // output_width
            j = k % output_width
            window = input_array[0, c, i * stride_height:i * stride_height + pool_height,
                                 j * stride_width:j * stride_width + pool_width]

            if torch.isnan(maxval):
                window = window.masked_fill(torch.isnan(window), float('-inf'))
                maxval = torch.max(window)
                
            if len(torch.where(window.ravel() == maxval.item())[0]) > 1:
                maxval = choose_probability(window.ravel(), min_intensity, max_intensity)

            # Calculate the indices for 1D representation
            max_index_1d = (i * stride_height + (max_index // pool_width)) * input_width + (j * stride_width + (max_index % pool_width))

            # Update output arrays
            output_array[0, c, i, j] = maxval
            index_array[0, c, i, j] = max_index_1d

    return torch.Tensor(output_array), torch.Tensor(index_array).type(torch.int64)

##
# Encoder/Decoder definitions
##
class CompetitiveEncoderBlock(CompetitiveDenseBlock):
    """
    Encoder Block = CompetitiveDenseBlock + Max Pooling.

    Attributes
    ----------
    maxpool
        Maxpool layer.

    Methods
    -------
    forward
        Feed forward trough graph.
    """

    def __init__(self, params: Dict):
        """
        Construct CompetitiveEncoderBlock object.

        Parameters
        ----------
        params : Dict
            Parameters like number of channels, stride etc.
        """
        super(CompetitiveEncoderBlock, self).__init__(params)
        self.maxpool = nn.MaxPool2d(
            kernel_size=params["pool"],
            stride=params["stride_pool"],
            return_indices=True,
        )  # For Unpooling later on with the indices
        
        self.params = params


    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Feed forward trough Encoder Block.

        * CompetitiveDenseBlock
        * Max Pooling (+ retain indices)

        Parameters
        ----------
        x : Tensor
            Feature map from previous block.

        Returns
        -------
        out_encoder : Tensor
            Original feature map.
        out_block : Tensor
            Maxpooled feature map.
        indicies : Tensor
            Maxpool indices.
        """
        # print('Before Competitive Encoder Block', torch.set_num_threads(1))
        # print('Before Competitive Encoder Block', torch.get_num_threads())
        print('HERE')

        out_block = super(CompetitiveEncoderBlock, self).forward(
            x
        )  # To be concatenated as Skip Connection

        print('OUTBLOCK', out_block.shape)
        start_time=time.time()
        # if os.environ['NAN_ACTIVE'] == "true":
        maxpool = NaNPool2d(max_threshold=int(os.environ['MULTI_MAXVAL']), probabilistic=(os.environ["NAN_PROB"] == 'true'), nan_probability=float(os.environ['NAN_PROBVAL']))
        out_encoder, indices = maxpool(out_block, (self.params["pool"], self.params["pool"]), (self.params["stride_pool"], self.params["stride_pool"]) ) 
        # out_encoder, indices = max_pooling2d(out_block, (self.params["pool"], self.params["pool"]), (self.params["stride_pool"], self.params["stride_pool"]))
        print('Pooling')
        print(self.params["pool"], self.params["stride_pool"])
        print(self.maxpool.kernel_size, self.maxpool.stride)
        # else:
        #     out_encoder, indices = self.maxpool(
        #         out_block
        #     )  # Max Pool as Input to Next Layer
            
        print(f'Took: {start_time - time.time()} seconds')
        print(out_encoder.shape)
        # pickle.dump(out_encoder, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_pool_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        # print('After Competitive Encoder Block', torch.set_num_threads(1))
        # print('After Competitive Encoder Block', torch.get_num_threads())

        return out_encoder, out_block, indices


class CompetitiveEncoderBlockInput(CompetitiveDenseBlockInput):
    """
    Encoder Block = CompetitiveDenseBlockInput + Max Pooling.
    """

    def __init__(self, params: Dict):
        """
        Construct CompetitiveEncoderBlockInput object.

        Parameters
        ----------
        params : Dict
            Parameters like number of channels, stride etc.
        """
        super(CompetitiveEncoderBlockInput, self).__init__(
            params
        )  # The init of CompetitiveDenseBlock takes in params
        self.maxpool = nn.MaxPool2d(
            kernel_size=params["pool"],
            stride=params["stride_pool"],
            return_indices=True,
        )  # For Unpooling later on with the indices

        self.params = params

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Feed forward trough Encoder Block.

        * CompetitiveDenseBlockInput
        * Max Pooling (+ retain indices)

        Parameters
        ----------
        x : Tensor
            Feature map from previous block.

        Returns
        -------
        Tensor
            The original feature map as received by the block.
        Tensor
            The maxpooled feature map after applying max pooling to the original feature map.
        Tensor
            The indices of the maxpool operation.
        """
        out_block = super(CompetitiveEncoderBlockInput, self).forward(
            x
        )  # To be concatenated as Skip Connection

        print('OUTBLOCK', out_block.shape)

        start_time = time.time()
        # if os.environ['NAN_ACTIVE'] == "true":
        maxpool = NaNPool2d(max_threshold=int(os.environ['MULTI_MAXVAL']), probabilistic=(os.environ["NAN_PROB"] == 'true'), nan_probability=float(os.environ['NAN_PROBVAL']))
        out_encoder, indices = maxpool(out_block, (self.params["pool"], self.params["pool"]), (self.params["stride_pool"], self.params["stride_pool"]))
        print('Pooling')
        print(self.params["pool"], self.params["stride_pool"])
        print(self.maxpool.kernel_size, self.maxpool.stride)
        # else:
        #     out_encoder, indices = self.maxpool(
        #         out_block
        #     )  # Max Pool as Input to Next Layer

        print(f'Took: {start_time - time.time()} seconds')

        return out_encoder, out_block, indices


class CompetitiveDecoderBlock(CompetitiveDenseBlock):
    """
    Decoder Block = (Unpooling + Skip Connection) --> Dense Block.
    """

    def __init__(self, params: Dict, outblock: bool = False):
        """
        Construct CompetitiveDecoderBlock object.

        Parameters
        ----------
        params : Dict
            Parameters like number of channels, stride etc.
        outblock : bool
            Flag, indicating if last block of network before classifier
            is created.(Default value = False)
        """
        super(CompetitiveDecoderBlock, self).__init__(params, outblock=outblock)
        self.unpool = nn.MaxUnpool2d(
            kernel_size=params["pool"], stride=params["stride_pool"]
        )

    def forward(self, x: Tensor, out_block: Tensor, indices: Tensor) -> Tensor:
        """
        Feed forward trough Decoder block.

        * Unpooling of feature maps from lower block
        * Maxout combination of unpooled map + skip connection
        * Forwarding toward CompetitiveDenseBlock

        Parameters
        ----------
        x : Tensor
            Input feature map from lower block (gets unpooled and maxed with out_block).
        out_block : Tensor
            Skip connection feature map from the corresponding Encoder.
        indices : Tensor
            Indices for unpooling from the corresponding Encoder (maxpool op).

        Returns
        -------
        out_block
            Processed feature maps.
        """
        # print('Before Competitive Decoder Block', torch.set_num_threads(1))
        # print('Before Competitive Decoder Block', torch.get_num_threads())

        unpool = self.unpool(x, indices)
        # pickle.dump(unpool, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_unpool_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        concat_max = torch.maximum(unpool, out_block)
        # pickle.dump(concat_max, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_concatmax_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))
        
        out_block = super(CompetitiveDecoderBlock, self).forward(concat_max)
        # pickle.dump(out_block, open(f"/output/outp_bn4_v2.pkl", 'wb'))
        print(out_block.shape)

        # print('After Competitive Decoder Block', torch.set_num_threads(1))
        # print('After Competitive Decoder Block', torch.get_num_threads())

        return out_block


class OutputDenseBlock(nn.Module):
    """
    Dense Output Block = (Upinterpolated + Skip Connection) --> Semi Competitive Dense Block.

    Attributes
    ----------
    conv0, conv1, conv2, conv3
        Convolution layers.
    gn0, gn1, gn2, gn3, gn4
        Normalization layers.
    prelu
        PReLU activation layer.

    Methods
    -------
    forward
        Feed forward trough graph.
    """

    def __init__(self, params: dict):
        """
        Construct OutputDenseBlock object.

        Parameters
        ----------
        params : dict
            Parameters like number of channels, stride etc.
        """
        super(OutputDenseBlock, self).__init__()

        self.params = params

        # Padding to get output tensor of same dimensions
        padding_h = int((params["kernel_h"] - 1) / 2)
        padding_w = int((params["kernel_w"] - 1) / 2)

        # Sub-layer output sizes for BN; and
        conv0_in_size = int(params["num_channels"])  # num_channels
        conv1_in_size = int(params["num_filters"])
        conv2_in_size = int(params["num_filters"])
        
        self.conv0_in_size = conv0_in_size
        self.conv1_in_size = conv1_in_size
        self.conv2_in_size = conv2_in_size
        
        # Define the learnable layers
        self.conv0 = nn.Conv2d(
            in_channels=conv0_in_size,
            out_channels=params["num_filters"],
            kernel_size=(params["kernel_h"], params["kernel_w"]),
            stride=params["stride_conv"],
            padding=(padding_h, padding_w),
        )

        self.conv1 = nn.Conv2d(
            in_channels=conv1_in_size,
            out_channels=params["num_filters"],
            kernel_size=(params["kernel_h"], params["kernel_w"]),
            stride=params["stride_conv"],
            padding=(padding_h, padding_w),
        )

        self.conv2 = nn.Conv2d(
            in_channels=conv1_in_size,
            out_channels=params["num_filters"],
            kernel_size=(params["kernel_h"], params["kernel_w"]),
            stride=params["stride_conv"],
            padding=(padding_h, padding_w),
        )

        # D \times D convolution for the last block
        self.conv3 = nn.Conv2d(
            in_channels=conv2_in_size,
            out_channels=params["num_filters"],
            kernel_size=(params["kernel_h"], params["kernel_w"]),
            stride=params["stride_conv"],
            padding=(padding_h, padding_w),
        )

        self.gn1 = nn.BatchNorm2d(conv1_in_size)
        self.gn2 = nn.BatchNorm2d(conv2_in_size)
        self.gn3 = nn.BatchNorm2d(conv2_in_size)
        self.gn4 = nn.BatchNorm2d(conv2_in_size)

        self.prelu = nn.PReLU()  # Learnable ReLU Parameter

    def forward(self, x: Tensor, out_block: Tensor) -> Tensor:
        """
        Feed forward trough Output block.

        * Maxout combination of unpooled map from previous block + skip connection
        * Forwarding toward CompetitiveDenseBlock

        Parameters
        ----------
        x : Tensor
            Up-interpolated input feature map from lower block (gets maxed with out_block).
        out_block : Tensor
            Skip connection feature map from the corresponding Encoder.

        Returns
        -------
        out
            Processed feature maps.
        """

        print("OUTPUT")
        # Concatenation along channel (different number of channels from decoder and skip connection)
        concat = torch.cat((x, out_block), dim=1)
        # pickle.dump(concat, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_concat0_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        # Activation from pooled input
        x0 = self.prelu(concat)
        # pickle.dump(x0, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_prelu0_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        # Convolution block1 (no maxout here), could optionally add dense connection; 3x3
        print('Conv 0')
        if os.environ['NAN_ACTIVE'] == "true":
            conv0 = NaNConv2d(train=False, kernel=self.conv0.weight, bias=self.conv0.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
            print(self.conv0.stride, self.conv0.padding)
            print(self.params["stride_conv"], int((self.params["kernel_h"] - 1) / 2))
            x0 = conv0(x0)
        else:
            x0 = self.conv0(x0)

        # conv0 = ConvCustom(x0, self.conv0.weight, self.conv0.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
        # x0 = conv0.forward()
        print('SKIP count', count_skip_conv2d(x0, self.conv0.weight, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD'])))
        # pickle.dump(x0, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_conv0_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        x1_gn = self.gn1(x0)
        # gn1 = ManualBatchNorm2d(num_features=self.conv1_in_size)
        # gn1.train()
        # x1_gn = gn1(x0, threshold=float(os.environ['NANCONV_THRESHOLD']))
        # pickle.dump(x1_gn, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn0_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        x1 = self.prelu(x1_gn)
        # pickle.dump(x1, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_prelu1_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        # Convolution block2; 5x5
        print('Conv 1')
        if os.environ['NAN_ACTIVE'] == "true":
            conv1 = NaNConv2d(train=False, kernel=self.conv1.weight, bias=self.conv1.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
            print(self.conv1.stride, self.conv1.padding)
            print(self.params["stride_conv"], int((self.params["kernel_h"] - 1) / 2))
            x1 = conv1(x1)
        else:
            x1 = self.conv1(x1)
            
        # conv1 = ConvCustom(x1, self.conv1.weight, self.conv1.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
        # x1 = conv1.forward()
        print('SKIP count', count_skip_conv2d(x1, self.conv1.weight, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD'])))
        # pickle.dump(x1, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_conv1_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        x2_gn = self.gn2(x1)
        # gn2 = ManualBatchNorm2d(num_features=self.conv2_in_size)
        # gn2.train()
        # x2_gn = gn2(x1, threshold=float(os.environ['NANCONV_THRESHOLD']))
        # pickle.dump(x2_gn, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn1_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        # First Maxout
        x2_max = torch.maximum(x1_gn, x2_gn)
        # pickle.dump(x2_max, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_maxout0_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        x2 = self.prelu(x2_max)
        # pickle.dump(x2, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_prelu2_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        # Convolution block3; 7x7
        print('Conv 2')
        if os.environ['NAN_ACTIVE'] == "true":
            conv2 = NaNConv2d(train=False, kernel=self.conv2.weight, bias=self.conv2.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
            print(self.conv2.stride, self.conv2.padding)
            print(self.params["stride_conv"], int((self.params["kernel_h"] - 1) / 2))
            x2 = conv2(x2)
        else:
            x2 = self.conv2(x2)

        # conv2 = ConvCustom(x2, self.conv2.weight, self.conv2.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
        # x2 = conv2.forward()
        print('SKIP count', count_skip_conv2d(x2, self.conv2.weight, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD'])))
        # pickle.dump(x2, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_conv2_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        x3_gn = self.gn3(x2)
        # gn3 = ManualBatchNorm2d(num_features=self.conv2_in_size)
        # gn3.train()
        # x3_gn = gn3(x2, threshold=float(os.environ['NANCONV_THRESHOLD']))
        # pickle.dump(x3_gn, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn2_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        # Second Maxout
        x3_max = torch.maximum(x3_gn, x2_max)
        # pickle.dump(x3_max, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_maxout1_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        x3 = self.prelu(x3_max)
        # pickle.dump(x3, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_prelu3_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        # Convolution block 4; 9x9
        print('Conv 3')
        if os.environ['NAN_ACTIVE'] == "true":
            conv3 = NaNConv2d(train=False, kernel=self.conv3.weight, bias=self.conv3.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
            print(self.conv3.stride, self.conv3.padding)
            print(self.params["stride_conv"], int((self.params["kernel_h"] - 1) / 2))
            out = conv3(x3)
        else:
            out = self.conv3(x3)

        # conv3 = ConvCustom(x3, self.conv3.weight, self.conv3.bias, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
        # out = conv3.forward()
        print('SKIP count', count_skip_conv2d(x3, self.conv3.weight, padding=int((self.params["kernel_h"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD'])))
        # pickle.dump(out, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_conv3_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        out = self.gn4(out)
        # gn4 = ManualBatchNorm2d(num_features=self.conv2_in_size)
        # gn4.train()
        # out = gn4(out, threshold=float(os.environ['NANCONV_THRESHOLD']))
        # pickle.dump(out, open(f"/output/{os.environ['NANCONV_THRESHOLD']}_outp_bn3_{os.environ['SLURM_ARRAY_TASK_ID']}.pkl", 'wb'))

        return out


class ClassifierBlock(nn.Module):
    """
    Classification Block.
    """

    def __init__(self, params: dict):
        """
        Construct ClassifierBlock object.

        Parameters
        ----------
        params : dict
            Parameters like number of channels, stride etc.
        """

        self.params = params
        super(ClassifierBlock, self).__init__()
        self.conv = nn.Conv2d(
            params["num_channels"],
            params["num_classes"],
            params["kernel_c"],
            params["stride_conv"],
        )  # To generate logits

    def forward(self, x: Tensor) -> Tensor:
        """
        Feed forward trough classifier.

        Parameters
        ----------
        x : Tensor
            Output of last CompetitiveDenseDecoder Block.

        Returns
        -------
        logits
            Prediction logits.
        """
        print('Class conv')
        if os.environ['NAN_ACTIVE'] == "true":
            conv = NaNConv2d(train=False, kernel=self.conv.weight, bias=self.conv.bias, padding=0, stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
            print(self.conv.stride, self.conv.padding)
            print(self.params["stride_conv"], int((self.params["kernel_h"] - 1) / 2))
            logits = conv(x)
        else:
            logits = self.conv(x)

        # conv = ConvCustom(x, self.conv.weight, self.conv.bias, padding=int((self.params["kernel_c"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD']))
        # logits = conv.forward()
        print('SKIP count', count_skip_conv2d(x, self.conv.weight, padding=int((self.params["kernel_c"] - 1) / 2), stride=self.params["stride_conv"], threshold=float(os.environ['NANCONV_THRESHOLD'])))

        return logits
