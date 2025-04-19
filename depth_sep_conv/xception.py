import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import torch.nn.init as init
from typing import Tuple
from math import prod
import torch.utils.model_zoo as model_zoo
from typing import Tuple, Optional
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
import os
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import urllib
from PIL import Image
from nan_ops import count_skip_conv2d, NaNConv2d
import csv
from filelock import FileLock
import sys

""" 
Model implementation courtesy of: https://github.com/tstandley/Xception-PyTorch/blob/master/xception.py 
Creates an Xception Model as defined in:

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf


This weights ported from the Keras implementation. Achieves the following performance on the validation set:

Loss:0.9173 Prec@1:78.892 Prec@5:94.292

REMEMBER to set your image size to 3x299x299 for both test and validation

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""


__all__ = ['xception']
NAN_CONV_THRESH = float(os.environ['NAN_CONV_THRESH'])
EPSILON = float(os.environ['EPSILON'])

model_urls = {
#     'xception':'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
    'xception':'http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth'
}

class NaNPool2d(nn.Module):

    def __init__(self, max_threshold: int = 1, probabilistic: bool = False, rtol_epsilon: float = 1e-7, nan_probability: Optional[float] = 0.8, padding: int = 1, pool_size: Tuple = (3,3), strides: Tuple = (2,2)):
        """
        Initializes the NaNPool2d object.

        Args:
            max_threshold (float, optional): Max threshold value for determining multiple max value occurrence ratio. Defaults to 0.5.
        """
        super().__init__()
        torch.manual_seed(0)
        self.max_threshold = max_threshold
        self.nan_probability = nan_probability
        self.probabilistic = probabilistic
        self.rtol_epsilon = rtol_epsilon
        self.padding = padding

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


    def check_for_nans(self, c, i, j, window, maxval, max_index):

        # Handle NaNs across the batch in a vectorized way
        nan_mask = torch.isnan(window)  # Find NaNs in maxval
        # print(window.shape, torch.isnan(window).sum())
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


        ## if entire window is NaN and maxval=NaN then max_index is 1D instead of 2D
        if torch.isnan(maxval).all() or torch.isinf(maxval).all():
          max_index = torch.zeros((self.batch_size, 2), dtype=torch.long)  # Default to index [0,0] for NaNs


        # Compute 1D indices for output
        try:
          max_index_1d = (i * self.stride_height + max_index[:, 0]) * self.input_width + (j * self.stride_width + max_index[:, 1])
        except:
          print(maxval, max_index, i,j )

          
        # Assign the values and indices
        self.output_array[:, c, i, j] = maxval
        self.index_array[:, c, i, j] = max_index_1d


    # def __call__(self, input_array: torch.Tensor, pool_size: Tuple, strides: Tuple = None) -> Tuple:
    def __call__(self, input_array: torch.Tensor, ) -> Tuple:
        """
        Perform NaN-aware max pooling on the input array.

        Args:
            input_array (torch.Tensor): Input tensor of shape (batch_size, channels, input_height, input_width).
            pool_size (tuple): Size of the pooling window (pool_height, pool_width).
            strides (tuple, optional): Strides for pooling (stride_height, stride_width). Defaults to None.

        Returns:
            tuple: A tuple containing output array and index array after pooling.
        """

        self.image_padded = torch.nn.functional.pad(input_array, (self.padding, self.padding, self.padding, self.padding)).to(dtype=torch.float32)


        batch_size, channels, input_height, input_width = self.image_padded.shape
        # Force values to int
        self.batch_size = int(batch_size)
        channels = int(channels)
        self.input_height = int(input_height)
        self.input_width = int(input_width)

        # Calculate simplified intensity distribution of the layer
        self.min_intensity = torch.min(self.image_padded)
        self.max_intensity = torch.max(self.image_padded)

        # Calculate the output dimensions
        output_height = int((input_height - self.pool_height) // self.stride_height + 1)
        output_width = int((input_width - self.pool_width) // self.stride_width + 1)

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
                    self.image_padded[
                        :,
                        c,
                        i * self.stride_height : i * self.stride_height + self.pool_height,
                        j * self.stride_width : j * self.stride_width + self.pool_width,
                    ]
                ]
            ]

        return torch.Tensor(self.output_array) #, torch.Tensor(self.index_array).type(torch.int64)


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        # x = self.conv1(x)
        # weights_and_biases = {
        #     "conv1_weights": self.conv1.weight,
        #     "padding": self.conv1.padding,
        #     "stride": self.conv1.stride,
        #     "groups": self.conv1.groups,
        # }
        # torch.save(weights_and_biases, 'skipconv.pth')
        print('SKIP count', count_skip_conv2d(x, self.conv1.weight, padding=self.conv1.padding[0],
                                                     stride=self.conv1.stride[0], threshold=NAN_CONV_THRESH) )
        conv1 = NaNConv2d(train=False, kernel=self.conv1.weight, bias=self.conv1.bias,
                          padding=self.conv1.padding[0], stride=self.conv1.stride[0], groups=self.conv1.groups, threshold=NAN_CONV_THRESH)
        x = conv1(x)
        print('DIFF SKIP count', torch.isnan(x).sum(), x.numel())

        # x = self.pointwise(x)
        print('SKIP count', count_skip_conv2d(x, self.pointwise.weight, padding=self.pointwise.padding[0],
                                              stride=self.pointwise.stride[0], threshold=NAN_CONV_THRESH ) )
        pointwise = NaNConv2d(train=False, kernel=self.pointwise.weight, bias=self.pointwise.bias,
                          padding=self.pointwise.padding[0], stride=self.pointwise.stride[0], groups=self.pointwise.groups, threshold=NAN_CONV_THRESH)
        x = pointwise(x)
        print('DIFF SKIP count', torch.isnan(x).sum(), x.numel())


        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            print('strides', strides)
            # rep.append(nn.MaxPool2d(3,strides,1))
            # nn.MaxPool2d(kernel_size=3, stride=strides, padding=1)

            rep.append( NaNPool2d(max_threshold=1, probabilistic=True, rtol_epsilon=EPSILON ) ) #hardcode it to 3, strides, 1
            # x = maxpool(out_block, (self.params["pool"], self.params["pool"]), (self.params["stride_pool"], self.params["stride_pool"]) )

        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp, )

        if self.skip is not None:
            # skip = self.skip(inp)
            print('SKIP count', count_skip_conv2d(x, self.skip.weight, padding=self.skip.padding[0],
                                                         stride=self.skip.stride[0], threshold=NAN_CONV_THRESH) )
            skip_conv = NaNConv2d(train=False, kernel=self.skip.weight, bias=self.skip.bias,
                          padding=self.skip.padding[0], stride=self.skip.stride[0], groups=self.skip.groups, threshold=NAN_CONV_THRESH)
            skip = skip_conv(inp)
            print('DIFF SKIP count', torch.isnan(x).sum(), x.numel())

            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x



class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()


        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)



        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------





    def forward(self, x):
        # x = self.conv1(x)
        print('SKIP count', count_skip_conv2d(x, self.conv1.weight, padding=self.conv1.padding[0],
                                                     stride=self.conv1.stride[0],  threshold=NAN_CONV_THRESH) )
        conv1 = NaNConv2d(train=False, kernel=self.conv1.weight, bias=self.conv1.bias,
                          padding=self.conv1.padding[0], stride=self.conv1.stride[0], threshold=NAN_CONV_THRESH)
        x = conv1(x)
        print('DIFF SKIP count', torch.isnan(x).sum(), x.numel())
        print(torch.isnan(x).any())
        x = self.bn1(x)
        x = self.relu(x)

        # x = self.conv2(x)
        print('SKIP count', count_skip_conv2d(x, self.conv2.weight, padding=self.conv2.padding[0],
                                                     stride=self.conv2.stride[0], threshold=NAN_CONV_THRESH) )
        conv2 = NaNConv2d(train=False, kernel=self.conv2.weight, bias=self.conv2.bias,
                          padding=self.conv2.padding[0], stride=self.conv2.stride[0], groups=self.conv2.groups, threshold=NAN_CONV_THRESH)
        x = conv2(x)
        print('DIFF SKIP count', torch.isnan(x).sum(), x.numel())
        print(torch.isnan(x).any())
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        print(torch.isnan(x).any(), 'block1')
        # plt.imshow(x.squeeze().mean(0));
        # plt.show();
        print(x.shape)
        # return x
        x = self.block2(x)
        print(torch.isnan(x).any(), 'block2')
        # plt.imshow(x.squeeze().mean(0));
        # plt.show();
        x = self.block3(x)
        print(torch.isnan(x).any(), 'block3')
        # plt.imshow(x.squeeze().mean(0));
        # plt.show();
        x = self.block4(x)
        print(torch.isnan(x).any(), 'block4')
        # plt.imshow(x.squeeze().mean(0));
        # plt.show();
        x = self.block5(x)
        print(torch.isnan(x).any(), 'block5')
        # plt.imshow(x.squeeze().mean(0));
        # plt.show();
        x = self.block6(x)
        print(torch.isnan(x).any(), 'block6')
        # plt.imshow(x.squeeze().mean(0));
        # plt.show();
        x = self.block7(x)
        print(torch.isnan(x).any(), 'block7')
        # plt.imshow(x.squeeze().mean(0));
        # plt.show();
        x = self.block8(x)
        print(torch.isnan(x).any(), 'block8')
        # plt.imshow(x.squeeze().mean(0));
        # plt.show();
        x = self.block9(x)
        print(torch.isnan(x).any(), 'block9')
        # plt.imshow(x.squeeze().mean(0));
        # plt.show();
        x = self.block10(x)
        print(torch.isnan(x).any(), 'block10')
        # plt.imshow(x.squeeze().mean(0));
        # plt.show();
        x = self.block11(x)
        print(torch.isnan(x).any(), 'block11')
        # plt.imshow(x.squeeze().mean(0));
        # plt.show();
        x = self.block12(x)
        print(torch.isnan(x).any(), 'block12')
        # plt.imshow(x.squeeze().mean(0));
        # plt.show();

        x = self.conv3(x)
        print(torch.isnan(x).any())
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        print(torch.isnan(x).any())
        x = self.bn4(x)
        x = self.relu(x)
        # plt.imshow(x.squeeze().mean(0));
        # plt.show();

        x = torch.nan_to_num(x)
        # plt.imshow(x.squeeze().mean(0));
        # plt.show();

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def xception(pretrained=False,**kwargs):
    """
    Construct Xception.
    """
    model = Xception(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['xception']))
    return model

def load_model(filename, **kwargs):
    model = Xception(**kwargs)
    model.load_state_dict(torch.load(filename))
    print(f"Model loaded from {filename}")
    return model

def write_prediction_row(filename, preds, csv_path="predictions.csv"):
    lock = FileLock(csv_path + ".lock")  # Will create predictions.csv.lock
    with lock:  # Waits for the lock to be available
        with open(csv_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([filename] + list(preds.numpy()))
    # lock.release()


if __name__ == "__main__":

    # url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
    # urllib.request.urlretrieve(url, filename)
    # with open("imagenet_classes.txt", "r") as f:
    #     categories = [s.strip() for s in f.readlines()]

    with open('imagenet_classes.txt', "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # model = xception(pretrained=True)
    model = load_model('xception.pt')
    model.eval()  

    transform = transforms.Compose([
        transforms.Resize(333),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # Example image loading
    img = Image.open(f"/scratch/ine5/ILSVRC2012_val/ILSVRC2012_val_{str(os.environ['SUBJECT_ID']).zfill(8)}.JPEG").convert("RGB")
    input_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1)
        print(f"Predicted class index: {predicted_class.item()}")

    probs = torch.nn.functional.softmax(output, dim=1)


    top5_prob, top5_catid = torch.topk(probs, 5)

    #write_prediction_row(f"ILSVRC2012_val_{str(os.environ['SUBJECT_ID']).zfill(8)}", top5_catid, csv_path=f"predictions_{''.join(os.environ['NAN_CONV_THRESH'].split('.'))}_{os.environ['EPSILON']}.csv")
    write_prediction_row(f"ILSVRC2012_val_{str(os.environ['SUBJECT_ID']).zfill(8)}", top5_catid, csv_path=f"predictions_{os.environ['NAN_CONV_THRESH']}_{os.environ['EPSILON']}.csv")

    print('Done')
    sys.exit()

    # for i in range(top5_prob.size(1)):
    #     # print(top5_catid[i])
    #     print(categories[top5_catid.squeeze()[i].item()], top5_prob.squeeze()[i].item())
    #     # print(categories[top5_catid[i]], top5_prob[i].item())

