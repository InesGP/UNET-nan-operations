import sys
import torch
from torch import nn
import pickle
import os
import time
__all__ = ["FONDUE"]

VIEWS = ['axial', 'coronal', 'sagittal']
#sys.path.insert(1, "/FONDUE/nan_ops.py")
from nan_ops import NaNPool2d, count_skip_conv2d, NaNConv2d
THRESH = float(os.environ['THRESHOLD'])
FILE_ROOT = os.environ['FILE_ROOT']
# Check if the environment variable is set
env_var_value = os.getenv('EPSILON')

# Initialize a global variable if the environment variable is set
if env_var_value:
    EPSILON = float(env_var_value)
else:
    EPSILON = 1e-7

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, name):
        super().__init__()
        self.relu1 = nn.PReLU()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.name = name

    def forward(self, x):
        # BLOCK_1
        # print(x.isnan().sum().item(), flush=True)
        # print(f"{time.ctime()}, Start {self.name}", flush=True)
        x0 = self.relu1(x)
        # print(x0.isnan().sum().item(), flush=True)
        conv1_skipped, conv1_total = count_skip_conv2d(x0, self.conv1.weight.data, padding=1, stride=1, threshold=THRESH)
        x0 = NaNConv2d(train=False,threshold=THRESH, kernel=self.conv1.weight, bias=self.conv1.bias, stride=1, padding=1)(x0)
        # print(x0.isnan().sum().item(), flush=True)
        x_ = torch.unsqueeze(x, 4)
        x0_ = torch.unsqueeze(x0, 4)  # Add Singleton Dimension along 5th
        x1 = torch.cat((x_, x0_), dim=4)  # Concatenating along the 5th dimension
        x1_max, _ = torch.max(x1, 4)
        # print(x1_max.isnan().sum().item(), flush=True)

        x1 = self.relu1(x1_max)
        # print(x1.isnan().sum().item(), flush=True)
        if self.name in ['0_1', '1_0']:
            conv2_skipped, conv2_total = conv1_skipped, conv1_total
        else:
            conv2_skipped, conv2_total = count_skip_conv2d(x1, self.conv2.weight.data, padding=1, stride=1, threshold=THRESH)
        # print(x0.isnan().sum().item(), flush=True)
        x1 = NaNConv2d(train=False, threshold=THRESH, kernel=self.conv2.weight, bias=self.conv2.bias, stride=1, padding=1)(x1)
        # print(x1.isnan().sum().item(), flush=True)
        x1_ = torch.unsqueeze(x1, 4)
        x1_max_ = torch.unsqueeze(x1_max, 4)
        x2 = torch.cat((x1_, x1_max_), dim=4)  # Concatenating along the 5th dimension
        x2_max, _ = torch.max(x2, 4)
        # print(x2_max.isnan().sum().item(), flush=True)

        x2 = self.relu1(x2_max)
        # print(x2.isnan().sum().item(), flush=True)
        if self.name in ['0_1', '1_0']:
            conv3_skipped, conv3_total = conv1_skipped, conv1_total
        else:
            conv3_skipped, conv3_total = count_skip_conv2d(x2, self.conv3.weight.data, padding=1, stride=1, threshold=THRESH)
        # print(x2.isnan().sum().item(), flush=True)
        out = NaNConv2d(train=False, threshold=THRESH, kernel=self.conv3.weight, bias=self.conv3.bias, stride=1, padding=1)(x2)
        # print(out.isnan().sum().item(), flush=True)
        print(f"{time.ctime()}, {self.name}: Conv1: {conv1_skipped}/{conv1_total}, Conv2: {conv2_skipped}/{conv2_total}, Conv3: {conv3_skipped}/{conv3_total}", flush=True)

        return out


class VGGBlockInput(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, name):
        super().__init__()
        self.relu1 = nn.PReLU()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.name = name

    def forward(self, x):
        # Batch normalization for input:
        # conv1_skipped, conv1_total = count_skip_conv2d(x, self.conv1.weight.data, padding=1, stride=1, threshold=1.0)
        conv1_skipped, conv1_total = count_skip_conv2d(x, self.conv1.weight.data, padding=1, stride=1, threshold=THRESH)
        x0 = NaNConv2d(train=False,threshold=THRESH, kernel=self.conv1.weight, bias=self.conv1.bias, stride=1, padding=1)(x)
        x0 = self.relu1(x0)
        x1 = NaNConv2d(train=False,threshold=THRESH, kernel=self.conv2.weight, bias=self.conv2.bias, stride=1, padding=1)(x0)
        # First MaxOut
        x0_ = torch.unsqueeze(x0, 4)  # [BS, C, H, W, 1]
        x1_ = torch.unsqueeze(x1, 4)  # Add Singleton Dimension along 5th [BS, C, H, W, 1]
        x2 = torch.cat((x0_, x1_), dim=4)  # Concatenating along the 5th dimension [BS, C, H, W, 2]
        x2_max, _ = torch.max(x2, 4)  # [BS, C, H, W, 1]

        out = self.relu1(x2_max)
        out = NaNConv2d(train=False,threshold=THRESH, kernel=self.conv3.weight, bias=self.conv3.bias, stride=1, padding=1)(out)
        print(f"{self.name}: Conv1: {conv1_skipped}/{conv1_total}, Conv2: {conv1_skipped}/{conv1_total}, Conv3: {conv1_skipped}/{conv1_total}", flush=True)


        return out


class FONDUE(nn.Module):
    def __init__(self, num_classes, input_channels=7, deep_supervision=False,**kwargs):
        super().__init__()
        # CHANGED: Sets the which view the model is working on.

        nb_filter = 64

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), return_indices=True, ceil_mode=True)
        self.up_even = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

        self.up_odd_both = nn.MaxUnpool2d(kernel_size=(1, 1), stride=(2, 2), padding=(0, 0))
        self.up_odd_h = nn.MaxUnpool2d(kernel_size=(1, 2), stride=(2, 2), padding=(0, 0))
        self.up_odd_w = nn.MaxUnpool2d(kernel_size=(2, 1), stride=(2, 2), padding=(0, 0))

        self.lambda_1 = nn.Parameter(torch.ones(1) * (1 / 4))
        self.lambda_2 = nn.Parameter(torch.ones(1) * (1 / 4))
        self.lambda_3 = nn.Parameter(torch.ones(1) * (1 / 4))
        self.lambda_4 = nn.Parameter(torch.ones(1) * (1 / 4))

        self.alpha_1 = nn.Parameter(torch.ones(1) * (1 / 6))
        self.alpha_2 = nn.Parameter(torch.ones(1) * (1 / 6))
        self.alpha_3 = nn.Parameter(torch.ones(1) * (1 / 6))
        self.alpha_4 = nn.Parameter(torch.ones(1) * (1 / 6))
        self.alpha_5 = nn.Parameter(torch.ones(1) * (1 / 6))
        self.alpha_6 = nn.Parameter(torch.ones(1) * (1 / 6))

        self.conv0_0 = VGGBlockInput(input_channels, nb_filter, nb_filter, '0_0')
        self.conv1_0 = VGGBlock(nb_filter, nb_filter, nb_filter, '1_0')
        self.conv2_0 = VGGBlock(nb_filter, nb_filter, nb_filter, '2_0')
        self.conv3_0 = VGGBlock(nb_filter, nb_filter, nb_filter, '3_0')
        self.conv4_0 = VGGBlock(nb_filter, nb_filter, nb_filter, '4_0')
        self.conv5_0 = VGGBlock(nb_filter, nb_filter, nb_filter, '5_0')
        self.conv6_0 = VGGBlock(nb_filter, nb_filter, nb_filter, '6_0')

        self.conv0_1 = VGGBlock(nb_filter, nb_filter, nb_filter, '0_1')
        self.conv1_1 = VGGBlock(nb_filter, nb_filter, nb_filter, '1_1')
        self.conv2_1 = VGGBlock(nb_filter, nb_filter, nb_filter, '2_1')
        self.conv3_1 = VGGBlock(nb_filter, nb_filter, nb_filter, '3_1')
        self.conv4_1 = VGGBlock(nb_filter, nb_filter, nb_filter, '4_1')
        self.conv5_1 = VGGBlock(nb_filter, nb_filter, nb_filter, '5_1')

        self.conv0_2 = VGGBlock(nb_filter, nb_filter, nb_filter, '0_2')
        self.conv1_2 = VGGBlock(nb_filter, nb_filter, nb_filter, '1_2')
        self.conv2_2 = VGGBlock(nb_filter, nb_filter, nb_filter, '2_2')
        self.conv3_2 = VGGBlock(nb_filter, nb_filter, nb_filter, '3_2')
        self.conv4_2 = VGGBlock(nb_filter, nb_filter, nb_filter, '4_2')

        self.conv0_3 = VGGBlock(nb_filter, nb_filter, nb_filter, '0_3')
        self.conv1_3 = VGGBlock(nb_filter, nb_filter, nb_filter, '1_3')
        self.conv2_3 = VGGBlock(nb_filter, nb_filter, nb_filter, '2_3')
        self.conv3_3 = VGGBlock(nb_filter, nb_filter, nb_filter, '3_3')

        self.conv0_4 = VGGBlock(nb_filter, nb_filter, nb_filter, '0_4')
        self.conv1_4 = VGGBlock(nb_filter, nb_filter, nb_filter, '1_4')
        self.conv2_4 = VGGBlock(nb_filter, nb_filter, nb_filter, '2_4')

        self.conv0_5 = VGGBlock(nb_filter, nb_filter, nb_filter, '0_5')
        self.conv1_5 = VGGBlock(nb_filter, nb_filter, nb_filter, '1_5')

        self.conv0_6 = VGGBlock(nb_filter, nb_filter, nb_filter, '0_6')
        self.norm_intensity = nn.Hardtanh(min_val=0.0, max_val=1.0, inplace=True)
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter, num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter, num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter, num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter, num_classes, kernel_size=1)
            self.final5 = nn.Conv2d(nb_filter, num_classes, kernel_size=1)
            self.final6 = nn.Conv2d(nb_filter, num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter * 4, num_classes, kernel_size=1)
        # self.final = nn.ConvTranspose2d(nb_filter, num_classes, kernel_size=1)
        # self.final = self.final.cuda()

    def forward(self, input, orig_zoom, view):

        def ceil_next_even(f):  # rounds to closest number divisible by 32 with integer remainder
            return 32 * round((f / 32.0) + 1.0)

        def dim_changed(before, after):
            h_before = before.size()[2]
            w_before = before.size()[3]

            h_after = after.size()[2]
            w_after = after.size()[3]

            if h_before / 2.0 != h_after:
                h_ch = True
            else:
                h_ch = False
            if w_before / 2.0 != w_after:
                w_ch = True
            else:
                w_ch = False

            if h_ch == True and w_ch == False:
                return "only_h"
            elif h_ch == False and w_ch == True:
                return "only_w"
            elif h_ch == True and w_ch == True:
                return "both"
            else:
                return "none"

        def unpool(dim_changed, filters, indices):
            return self.up_even(filters, indices)

        def get_embeddings(layer, name):
            # if NAN_OPS_ENABLED:
            #     pickle.dump(layer, open(f"./embeddings/nan_2_0/nan_2_0_{view}/{name}.pkl", "wb"))
            # else:
            #     pickle.dump(layer, open(f"./embeddings/ieee/ieee_{view}/{name}.pkl", "wb"))
            if os.environ['NAN_OPS_ENABLED'] == 'True':
                # index = os.environ["SLURM_ARRAY_TASK_ID"]
                thresh = "".join(os.environ['THRESHOLD'].split('.'))
                # pickle.dump(layer, open(f"./embeddings/nan_skip_{folder}_{view}/{name}_{index}.pkl", "wb"))
                pickle.dump(layer, open(f"/embeddings/thresh{thresh}/raw/{os.environ['VIEW']}_{name}.pkl", "wb"))
            else:
                pickle.dump(layer, open(f"/embeddings/ieee/ieee_{view}/{os.environ['VIEW']}_{name}.pkl", "wb"))
      

        input = torch.squeeze(input)
        shapes = input.size()
        if len(input.size()) == 3 and shapes[0] == 7:
            input = torch.unsqueeze(input, 0)
            shapes = input.size()
        elif len(input.size()) == 3 and shapes[0] != 7:
            print("Error: first dimension is not thick slice shape")

        h = shapes[2]
        w = shapes[3]
        orig_shape = (h, w)

        inner_zoom = 1.0
        interp_mode = "bilinear"
        orig_zoom = orig_zoom[0]
        # alpha = random.gauss(0, 0.1)
        factor = inner_zoom / orig_zoom
        # factor = factor.item()
        # inner_shape2 = (bs, nfilt, ceil_next_even(h*factor), ceil_next_even(w*factor))
        inner_shape = (ceil_next_even(h / factor), ceil_next_even(w / factor))
        # x0_0:
        x0_0 = self.conv0_0(
            input
        )  # input is [BS, 7, max_size, max_size]. x0_0 is [BS, Filtnum, max_size, max_size] and max_size is 320
        # get_embeddings(x0_0, "0_0_base_output")
        x0_02 = torch.nn.functional.interpolate(
            x0_0, size=inner_shape, mode=interp_mode, align_corners=None, recompute_scale_factor=None
        )
        # get_embeddings(x0_02, "0_0_bilinear")
        x1_0 = self.conv1_0(x0_02)  # "x1_0" is [BS, Filtnum, max_size/2, max_size/2]
        get_embeddings(x1_0, "1_0_pool_input")

        # x0_1:
        # Same_level_left_side:
        c1 = x0_0  # [BS, Filtnum, max_size, 448]
        c1 = torch.unsqueeze(c1, 4)  # [BS, Filtnum, max_size, max_size, 1]
        # Diagonal_inferior
        c2 = torch.nn.functional.interpolate(
            x1_0, size=orig_shape, mode=interp_mode, align_corners=None, recompute_scale_factor=None
        )
        # get_embeddings(c2, "1_0_bilinear")
        c2 = torch.unsqueeze(c2, 4)  # [BS, Filtnum, max_size, max_size]
        c_all = torch.cat((c1, c2), dim=4)  # [BS, Filtnum, max_size, max_size, 1]
        c_max, _ = torch.max(c_all, 4)  # [BS, Filtnum, max_size, max_size, 2]
        x0_1 = self.conv0_1(c_max)  # [BS, Filtnum, max_size, max_size, 1]
        get_embeddings(x0_1, "1_0_bilinear")

        #print(f"{time.ctime()}, Pre NanPool 1", flush=True)
        pool_x1_0, indices_x1 = NaNPool2d(rtol_epsilon=EPSILON)(x1_0, (2, 2), (2, 2))
        #print(f"{time.ctime()}, Post NanPool 1", flush=True)
        dim_change_1 = dim_changed(x1_0, pool_x1_0)
        get_embeddings(indices_x1, "1_0_pool_indices")
        get_embeddings(pool_x1_0, "1_0_pool")
        #print(f"{time.ctime()}, Pre 2_0", flush=True)
        x2_0 = self.conv2_0(pool_x1_0)  # "x2_0 is [BS, Filtnum, max_size/4, max_size/4]
        get_embeddings(x2_0, "2_0_pool")


        c1 = x1_0
        c1 = torch.unsqueeze(c1, 4)

        c2 = unpool(dim_change_1, x2_0, indices_x1)
        get_embeddings(c2, "2_0_unpool")
        c2 = torch.unsqueeze(c2, 4)
        c_all = torch.cat((c1, c2), dim=4)
        c_max, _ = torch.max(c_all, 4)
        get_embeddings(c_max, "1_1_max_input")
        x1_1 = self.conv1_1(c_max)

        # x0_2:
        # Same_level_left_side:
        c1 = x0_0
        c1 = torch.unsqueeze(c1, 4)
        c2 = x0_1
        c2 = torch.unsqueeze(c2, 4)
        # Diagonal inferior:
        c3 = torch.nn.functional.interpolate(
            x1_1, size=orig_shape, mode=interp_mode, align_corners=None, recompute_scale_factor=None
        )
        get_embeddings(c3, "1_1_bilinear")
        c3 = torch.unsqueeze(c3, 4)
        c_all = torch.cat((c1, c2, c3), dim=4)
        c_max, _ = torch.max(c_all, 4)
        x0_2 = self.conv0_2(c_max)  # 2GB added to VRAM

        #print(f"{time.ctime()}, Pre NanPool 2", flush=True)
        pool_x2_0, indices_x2 = NaNPool2d(rtol_epsilon=EPSILON)(x2_0, (2, 2), (2, 2))
        #print(f"{time.ctime()}, Pre NanPool 2", flush=True)
        dim_change_2 = dim_changed(x2_0, pool_x2_0)
        get_embeddings(indices_x2, "2_0_pool_indices")
        get_embeddings(pool_x2_0, "2_0_pool")
        x3_0 = self.conv3_0(pool_x2_0)

        # Computing x2_1:
        # Same_level_left_side:
        c1 = x2_0
        c1 = torch.unsqueeze(c1, 4)
        # Diagonal_inferior:
        c2 = unpool(dim_change_2, x3_0, indices_x2)
        get_embeddings(c2, "3_0_unpool")
        c2 = torch.unsqueeze(c2, 4)
        c_all = torch.cat((c1, c2), dim=4)
        c_max, _ = torch.max(c_all, 4)
        get_embeddings(c_max, "2_1_max_input")
        x2_1 = self.conv2_1(c_max)

        # Computing x1_2:
        # Same_level_left_side:
        c1 = x1_0
        c1 = torch.unsqueeze(c1, 4)
        c2 = x1_1
        c2 = torch.unsqueeze(c2, 4)
        # Diagonal_inferior:
        c3 = unpool(dim_change_1, x2_1, indices_x1)
        get_embeddings(c3, "2_1_unpool")
        c3 = torch.unsqueeze(c3, 4)
        c_all = torch.cat((c1, c2, c3), dim=4)
        c_max, _ = torch.max(c_all, 4)
        get_embeddings(c_max, "1_2_max_input")
        x1_2 = self.conv1_2(c_max)

        # Computing x0_3:
        # Same_level_left_side
        c1 = x0_0
        c2 = x0_1
        c3 = x0_2
        c4 = torch.nn.functional.interpolate(
            x1_2, size=orig_shape, mode=interp_mode, align_corners=None, recompute_scale_factor=None
        )
        get_embeddings(c4, "1_2_bilinear")
        c1 = torch.unsqueeze(c1, 4)
        c2 = torch.unsqueeze(c2, 4)
        c3 = torch.unsqueeze(c3, 4)
        c4 = torch.unsqueeze(c4, 4)
        c_all = torch.cat((c1, c2, c3, c4), dim=4)
        c_max, _ = torch.max(c_all, 4)
        x0_3 = self.conv0_3(c_max)  # 2GB added to the model

        #print(f"{time.ctime()}, Pre NanPool 3", flush=True)
        pool_x3_0, indices_x3 = NaNPool2d(rtol_epsilon=EPSILON)(x3_0, (2, 2), (2, 2))
        #print(f"{time.ctime()}, Pre NanPool 3", flush=True)
        dim_change_3 = dim_changed(x3_0, pool_x3_0)
        get_embeddings(indices_x3, "3_0_pool_indices")
        get_embeddings(pool_x3_0, "3_0_pool")
        x4_0 = self.conv4_0(pool_x3_0)

        # Computing x3_1:
        # Same_level_left_side
        c1 = x3_0
        c2 = unpool(dim_change_3, x4_0, indices_x3)
        get_embeddings(c2, "4_0_unpool")
        c1 = torch.unsqueeze(c1, 4)
        c2 = torch.unsqueeze(c2, 4)
        c_all = torch.cat((c1, c2), dim=4)
        c_max, _ = torch.max(c_all, 4)
        get_embeddings(c_max, "3_1_max_input")
        x3_1 = self.conv3_1(c_max)


        c1 = x2_0
        c2 = x2_1
        c3 = unpool(dim_change_2, x3_1, indices_x2)
        get_embeddings(c3, "3_1_unpool")
        c1 = torch.unsqueeze(c1, 4)
        c2 = torch.unsqueeze(c2, 4)
        c3 = torch.unsqueeze(c3, 4)
        c_all = torch.cat((c1, c2, c3), dim=4)
        c_max, _ = torch.max(c_all, 4)
        x2_2 = self.conv2_2(c_max)  # Until here theres 20.2GB of VRAM used
        get_embeddings(c_max, "2_2_max_input")

        c1 = x1_0
        c2 = x1_1
        c3 = x1_2
        c4 = unpool(dim_change_1, x2_2, indices_x1)
        get_embeddings(c4, "2_2_unpool")
        c1 = torch.unsqueeze(c1, 4)
        c2 = torch.unsqueeze(c2, 4)
        c3 = torch.unsqueeze(c3, 4)
        c4 = torch.unsqueeze(c4, 4)
        c_all = torch.cat((c1, c2, c3, c4), dim=4)
        c_max, _ = torch.max(c_all, 4)
        get_embeddings(c_max, "1_3_bilinear_max_input")
        x1_3 = self.conv1_3(c_max)

        c1 = x0_0
        c2 = x0_1
        c3 = x0_2
        c4 = x0_3
        c5 = torch.nn.functional.interpolate(
            x1_3, size=orig_shape, mode=interp_mode, align_corners=None, recompute_scale_factor=None
        )
        get_embeddings(c5, "1_3_bilinear")
        c1 = torch.unsqueeze(c1, 4)
        c2 = torch.unsqueeze(c2, 4)
        c3 = torch.unsqueeze(c3, 4)
        c4 = torch.unsqueeze(c4, 4)
        c5 = torch.unsqueeze(c5, 4)
        c_all = torch.cat((c1, c2, c3, c4, c5), dim=4)
        c_max, _ = torch.max(c_all, 4)  # Up until here theres 22.2GB of VRAM used
        get_embeddings(c_max, "1_3_bilinear_max_output")
        x0_4 = self.conv0_4(c_max)

        #print(f"{time.ctime()}, Pre NanPool 4", flush=True)
        pool_x4_0, indices_x4 = NaNPool2d(rtol_epsilon=EPSILON)(x4_0, (2, 2), (2, 2))
        #print(f"{time.ctime()}, Pre NanPool 4", flush=True)
        dim_change_4 = dim_changed(x4_0, pool_x4_0)
        get_embeddings(indices_x4, "4_0_pool_indices")
        get_embeddings(pool_x4_0, "4_0_pool")
        x5_0 = self.conv5_0(pool_x4_0)

        c1 = x4_0

        c2 = unpool(dim_change_4, x5_0, indices_x4)
        get_embeddings(c2, "5_0_unpool")
        c1 = torch.unsqueeze(c1, 4)
        c2 = torch.unsqueeze(c2, 4)
        c_all = torch.cat((c1, c2), dim=4)
        c_max, _ = torch.max(c_all, 4)
        x4_1 = self.conv4_1(c_max)

        c1 = x3_0
        c2 = x3_1
        c3 = unpool(dim_change_3, x4_1, indices_x3)
        get_embeddings(c3, "4_1_unpool")
        c1 = torch.unsqueeze(c1, 4)
        c2 = torch.unsqueeze(c2, 4)
        c3 = torch.unsqueeze(c3, 4)
        c_all = torch.cat((c1, c2, c3), dim=4)
        c_max, _ = torch.max(c_all, 4)
        x3_2 = self.conv3_2(c_max)

        c1 = x2_0
        c2 = x2_1
        c3 = x2_2
        c4 = unpool(dim_change_2, x3_2, indices_x2)
        get_embeddings(c4, "3_2_unpool")
        c1 = torch.unsqueeze(c1, 4)
        c2 = torch.unsqueeze(c2, 4)
        c3 = torch.unsqueeze(c3, 4)
        c4 = torch.unsqueeze(c4, 4)
        c_all = torch.cat((c1, c2, c3, c4), dim=4)
        c_max, _ = torch.max(c_all, 4)
        x2_3 = self.conv2_3(c_max)

        c1 = x1_0
        c2 = x1_1
        c3 = x1_2
        c4 = x1_3
        c5 = unpool(dim_change_1, x2_3, indices_x1)
        get_embeddings(c5, "2_3_unpool")
        c1 = torch.unsqueeze(c1, 4)
        c2 = torch.unsqueeze(c2, 4)
        c3 = torch.unsqueeze(c3, 4)
        c4 = torch.unsqueeze(c4, 4)
        c5 = torch.unsqueeze(c5, 4)
        c_all = torch.cat((c1, c2, c3, c4, c5), dim=4)
        c_max, _ = torch.max(c_all, 4)  # Up until here theres 22.2GB of VRAM used
        x1_4 = self.conv1_4(c_max)

        c1 = x0_0
        c2 = x0_1
        c3 = x0_2
        c4 = x0_3
        c5 = x0_4
        # c6 = self.up(x1_4, indices_x0)
        c6 = torch.nn.functional.interpolate(
            x1_4, size=orig_shape, mode=interp_mode, align_corners=None, recompute_scale_factor=None
        )
        get_embeddings(c6, "1_4_bilinear")
        c1 = torch.unsqueeze(c1, 4)
        c2 = torch.unsqueeze(c2, 4)
        c3 = torch.unsqueeze(c3, 4)
        c4 = torch.unsqueeze(c4, 4)
        c5 = torch.unsqueeze(c5, 4)
        c6 = torch.unsqueeze(c6, 4)
        c_all = torch.cat((c1, c2, c3, c4, c5, c6), dim=4)
        c_max, _ = torch.max(c_all, 4)  # Up until here theres 22.2GB of VRAM used
        x0_5 = self.conv0_5(c_max)

        #print(f"{time.ctime()}, Pre NanPool 5", flush=True)
        pool_x5_0, indices_x5 = NaNPool2d(rtol_epsilon=EPSILON)(x5_0, (2, 2), (2, 2))
        dim_change_5 = dim_changed(x5_0, pool_x5_0)
        #print(f"{time.ctime()}, Pre NanPool 5", flush=True)
        get_embeddings(indices_x5, "5_0_pool_indices")
        get_embeddings(pool_x5_0, "5_0_pool")
        x6_0 = self.conv6_0(pool_x5_0)

        c1 = x5_0
        c2 = unpool(dim_change_5, x6_0, indices_x5)
        get_embeddings(c2, "6_0_unpool")
        c1 = torch.unsqueeze(c1, 4)
        c2 = torch.unsqueeze(c2, 4)
        c_all = torch.cat((c1, c2), dim=4)
        c_max, _ = torch.max(c_all, 4)
        x5_1 = self.conv5_1(c_max)

        c1 = x4_0
        c2 = x4_1
        c3 = unpool(dim_change_4, x5_1, indices_x4)
        get_embeddings(c3, "5_1_unpool")
        c1 = torch.unsqueeze(c1, 4)
        c2 = torch.unsqueeze(c2, 4)
        c3 = torch.unsqueeze(c3, 4)
        c_all = torch.cat((c1, c2, c3), dim=4)
        c_max, _ = torch.max(c_all, 4)
        x4_2 = self.conv4_2(c_max)

        c1 = x3_0
        c2 = x3_1
        c3 = x3_2
        c4 = unpool(dim_change_3, x4_2, indices_x3)
        get_embeddings(c4, "4_2_unpool")
        c1 = torch.unsqueeze(c1, 4)
        c2 = torch.unsqueeze(c2, 4)
        c3 = torch.unsqueeze(c3, 4)
        c4 = torch.unsqueeze(c4, 4)
        c_all = torch.cat((c1, c2, c3, c4), dim=4)
        c_max, _ = torch.max(c_all, 4)
        x3_3 = self.conv3_3(c_max)

        c1 = x2_0
        c2 = x2_1
        c3 = x2_2
        c4 = x2_3
        c5 = unpool(dim_change_2, x3_3, indices_x2)
        get_embeddings(c5, "3_3_unpool")
        c1 = torch.unsqueeze(c1, 4)
        c2 = torch.unsqueeze(c2, 4)
        c3 = torch.unsqueeze(c3, 4)
        c4 = torch.unsqueeze(c4, 4)
        c5 = torch.unsqueeze(c5, 4)
        c_all = torch.cat((c1, c2, c3, c4, c5), dim=4)
        c_max, _ = torch.max(c_all, 4)  # Up until here theres 22.2GB of VRAM used
        x2_4 = self.conv2_4(c_max)

        c1 = x1_0
        c2 = x1_1
        c3 = x1_2
        c4 = x1_3
        c5 = x1_4
        c6 = unpool(dim_change_1, x2_4, indices_x1)
        get_embeddings(c6, "2_4_unpool")
        c1 = torch.unsqueeze(c1, 4)
        c2 = torch.unsqueeze(c2, 4)
        c3 = torch.unsqueeze(c3, 4)
        c4 = torch.unsqueeze(c4, 4)
        c5 = torch.unsqueeze(c5, 4)
        c6 = torch.unsqueeze(c6, 4)
        c_all = torch.cat((c1, c2, c3, c4, c5, c6), dim=4)
        c_max, _ = torch.max(c_all, 4)  # Up until here theres 22.2GB of VRAM used
        x1_5 = self.conv1_5(c_max)

        c1 = x0_0
        c2 = x0_1
        c3 = x0_2
        c4 = x0_3
        c5 = x0_4
        c6 = x0_5
        # c7 = self.up(x1_5, indices_x0)
        c7 = torch.nn.functional.interpolate(
            x1_5, size=orig_shape, mode=interp_mode, align_corners=None, recompute_scale_factor=None
        )
        get_embeddings(c7, "1_5_bilinear")
        c1 = torch.unsqueeze(c1, 4)
        c2 = torch.unsqueeze(c2, 4)
        c3 = torch.unsqueeze(c3, 4)
        c4 = torch.unsqueeze(c4, 4)
        c5 = torch.unsqueeze(c5, 4)
        c6 = torch.unsqueeze(c6, 4)
        c7 = torch.unsqueeze(c7, 4)
        c_all = torch.cat((c1, c2, c3, c4, c5, c6, c7), dim=4)
        c_max, _ = torch.max(c_all, 4)  # Up until here theres 22.2GB of VRAM used
        x0_6 = self.conv0_6(c_max)

        if self.deep_supervision:
            output1 = torch.mul(self.final1(x0_1), self.alpha_1)
            get_embeddings(output1, "y1")
            output2 = torch.mul(self.final2(x0_2), self.alpha_2)
            get_embeddings(output2, "y2")
            output3 = torch.mul(self.final3(x0_3), self.alpha_3)
            get_embeddings(output3, "y3")
            output4 = torch.mul(self.final4(x0_4), self.alpha_4)
            get_embeddings(output4, "y4")
            output5 = torch.mul(self.final4(x0_5), self.alpha_5)
            get_embeddings(output5, "y5")
            output6 = torch.mul(self.final4(x0_6), self.alpha_6)
            get_embeddings(output6, "y6")
            alpha_sum = self.alpha_1 + self.alpha_2 + self.alpha_3 + self.alpha_4 + self.alpha_5 + self.alpha_6
            output = (output1 + output2 + output3 + output4 + output5 + output6) / alpha_sum
            input = input[:, 3]
            input = torch.unsqueeze(input, 1)
            denoised = input - output
            denoised = self.norm_intensity(denoised)
            get_embeddings(denoised, "denoised")
            return denoised, self.alpha_1, self.alpha_2, self.alpha_3, self.alpha_4, self.alpha_5, self.alpha_6

        # else:
        #     output = self.final(x0_4)
        #     return input-output

        else:
            supercat = torch.cat((x0_1, x0_2, x0_3, x0_4), dim=1)
            # supercat = torch.max(supercat,dim=1)
            output = self.final(supercat)  # should be size [BS, 1, 448, 448]
            # output = self.bbproj(output)

            input = input[:, 3, :, :]  # To convert output from [BS, 7, 448, 448] to [BS, 448, 448].
            input = torch.unsqueeze(input, 1)  # Output has shape [BS, 448, 448] as of now
            return input - output

