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

# IMPORTS
import time
from typing import Optional, Dict, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.autograd import Variable

from FastSurferCNN.utils import logging
from FastSurferCNN.models.networks import build_model
from FastSurferCNN.data_loader.augmentation import ToTensorTest
from FastSurferCNN.data_loader.data_utils import map_prediction_sagittal2full
from FastSurferCNN.data_loader.dataset import MultiScaleOrigDataThickSlices

# from joblib import Parallel, delayed
import pickle
logger = logging.getLogger(__name__)


class Inference:

    permute_order: Dict[str, Tuple[int, int, int, int]]
    device: Optional[torch.device]
    default_device: torch.device

    def __init__(
        self,
        cfg,
        device: torch.device,
        ckpt: str = "",
        lut: Union[None, str, np.ndarray] = None,
    ):
        # Set random seed from configs.
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        self.cfg = cfg

        # Switch on denormal flushing for faster CPU processing
        # seems to have less of an effect on VINN than old CNN
        torch.set_flush_denormal(True)

        self.default_device = device

        # Options for parallel run
        self.model_parallel = (
            torch.cuda.device_count() > 1
            and self.default_device.type == "cuda"
            and self.default_device.index is None
        )

        # Initial model setup
        self.model = None
        self._model_not_init = None
        self.setup_model(cfg, device=self.default_device)
        self.model_name = self.cfg.MODEL.MODEL_NAME

        self.alpha = {"sagittal": 0.2}
        self.permute_order = {
            "axial": (3, 0, 2, 1),
            "coronal": (2, 3, 0, 1),
            "sagittal": (0, 3, 2, 1),
        }
        self.lut = lut

        # Initial checkpoint loading
        if ckpt:
            # this also moves the model to the para
            self.load_checkpoint(ckpt)

    def setup_model(self, cfg=None, device: torch.device = None):
        if cfg is not None:
            self.cfg = cfg
        if device is None:
            device = self.default_device

        # Set up model
        self._model_not_init = build_model(
            self.cfg
        )  # ~ model = FastSurferCNN(params_network)
        self._model_not_init.to(device)
        self.device = None

    def set_cfg(self, cfg):
        self.cfg = cfg

    def to(self, device: Optional[torch.device] = None):
        if self.model_parallel:
            raise RuntimeError(
                "Moving the model to other devices is not supported for multi-device models."
            )
        _device = self.default_device if device is None else device
        self.device = _device
        self.model.to(device=_device)

    def load_checkpoint(self, ckpt):
        logger.info("Loading checkpoint {}".format(ckpt))

        self.model = self._model_not_init
        # If device is None, the model has never been loaded (still in random initial configuration)
        if self.device is None:
            self.device = self.default_device

        # workaround for mps (directly loading to map_location=mps results in zeros)
        device = self.device
        if self.device.type == "mps":
            self.model.to("cpu")
            device = "cpu"
        else:
            # make sure the model is, where it is supposed to be
            self.model.to(self.device)

        model_state = torch.load(ckpt, map_location=device)
        self.model.load_state_dict(model_state["model_state"])

        # workaround for mps (move the model back to mps)
        if self.device.type == "mps":
            self.model.to(self.device)

        if self.model_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def get_modelname(self):
        return self.model_name

    def get_cfg(self):
        return self.cfg

    def get_num_classes(self):
        return self.cfg.MODEL.NUM_CLASSES

    def get_plane(self):
        return self.cfg.DATA.PLANE

    def get_model_height(self):
        return self.cfg.MODEL.HEIGHT

    def get_model_width(self):
        return self.cfg.MODEL.WIDTH

    def get_max_size(self):
        if self.cfg.MODEL.OUT_TENSOR_WIDTH == self.cfg.MODEL.OUT_TENSOR_HEIGHT:
            return self.cfg.MODEL.OUT_TENSOR_WIDTH
        else:
            return self.cfg.MODEL.OUT_TENSOR_WIDTH, self.cfg.MODEL.OUT_TENSOR_HEIGHT

    def get_device(self):
        return self.device
    

    @torch.no_grad()
    def eval(
        self,
        init_pred: torch.Tensor,
        test_dataset: Dataset,
        # val_loader: DataLoader,
        *,
        out_scale=None,
        out: Optional[torch.Tensor] = None,
    ):
        """Perform prediction and inplace-aggregate views into pred_prob. Return pred_prob."""
        self.model.eval()
        # we should check here, whether the DataLoader is a Random or a SequentialSampler, but we cannot easily.
        # if not isinstance(val_loader.sampler, torch.utils.data.SequentialSampler):
        #     logger.warning(
        #         "The Validation loader seems to not use the SequentialSampler. This might interfere with "
        #         "the assumed sorting of batches."
        #     )

        start_index = 0
        plane = self.cfg.DATA.PLANE
        index_of_current_plane = self.permute_order[plane].index(0)
        target_shape = init_pred.shape
        ii = [slice(None) for _ in range(4)]
        pred_ii = tuple(slice(i) for i in target_shape[:3])

        from tqdm.contrib.logging import logging_redirect_tqdm
        from tqdm import tqdm
        import os


        if out is None:
            out = init_pred.detach().clone()
        with logging_redirect_tqdm():

            print("VALLOADER", len(test_dataset))

            batch_idx = int(os.environ['BRAIN_SLICE_INDEX'])
            start_index = int(os.environ['BRAIN_SLICE_INDEX'])

            sample_batch = test_dataset[int(os.environ['BRAIN_SLICE_INDEX'])] 

            x = sample_batch['image']
            x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])

            images = Variable(torch.from_numpy(x))
            scale_factors = sample_batch['scale_factor']

            # predict the current batch, outputs logits
            pred = self.model(images, scale_factors, out_scale)
            batch_size = pred.shape[0]
            end_index = start_index + batch_size

            # print(index_of_current_plane)
            # print(start_index, end_index)
            # exit(0)

            # check if we need a special mapping (e.g. as for sagittal)
            if self.get_plane() == "sagittal":
                pred = map_prediction_sagittal2full(
                    pred, num_classes=self.get_num_classes(), lut=self.lut
                )

            # permute the prediction into the out slice order
            pred = pred.permute(*self.permute_order[plane]).to(
                out.device
            )  # the to-operation is implicit

            # cut prediction to the image size
            pred = pred[pred_ii]
            # print(pred.shape)

            # add prediction logits into the output (same as multiplying probabilities)
            ii[index_of_current_plane] = slice(start_index, end_index)
            # print(ii)

            out[tuple(ii)].add_(pred, alpha=self.alpha.get(plane, 0.4))
            # start_index = end_index
            if '.' in os.environ['THRESHOLD']:
                thresh = "".join(os.environ['THRESHOLD'].split('.'))
                pickle.dump({"index": tuple(ii), "slice": pred}, open(f"/output/{self.get_plane()}/thresh{thresh}/{os.environ['BRAIN_SLICE_INDEX']}_slice.pkl", "wb"))
            else:
                pickle.dump({"index": tuple(ii), "slice": pred}, open(f"/output/{self.get_plane()}/thresh{os.environ['THRESHOLD']}/{os.environ['BRAIN_SLICE_INDEX']}_slice.pkl", "wb"))

            # pickle.dump({"index": tuple(ii), "slice": pred}, open(f"/output/{self.get_plane()}/{os.environ['BRAIN_SLICE_INDEX']}_slice.pkl", "wb"))
            # pickle.dump({"index": tuple(ii), "slice": pred}, open(f"/output/{os.environ['BRAIN_SLICE_INDEX']}_slice.pkl", "wb"))

            exit(0)

            print(self.get_plane())
            # for view in ['sagittal', 'axial', 'coronal']:
            for i in range(256):
                
                a=pickle.load(open(f'/output/{self.get_plane()}/{i}_slice.pkl', 'rb'))
                # if '.' in os.environ['THRESHOLD']:
                #     thresh = "".join(os.environ['THRESHOLD'].split('.'))
                #     a=pickle.load(open(f"/output/{self.get_plane()}/thresh{thresh}/{i}_slice.pkl", 'rb'))

                # else:
                #     a=pickle.load(open(f"/output/{self.get_plane()}/thresh{os.environ['THRESHOLD']}/{i}_slice.pkl", 'rb'))

                # a=pickle.load(open(f'/output/{i}_slice.pkl', 'rb'))

                if i == 0: 
                    # a=pickle.load(open(f'/output/{self.get_plane()}/{i}_slice.pkl', 'rb'))
                    print(a['index'])
                # out[a['index']] = a['slice']
                out[a['index']].add_(a['slice'], alpha=self.alpha.get(plane, 0.4))


            # for batch_idx, batch in tqdm(
            #     enumerate(val_loader), total=len(val_loader), unit="batch"
            # ):
            

            #     # move data to the model device
            #     images, scale_factors = batch["image"].to(self.device), batch[
            #         "scale_factor"
            #     ].to(self.device)

            #     # predict the current batch, outputs logits
            #     pred = self.model(images, scale_factors, out_scale)
            #     batch_size = pred.shape[0]
            #     end_index = start_index + batch_size

            #     # check if we need a special mapping (e.g. as for sagittal)
            #     if self.get_plane() == "sagittal":
            #         pred = map_prediction_sagittal2full(
            #             pred, num_classes=self.get_num_classes(), lut=self.lut
            #         )

            #     # permute the prediction into the out slice order
            #     pred = pred.permute(*self.permute_order[plane]).to(
            #         out.device
            #     )  # the to-operation is implicit

            #     # cut prediction to the image size
            #     pred = pred[pred_ii]

            #     # add prediction logits into the output (same as multiplying probabilities)
            #     ii[index_of_current_plane] = slice(start_index, end_index)
            #     out[tuple(ii)].add_(pred, alpha=self.alpha.get(plane, 0.4))
            #     start_index = end_index


        return out

    @torch.no_grad()
    def run(
        self,
        init_pred: torch.Tensor,
        image_name,
        orig_data,
        orig_zoom,
        out: Optional[torch.Tensor] = None,
        out_res=None,
        batch_size: int = None,
    ):
        """Run the loaded model on the data (T1) from orig_data and image_name (for messages only) with scale factors orig_zoom."""
        # Set up DataLoader
        test_dataset = MultiScaleOrigDataThickSlices(
            orig_data,
            orig_zoom,
            self.cfg,
            transforms=transforms.Compose([ToTensorTest()]),
        )

        test_data_loader = DataLoader(
            dataset=test_dataset,
            shuffle=False,
            batch_size=self.cfg.TEST.BATCH_SIZE if batch_size is None else batch_size,
        )



        # Run evaluation
        start = time.time()
        # out = self.eval(init_pred, test_data_loader, out=out, out_scale=out_res)
        out = self.eval(init_pred, test_dataset, out=out, out_scale=out_res)
        time_delta = time.time() - start
        logger.info(
            f"{self.cfg.DATA.PLANE.capitalize()} inference on {image_name} finished in "
            f"{time_delta:0.4f} seconds"
        )

        return out
