"""
MIT License

Copyright (c) 2024 Manuel Roeder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import torch
import torch.nn as nn
import numpy as np
from lightningdata.common.pre_process import inv_preprocess
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def inv_preprocess_img_tensor(img_tensor):
    return inv_preprocess(img_tensor.squeeze(0).permute(1, 2, 0)).cpu().detach().numpy()


@dataclass(frozen=False, init=True)
class GradCamWrapper:
    model: nn.Module
    target_layers: List[nn.Module]
    device: str
    targets: List[int]
    image_tensor: torch.Tensor
    image_numpy: np.ndarray
    reshape_transform: Optional[Callable] = None
    use_cuda: bool = field(init=False)
    target_categories: List[ClassifierOutputTarget] = field(init=False)

    def __post_init__(self) -> None:
        self.use_cuda = self.device == "cuda"
        self.target_categories = [
            ClassifierOutputTarget(target) for target in self.targets
        ]
        self.gradcam = self._init_gradcam_object()

    def _init_gradcam_object(self) -> GradCAM:
        return GradCAM(
            model=self.model,
            target_layers=self.target_layers,
            use_cuda=self.use_cuda,
            reshape_transform=self.reshape_transform,
        )

    def _generate_heatmap(self) -> np.ndarray:
        heatmap = self.gradcam(
            input_tensor=self.image_tensor,
            targets=self.target_categories,
        )
        return heatmap

    def display(self, labelstr="") -> None:
        heatmap = self._generate_heatmap()
        heatmap = heatmap[0, :]
        visualization = show_cam_on_image(self.image_numpy, heatmap, use_rgb=True)

        fig, axes = plt.subplots(figsize=(20, 10), ncols=3)
        fig.suptitle(labelstr, fontsize=25)

        axes[0].imshow(self.image_numpy)
        axes[0].axis("off")

        axes[1].imshow(heatmap)
        axes[1].axis("off")

        axes[2].imshow(visualization)
        axes[2].axis("off")

        plt.show()
