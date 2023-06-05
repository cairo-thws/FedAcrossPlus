"""
MIT License

Copyright (c) 2023 Manuel Roeder

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
import copy
import numpy as np
from lightningdata import Office31DataModule
from lightningdata.common.pre_process import inv_preprocess
from torchinfo import summary

from gradcam import GradCamWrapper
import os
import torch
from common import Defaults, create_empty_server_model

torch.manual_seed(0)

def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1 / 0.229, 1 / 0.224, 1 / 0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NET = "resnet34"

path_to_file = os.path.join("data", "pretrained", NET + "office31.pt")

# PREPARE SOURCE DATASET
dataset = Office31DataModule
# the first domain is server source domain
source_idx = 1
domain = dataset.get_domain_names()[source_idx]
source_dm = dataset(data_dir="data",
                    domain=domain,
                    batch_size=1,
                    num_workers=0,
                    shuffle=False
                    )

# prepare dataset mean
source_dm.prepare_data()
source_dm.setup()
data_loader = source_dm.test_dataloader()
best_source_model = create_empty_server_model(name=str(source_dm.get_dataset_name()),
                                              num_classes=31,
                                              lr=Defaults.SERVER_LR,
                                              momentum=Defaults.SERVER_LR_MOMENTUM,
                                              gamma=Defaults.SERVER_LR_GAMMA,
                                              weight_decay=Defaults.SERVER_LR_WD,
                                              epsilon=Defaults.SERVER_LOSS_EPSILON,
                                              net=NET,
                                              pretrain=False)
best_source_model.load_state_dict(torch.load(path_to_file, map_location=DEVICE))
#best_source_protos = torch.load(path_to_protos)

# move source model and prototypes to current device
best_source_model = best_source_model.to(DEVICE)
#best_source_protos = best_source_protos.to(DEVICE)

# go to evaluation mode
best_source_model.eval()

# torch info
#summary(best_source_model.model.head, input_size=(1, 1, 31, 31))
#print(best_source_model.model.backbone)
target_layers = [best_source_model.model.backbone.layer4[-1]]

image_id = 17
image_tensor = source_dm.train_set[image_id][0].unsqueeze(0).to(DEVICE)
targets = source_dm.train_set[image_id][1]
with torch.no_grad():
    _, predictions = best_source_model(image_tensor)
    # logits = nn.Softmax(dim=1)(predictions)
    _, predict = torch.max(predictions, 1)
#image_numpy = recreate_image(image_tensor.cpu().detach())
image_numpy = inv_preprocess(image_tensor.squeeze(0).permute(1, 2, 0)).cpu().detach().numpy()

"""
 for i in range(31):
    targets = np.array(list([i]))
    gradcam = GradCamWrapper(model=best_source_model,
                             target_layers=target_layers,
                             device=DEVICE,
                             targets=targets,
                             image_tensor=image_tensor,
                             image_numpy=image_numpy)
    label_str = "Dataset: " + source_dm.get_dataset_name() + ", Domain: " + source_dm.domain + ", Target in observation: " + source_dm.classes[targets.item()].decode("utf-8")
    gradcam.display(label_str)
"""

gradcam = GradCamWrapper(model=best_source_model,
                         target_layers=target_layers,
                         device=DEVICE,
                         targets=predict,
                         image_tensor=image_tensor,
                         image_numpy=image_numpy)
label_str = "Dataset: " + source_dm.get_dataset_name() + ", Domain: " + source_dm.domain + ", Model prediction: " + source_dm.classes[predict.item()].decode("utf-8")
gradcam.display(label_str)

