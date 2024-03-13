import onnx 
import onnxruntime
import torch.nn as nn 
import torch
from torchvision import datasets, models, transforms
import cv2
import torch 
from PIL import Image
import numpy as np 
import onnxruntime

class AddPrePostONNX(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.mean = [0.5, 0.5, 0.5]
        self.std  = [0.5, 0.5, 0.5]        
        # self.mean = [0.48145466, 0.4578275, 0.40821073]
        # self.std  = [0.26862954, 0.26130258, 0.27577711]        
        self.model = model
        self.model.eval()

    def forward(self, input):
        R = (input[:, 0:1, :, :] - self.mean[0]) / self.std[0]
        G = (input[:, 1:2, :, :] - self.mean[1]) / self.std[1]
        B = (input[:, 2:3, :, :] - self.mean[2]) / self.std[2]
        output_preprocess = torch.cat([R, G, B], dim = 1)

            
        logit = self.model(output_preprocess)
        prob = torch.softmax(logit, dim=1)
        return prob

device = "cpu"

from models.clip import CLipClassification
model_ft = CLipClassification(113)
model_ft.load_state_dict(torch.load("experiment/Openclip_ViT_L_open_backbone_augment_wo_keep_ratio_add_pseudo_label_0703_augment_all_2/checkpoint_model_best.pt", map_location="cpu"))
model_ft.to("cpu")
model_ft.eval() 

model_blood = AddPrePostONNX(model_ft)
model_blood.to(device)
model_blood.eval()
img = cv2.imread('/home2/tanminh/Car_classification/image_transforms.jpg')
img = cv2.resize(img, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(img)
img = np.expand_dims(img, 0).astype('float32') / 255.0
img = torch.Tensor(img).to(device)
img = torch.permute(img, (0, 3, 1, 2))

torch.onnx.export(model_blood,               # model being run
                    img,                         # model input (or a tuple for multiple inputs)
                    "Openclip_ViT_L_open_backbone_augment_wo_keep_ratio_add_pseudo_label_0703_augment_all_2_best.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=11,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
