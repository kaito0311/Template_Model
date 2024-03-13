import os 

import cv2 
import torch 
import numpy as np 
import pandas as pd
import seaborn as sn
from tqdm import tqdm 
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle

def eval_single_dataset(image_classifier, dataloader, device = "cuda", return_accur = True):
    # if args.freeze_encoder:
    #     model = image_classifier.classification_head
    #     input_key = 'features'
    #     image_enc = image_classifier.image_encoder
    # else:
    print('Evaluating ...')
    model = image_classifier
    model.eval()

    all_label = [] 
    all_pred = [] 

    batched_data = enumerate(dataloader)

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in batched_data:
            inputs, race_labels= data

            inputs = inputs.cuda()
            race_labels = race_labels.cuda()

            race_logits = model(inputs) 
             
            pred = torch.nn.functional.softmax(race_logits, dim=1)
     
            pred = race_logits.argmax(dim=1, keepdim=True).to(device)
            all_label.extend(race_labels.detach().cpu().numpy().tolist())
            all_pred.extend(pred.view_as(race_labels).detach().cpu().numpy().tolist())
            correct += pred.eq(race_labels.view_as(pred)).sum().item()
            n += race_logits.size(0)
            print(correct)
            print(n)
                

        top1 = correct / n
    print("Accuracy: ", top1)
    
    if return_accur:
        return top1
    else:
        return all_label, all_pred