import glob
import pandas as pd 

import torch
import numpy as np 
import albumentations
from PIL import Image 
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from utils.transforms import * 

class CarFolderDataset(Dataset): 
    def __init__(self, list_path_data, is_augment = None, num_class = None, is_train= False, sample_balance = False):
        super().__init__() 

        self.list_path_data = list_path_data 
        self.dict_map_clss_to_id = None 
        self.is_train = is_train

        self.list_image = [] 
        self.list_id = []
        self.sample_balance = sample_balance
        
        self.num_class = num_class
        self.is_augment  = is_augment

        self.process_path()
        
        self.augment = None

        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224,224), interpolation= transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize([0.5], [0.5])
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224), interpolation= transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                # transforms.Normalize([0.5], [0.5])
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711))
            ])



    def add_padding_image(self, image:Image, value = 0):

        image = np.array(image) 
        w, h, _ = image.shape 
        if w < h: 
            new_image = np.zeros((h, h, 3)) + value
            start = (h - w) // 2
            new_image[start:start + w, :, :] = image 
            mask = np.zeros((h, h, 1))
            mask[start:start + w, :, :] = 1 
        elif w > h: 
            new_image = np.zeros((w, w, 3)) + value
            start = (w - h ) // 2 
            new_image[:, start:start+h, :] = image 
            mask = np.zeros((w, w, 1)) 
            mask[:, start:start+h, :] = 1 
        
        else: 
            mask = np.zeros((w, h, 1)) + 1 
            new_image = image 
        new_image = np.array(new_image, dtype=np.uint8)
        return Image.fromarray((new_image)), mask 

    def augment_pipeline(self, image: Image, padding_to_keep_ratio= False): 

        # Input is RGB image

        # Blur augmentation
        if np.random.uniform() < 0.3:
            image = transform_random_blur(image)

        # Downscale augmentation
        if np.random.uniform() < 0.3:
            image = transform_resize(image, resize_range = (24, 112), target_size = 224)

        # Color augmentation
        if np.random.uniform() < 0.3:
            image = transform_adjust_gamma(image)

        if np.random.uniform() < 0.3:
            image = transform_color_jiter(image)
        

        # Noise augmentation
        if np.random.uniform() < 0.15:
            image = transform_gaussian_noise(image, mean = 0.0, var = 30.0)

        # Gray augmentation
        if np.random.uniform() < 0.2:
            image = transform_to_gray(image)
        
        # JPEG augmentation
        if np.random.uniform() < 0.5:
            image = transform_JPEGcompression(image, compress_range = (20, 80))
        
        if padding_to_keep_ratio:
            image, mask = self.add_padding_image(image) 
        
        if np.random.rand() < 0.5:
            w, h = image.size 
            if min(w, h) >= 224:
                random_crop = albumentations.augmentations.crops.transforms.RandomResizedCrop(224, 224, scale=(0.5,1), ratio=(1.0, 1.0), always_apply= True)
                image = np.array(image) 
                image = random_crop(image=image)["image"]
                image = Image.fromarray(image)

        return image 


    def process_path(self):
        list_cls = [] 
        list_is_augment = [] 
        
        for path_folder in self.list_path_data:
            list_cls.extend(os.listdir(path_folder))
        
        list_cls = list(set(list_cls))
        list_cls = sorted(list_cls)

        print("[INFO] Number class: ", len(list_cls))
        if self.num_class is not None:
            assert len(list_cls) == self.num_class

        max_sample = 0 
        if self.sample_balance:
            for cls in list_cls: 
                tmp_sum = 0
                for path_folder in self.list_path_data: 
                    if os.path.isdir(os.path.join(path_folder, cls)):
                        tmp_sum += len(os.listdir(os.path.join(path_folder, cls)))
                if tmp_sum > max_sample: max_sample = tmp_sum 
        
        for idx_folder, path_folder in enumerate(self.list_path_data):
            for idx_cls, cls in enumerate(list_cls): 
                path_cls = os.path.join(path_folder, cls)
                if os.path.isdir(path_cls) is False:
                    continue 
                tmp_list_image = glob.glob(path_cls + "/*.jpg")
                if self.is_augment is not None: 
                    print(path_cls, self.is_augment[idx_folder])
                    list_is_augment.extend([self.is_augment[idx_folder]] * len(tmp_list_image))


                if self.sample_balance and len(tmp_list_image) < max_sample // 2: 
                    tmp_list_image = tmp_list_image * (max_sample // len(tmp_list_image)) 

                tmp_list_id = [idx_cls] * len(tmp_list_image)
                self.list_image.extend(tmp_list_image)
                self.list_id.extend(tmp_list_id)

        self.list_is_augment = list_is_augment

    def __getitem__(self, index):

        path = self.list_image[index]
        id = self.list_id[index]

        image =  Image.open(path).convert("RGB")

        if self.is_train:
            if len(self.list_is_augment) > 0 and self.list_is_augment[index] is True:
                image = self.augment_pipeline(image, padding_to_keep_ratio= True)
            
            if len(self.list_is_augment) == 0: image = self.augment_pipeline(image, padding_to_keep_ratio= True)

        image_tensor = self.transform(image)
        id_tensor = torch.tensor(id).long()
        assert image_tensor.shape[0] == 3, path 
        return image_tensor, id_tensor 

    def __len__(self):
        return len(self.list_image)

