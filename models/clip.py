import open_clip


class CLipClassification(torch.nn.Module):
    def __init__(self, num_class):
        super().__init__()

        self.model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
        print(preprocess_train)
        print(preprocess_val)
        self.fc = torch.nn.Linear(512, num_class)

        # self.model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms( "ViT-L-14",
        #                                                                                     pretrained='laion2b_s32b_b82k')
        # print(preprocess_train)
        # print(preprocess_val)
        # self.fc = torch.nn.Linear(768, num_class)
        

    def freeze_apart(self):
        for p in self.model.parameters():
            p.requires_grad = True
        
        for p in self.model.visual.parameters():
            p.requires_grad = True 
        # # Open some last layer resnet 
        # for p in self.model.transformer.resblocks.parameters():
        #     p.requires_grad = True 
        
        # for p in self.model.token_embedding.parameters():
        #     p.requires_grad = True
        
 
        
    def forward(self, x):
        out = self.model(x)
        return self.fc(out[0])


