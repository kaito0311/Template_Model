class config: 
    batch_size = 8
    num_workers = 8

    epochs = 32
    START_STEP = 1

    device = "cuda"
    fp16 = True

    label_smoothing = False 

    lr = 1e-6
    wd = 0.01
    warmup_length = 0 


    experiment_dir = "experiment/Openclip_ViT_B_open_backbone_augment_1203_2/"
    pretrained = "experiment/Openclip_ViT_B_open_backbone_augment_1203/checkpoint_25501.pt"
    freeze_backbone = False
    save_interval = int(1000 * 64 / batch_size)
    print_interval = 50


    num_class = 129