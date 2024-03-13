import os
import copy
import time
import tqdm

import torch
import mlflow
import numpy as np
from torch import autocast
from torch.utils.data import DataLoader

from dataloader import CarFolderDataset

from models.model import CLipClassification
from eval import eval_single_dataset
from utils.utils import cosine_lr, AverageMeter, get_lr
from config import config


def calc_accuracy(logits, labels):
    pred = np.argmax(logits, axis=1)
    pred = np.squeeze(pred)
    labels = np.squeeze(labels)
    assert len(pred) == len(labels)
    tp = np.sum(np.where(pred == labels, 1, 0))

    return float(tp) / float(len(pred))


def finetune():

    use_cuda = True if config.device == "cuda" else False

    mlflow.log_params(config.__dict__)

    # Load Model
    model = CLipClassification(num_class= config.num_class)
    if config.freeze_backbone:
        model.eval() 
        for p in model.model.parameters():
            p.requires_grad = False 
        # model.freeze_apart()
        model.train()
        
    # model = ViTClassification(num_class= config.num_class, pretrained_backbone="/home2/tanminh/Car_classification/pretrained/vit.pt")

    if config.pretrained is not None:
        model.eval()
        model.load_state_dict(torch.load(config.pretrained))
        model.train()
    
    # ||. Data Loader
    print('  - Init train dataloader')
    train_dataset = CarFolderDataset(
        list_path_data= [
            # "/home2/tanminh/Data/CarDataset/Cropped_Data_116cls_2/train",
            "/home/data3/tanminh/car_brand/nice_data/Cropped_Data_New_110324/train"
            # "/home2/tanminh/Data/pseudo_label_2modelBL_0403"
        ],
        is_augment= [
            True,
            # True
        ],
        num_class= config.num_class,
        is_train=True, 
        sample_balance= False
    )
    val_dataset = CarFolderDataset(
        list_path_data= [
            # "/home2/tanminh/Data/CarDataset/Cropped_Data_116cls_2/val"
            "/home/data3/tanminh/car_brand/nice_data/Cropped_Data_New_110324/val"
        ],
        num_class= config.num_class,
        is_train= False,
        sample_balance= False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=True)

    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=False
    )

    num_batches = len(train_loader)
    print('    + Number of batches per epoch: {}'.format(num_batches))

    if use_cuda:
        model = model.cuda()

    # || Loss function
    if config.label_smoothing > 0:
        raise NotImplemented("Please Implemenet")
        print('  - Init LabelSmoothingLoss')
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        print('  - Init CrossEntropyLoss')


    params = [p for name, p in model.named_parameters() if p.requires_grad]
    params_name = [name for name, p in model.named_parameters()
                   if p.requires_grad]
    print('  - Total {} params to training: {}'.format(len(params_name),
          [pn for pn in params_name]))
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay= config.wd)
    scheduler = cosine_lr(optimizer, config.lr,
                          config.warmup_length, config.epochs * num_batches)
    print('  - Init AdamW with cosine learning rate scheduler')

    if config.fp16:
        print('  - Using Auto mixed precision')
        scaler = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)
    start_step = config.START_STEP
    best_acc = 0.0
    step = start_step

    for epoch in range(0, config.epochs):
        print(f"Step: {step}")
        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        avg_losses = AverageMeter()

        for i, batch in enumerate(train_loader):
            start_time = time.time()
            step = step + 1
            lr = scheduler(step)
            optimizer.zero_grad()

            inputs, race_labels = batch
            if use_cuda:
                inputs = inputs.cuda()
                race_labels = race_labels.cuda()
                
            data_time.update(time.time() - start_time)
            # data_time = time.time() - start_time
            start_time = time.time()
            # compute output
            if config.fp16:
                with autocast(device_type="cuda", dtype=torch.float16):
                    race_logits = model(inputs)
                    loss = loss_fn(race_logits, race_labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                # optimizer.zero_grad()

                avg_losses.update(loss.item(), inputs.size(0))
            else:
                race_logits = model(inputs)
                loss = loss_fn(race_logits, race_labels)
                # loss = loss_fn(logits, labels) + loss_teacher(logits, to_one_hot(labels, num_class= cfg.config.number_classes))

                avg_losses.update(loss.item(), inputs.size(0))

                # compute gradient and do SGD step
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

            batch_time.update(time.time() - start_time)
            # start_time = time.time()

            if i % config.print_interval == 0:
                logits_np = race_logits.detach().cpu().numpy()
                race_labels_np = race_labels.detach().cpu().numpy()

                print(logits_np.shape)
                print(race_labels_np)
                train_acc = calc_accuracy(logits_np, race_labels_np)
                print("Race acc: ", train_acc)

                percent_complete = 100 * i / len(train_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(train_loader)}]\t",
                    "Lr: {}\tLoss: {}\tAccuracy: {}\tData (t) {}\tBatch (t) {}".format(
                        get_lr(optimizer), avg_losses.avg, train_acc, data_time.avg, batch_time.avg),
                    flush=True
                )

                metrics = {
                    "avg_loss": avg_losses.avg, 
                    "accur_train": train_acc,
                }
                mlflow.log_metrics(metrics= metrics, step= step)

            # Saving model

            if config.experiment_dir is not None:
                os.makedirs(config.experiment_dir, exist_ok=True)
                if step % config.save_interval == 0:
                    model.eval()

                    tik = time.time()
                    val_acc = eval_single_dataset(
                        model, val_loader)
                    tok = time.time()
                    print('Eval done in', tok - tik)

                    model_path = os.path.join(
                        config.experiment_dir, f'checkpoint_{step+1}.pt')
                    torch.save(model.state_dict(), model_path)
                    is_best = val_acc > best_acc
                    if is_best:
                        print('  - Saving as best checkpoint')
                        torch.save(model.state_dict(), os.path.join(
                            config.experiment_dir, f'checkpoint_model_best.pt'))
                        best_acc = val_acc
                    model.train()
                    
                    metrics = {
                        "acc_val": val_acc
                    }
                    mlflow.log_metrics(metrics=metrics, step=step)


        print(f"Epoch {epoch}:\t Loss: {avg_losses.avg:.5f}\t"
              f"Data(t): {data_time.avg:.3f}\t Batch(t): {batch_time.avg:.3f}")

        # best_acc = max(mean_acc, best_acc)


if __name__ == '__main__':
    experiment_name = 'Template Classification'
    experiment = mlflow.set_experiment(experiment_name=experiment_name)
    run = mlflow.start_run(run_name="Freeze apart",
                           run_id=None,
                           experiment_id=experiment.experiment_id,
                           description="freeze backbone")
    finetune()
