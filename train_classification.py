
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


import random
import numpy as np
import argparse

from data_loader_voxel import LunaFalsePositiveDataset
from pytorch3dunet.unet3d.model import ResidualUNetFeatureExtract
from pytorch3dunet.unet3d.losses import DiceLoss
from pytorch3dunet.unet3d.metrics import MeanIoU

from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score




def parse_args():
    parser = argparse.ArgumentParser(description="Nudole Locolization")
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--train_csv', type=str, help="Training images csv file")
    parser.add_argument('--train_data_dir', type=str, help="Training images directory")
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for training")
    parser.add_argument('--val_stride', type=int, default=10, help="Stride to extract validation data")
    parser.add_argument('--output_csv', type=str, help="Output filename")

    return parser.parse_args()

# check device
def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else 'cpu'

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)


if __name__=='__main__':

    # get device
    device = get_device()
    print(f'DEVICE: {device}')
    args = parse_args()
    train_dataset = LunaFalsePositiveDataset(val_stride=args.val_stride, isValSet_bool=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    valid_dataset = LunaFalsePositiveDataset(val_stride=args.val_stride, isValSet_bool=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = ResidualUNetFeatureExtract(in_channels=1, out_channels=2)

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-5)

    criterion = nn.CrossEntropyLoss()


    train_losses_all = []
    train_accs_all   = []
    val_losses_all = []
    val_accs_all   = []

    old_val_acc = 0

    for epoch in range(args.epochs):
        model.train()


        train_losses = []
        train_accs = []

        for (inputs, targets) in tqdm(train_loader):
            inputs = inputs.reshape(-1, inputs.shape[-3], inputs.shape[-2], inputs.shape[-1]).unsqueeze(1)
            targets = targets.reshape(-1)
            optimizer.zero_grad()
            logits = model(inputs.to(device))  
            loss = criterion(logits, targets.to(device))
            loss.backward()
            optimizer.step()
            # if hasattr(model, 'final_activation') and model.final_activation is not None:
            #         output = model.final_activation(logits)
            # acc = acc_matrix(output, targets.to(device))
            output = torch.argmax(logits,axis=1)
            acc = precision_score(output.detach().cpu().numpy(), targets.detach().cpu().numpy())

            train_losses.append(loss.item())
            train_accs.append(acc.item())

        train_loss = sum(train_losses) / len(train_losses)
        train_acc = sum(train_accs) / len(train_accs)

        print(f"[ Train | {epoch + 1:03d}/{args.epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        train_losses_all.append(train_loss)
        train_accs_all.append(train_acc)


        model.eval()


        val_losses = []
        val_accs = []

        for (inputs, targets) in tqdm(valid_loader):
            inputs = inputs.reshape(-1, inputs.shape[-3], inputs.shape[-2], inputs.shape[-1]).unsqueeze(1)
            targets = targets.reshape(-1)
    
            logits = model(inputs.to(device))  

            loss = criterion(logits, targets.to(device))
            
            # if hasattr(model, 'final_activation') and model.final_activation is not None:
            #         output = model.final_activation(logits)
            # acc = acc_matrix(logits, targets.to(device))
            output = torch.argmax(logits,axis=1)
            acc = precision_score(output.detach().cpu().numpy(), targets.detach().cpu().numpy())

            val_losses.append(loss.item())
            val_accs.append(acc.item())

        val_loss = sum(val_losses) / len(val_losses)
        val_acc = sum(val_accs) / len(val_accs)

        print(f"[ val | {epoch + 1:03d}/{args.epochs:03d} ] loss = {val_loss:.5f}, acc = {val_acc:.5f}")

        if(val_acc > old_val_acc):
            print("saveing the model for final project...")
            torch.save(model.state_dict(), "classification.pth")
            old_val_acc = val_acc

        val_losses_all.append(val_loss)
        val_accs_all.append(val_acc)