import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import Subset, DataLoader

from source.constants import RESULTS_PATH
from source.utils.seeding import fix_seeds
from source.utils.train_utils import fit
from source.data.face_detection import get_fair_face
from source.data.utils import GroupEncodingDataset


###############
### Parsing ###
###############

parser = argparse.ArgumentParser()
# general
parser.add_argument("--target", default=0, type=int)
parser.add_argument("--encode_group", default=False, type=bool)
parser.add_argument("--group", default=0, type=int)
parser.add_argument("--network", default="resnet50")
parser.add_argument("--method_seed", default=42, type=int) 
parser.add_argument("--data_seed", default=42, type=int)
parser.add_argument("--device", default="cuda:0")
# Network
parser.add_argument("--lr", default=5e-2, type=float)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--weight_decay", default=1e-3, type=float)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--patience", default=0, type=int)
parser.add_argument("--num_workers", default=0, type=int)
# Ensemble
parser.add_argument("--num_networks", default=10, type=int)

# parse
args = parser.parse_args()

# convinience
method_seed, data_seed, device = args.method_seed, args.data_seed, args.device
print("Computation executed on >", device)

# check target
assert args.target in list(range(len(["age", "gender", "race (old)", "race"])))
# check network
assert args.network in ["resnet18", "resnet34", "resnet50", "efficientnet", "regnet"]

if args.encode_group:
    run_path = os.path.join(RESULTS_PATH,
                            f"fairface_pa{args.group}_target{args.target}_{args.network}_mseed{method_seed}_dseed{data_seed}")
else:
    run_path = os.path.join(RESULTS_PATH, 
                            f"fairface_target{args.target}_{args.network}_mseed{method_seed}_dseed{data_seed}")
os.makedirs(run_path, exist_ok=True)

# save command line arguments
formatted_args = "\n".join(f"{key}: {value}" for key, value in vars(args).items())
with open(os.path.join(run_path, "args.txt"), "w") as file:
    file.write(formatted_args)

#################
### LOAD DATA ###
#################

dataset, _ = get_fair_face(target=args.target, protected_attribute=args.group, binarize=True)
if args.encode_group:
    dataset = GroupEncodingDataset(dataset)
# protected attribute not needed for training
dataset.yield_protected_attribute = False

rng = np.random.default_rng(seed=data_seed)
splitting = 8

val_inds = rng.choice(np.arange(len(dataset)), size=len(dataset) // splitting, replace=False)
train_inds = np.delete(np.arange(len(dataset)), (val_inds))
fair_inds = rng.choice(train_inds, size=len(dataset) // splitting, replace=False)
train_inds = np.delete(np.arange(len(dataset)), np.concatenate((fair_inds, val_inds)))

print(len(train_inds), len(val_inds), len(fair_inds))

# for training just train and val datasets necessary
train_ds = Subset(dataset, indices=train_inds)
val_ds = Subset(dataset, indices=val_inds)

# save train / test indices for reproducibility
torch.save(torch.LongTensor(fair_inds), os.path.join(run_path, "fair_inds.pt"))
torch.save(torch.LongTensor(val_inds), os.path.join(run_path, "val_inds.pt"))

####################
### LEARN MODELS ###
####################

fix_seeds(seed=method_seed)

for n in range(args.num_networks):

    if args.network == "resnet18":
        # do not use pretrained weights
        network = tv.models.resnet18(weights=None) 
        network.fc = nn.Linear(in_features=512, out_features=2)
    elif args.network == "resnet34":
        # do not use pretrained weights
        network = tv.models.resnet34(weights=None) 
        network.fc = nn.Linear(in_features=512, out_features=2)
    elif args.network == "resnet50":
        # do not use pretrained weights
        network = tv.models.resnet50(weights=None) 
        network.fc = nn.Linear(in_features=2048, out_features=2)
    elif args.network == "efficientnet":
        network = tv.models.efficientnet_v2_s(weights=None)
        network.classifier = nn.Linear(in_features=1280, out_features=2)
    elif args.network == "regnet":
        network = tv.models.regnet_y_800mf(weights=None)
        network.fc = nn.Linear(in_features=784, out_features=2)
    else:
        raise NotImplementedError("Network not supported")
    network.to(device)
    network.train()

    network, val_accs = fit(network = network, 
                            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers), 
                            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
                            epochs = args.epochs,
                            lr = args.lr, 
                            weight_decay = args.weight_decay, 
                            use_adam = False, 
                            patience = args.patience, 
                            verbose = False)

    os.makedirs(os.path.join(run_path, "models"), exist_ok=True)
    torch.save(network.state_dict(), os.path.join(run_path, "models", f"model_{n}.pt"))
    
    # save val_accs to file as text file & remove if existed previously
    if n == 0 and os.path.exists(os.path.join(run_path, f"val_accs.txt")):
        os.remove(os.path.join(run_path, f"val_accs.txt"))
    with open(os.path.join(run_path, f"val_accs.txt"), "a") as file:
        file.write(f"{n}: {(max(val_accs) * 100):.2f}%\n")

    # print highest val_acc  
    print(f"Model {n} trained with val_acc: {(max(val_accs) * 100):.2f}%")
    