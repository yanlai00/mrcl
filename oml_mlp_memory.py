import argparse
import logging

import numpy as np
import torch
import torch.utils.data
import wandb

from datasets.task_sampler import OmniglotSampler
from model.modelfactory import mlp_config
import utils.utils as utils
from model.memory_meta_learner import MemoryMetaLearner
import torchvision.transforms as transforms
from datasets.omniglot import Omniglot
import datetime

def main(args):

    log_path = "./results/" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '_' + args.name
    wandb.init(project="meta-cl", entity="yanlaiy", config=args)
    device = torch.device('cuda')

    # Using first 963 classes of the omniglot as the meta-training set
    args.classes = list(range(963))
    args.traj_classes = list(range(int(963 / 2), 963))

    data_transform = transforms.Compose(
                [transforms.Resize((args.img_size, args.img_size)),
                 transforms.ToTensor()])

    train_dataset = Omniglot(args.path, background=True, download=True, train=True, transform=data_transform, all=True)
    test_dataset = Omniglot(args.path, background=True, download=True, train=False, transform=data_transform, all=True)

    # Iterators used for evaluation
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle=True, num_workers=1)

    sampler = OmniglotSampler(args.classes, train_dataset, test_dataset)

    config = mlp_config(input_dimension=2024, output_dimension=2024, hidden_size=2000, num_rep_layer=2, num_adapt_layer=2)

    maml = MemoryMetaLearner(model_config=config, adapt_lr=args.update_lr, meta_lr=args.meta_lr).to(device)

    for step in range(args.steps):

        t1 = np.random.choice(args.traj_classes, args.tasks, replace=False)

        d_traj_iterators = []
        for t in t1:
            d_traj_iterators.append(sampler.sample_task([t]))

        d_rand_iterator = sampler.get_complete_iterator()

        x_spt, y_spt, x_qry, y_qry = maml.sample_training_data(d_traj_iterators, d_rand_iterator,
                                                               steps=args.update_step, reset=not args.no_reset)
        
        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

        loss = maml(x_spt, y_spt, x_qry, y_qry)

        # Evaluation during training for sanity checks
        if step % 50 == 0:
            wandb.log({'/metatrain/train/loss': loss}, step=step)
            print(loss.item())
        if step % 1000 == 0:
            utils.val_memory_oml(maml, log_path, test_loader, device, step)
            utils.val_memory_oml(maml, log_path, train_loader, device, step)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--steps', type=int, help='epoch number', default=200000)
    argparser.add_argument('--gpus', type=int, help='meta-level outer learning rate', default=1)
    argparser.add_argument('--rank', type=int, help='meta batch size, namely task num', default=0)
    argparser.add_argument('--tasks', type=int, help='meta batch size, namely task num', default=1)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=10)
    argparser.add_argument('--dataset', help='Name of experiment', default="omniglot")
    argparser.add_argument("--no-reset", action="store_true")
    argparser.add_argument('--seed', help='Seed', default=90, type=int)
    argparser.add_argument('--name', help='Name of experiment', default="oml_memory")
    argparser.add_argument('--path', help='Path of the dataset', default="../")
    argparser.add_argument('--img_size', type=int, help='size of image', default=32)

    args = argparser.parse_args()

    main(args)
