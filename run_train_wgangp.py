import os
import sys
from wgan import Generator, Discriminator
import pandas as pd
import torch
from torch.utils.data import DataLoader

from constants import TARGET_COL
from dataset import TabularDataset
from split import split_data_normal_train_mixed_val
from train_wgangp import train_wgangp


def main(opt):
    if type(opt.seed) is int:
        torch.manual_seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv(opt.dataset_path)
    train_frac = 0.5
    random_state = 1337
    train, val = split_data_normal_train_mixed_val(data, train_frac, random_state=random_state)

    X_train, y_train = train.drop(columns=[TARGET_COL]), train[TARGET_COL]
    X_val, y_val = val.drop(columns=[TARGET_COL]), val[TARGET_COL]

    input_shape = X_train.shape[1]

    train_dataset = TabularDataset(X_train, y_train)

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    generator = Generator(opt, input_shape)
    discriminator = Discriminator(opt, input_shape)

    train_wgangp(opt, generator, discriminator, train_dataloader, device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--n_epochs", type=int, default=30,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=20,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64,
                        help="size of each image dimension")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="number of training steps for "
                             "discriminator per iter")
    parser.add_argument("--sample_interval", type=int, default=400,
                        help="interval betwen image samples")
    parser.add_argument("--seed", type=int, default=None,
                        help="value of a random seed")
    parser.add_argument("--dataset_path", type=str, default=r"C:\Users\ovolk\Machine Learning\Datasets\Anomaly Detection\Preprocessed\yeast.csv",
                        help="dataset path")
    opt = parser.parse_args()

    main(opt)
