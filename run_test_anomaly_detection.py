import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

from anomaly_detection import anomaly_detection
from constants import TARGET_COL
from dataset import TabularDataset
from split import split_data_normal_train_mixed_val
from wgan import Generator, Discriminator, Encoder


def main(opt):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv(opt.dataset_path)
    train_frac = 0.5
    random_state = 1337
    train, val = split_data_normal_train_mixed_val(data, train_frac, random_state=random_state)

    X_train, y_train = train.drop(columns=[TARGET_COL]), train[TARGET_COL]
    X_val, y_val = val.drop(columns=[TARGET_COL]), val[TARGET_COL]

    input_shape = X_train.shape[1]

    test_dataset = TabularDataset(X_val, y_val)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    generator = Generator(opt, input_shape)
    discriminator = Discriminator(opt, input_shape)
    encoder = Encoder(opt, input_shape)

    anomaly_detection(opt, generator, discriminator, encoder,
                           test_dataloader, device)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1,
                        help="number of image channels")
    parser.add_argument("--training_label", type=int, default=0,
                        help="label for normal images")
    parser.add_argument("--split_rate", type=float, default=0.8,
                        help="rate of split for normal training data")
    parser.add_argument("--dataset_path", type=str, default=r"C:\Users\ovolk\Machine Learning\Datasets\Anomaly Detection\Preprocessed\wine.csv",
                        help="dataset path")
    opt = parser.parse_args()

    main(opt)