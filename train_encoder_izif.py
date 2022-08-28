import os
import torch
import torch.nn as nn


def train_encoder_izif(opt, generator, discriminator, encoder,
                       dataloader, device, kappa=1.0):
    generator.load_state_dict(torch.load("results/generator"))
    discriminator.load_state_dict(torch.load("results/discriminator"))

    generator.to(device).eval()
    discriminator.to(device).eval()
    encoder.to(device)

    criterion = nn.MSELoss()

    optimizer_E = torch.optim.Adam(encoder.parameters(),
                                   lr=opt.lr, betas=(opt.b1, opt.b2))


    padding_epoch = len(str(opt.n_epochs))
    padding_i = len(str(len(dataloader)))

    batches_done = 0
    for epoch in range(opt.n_epochs):
        for i, (x, _) in enumerate(dataloader):
            real_x = x.to(device)
            optimizer_E.zero_grad()
            z = encoder(real_x)
            fake_x = generator(z)
            real_features = discriminator.forward_features(real_x)
            fake_features = discriminator.forward_features(fake_x)
            loss_x = criterion(fake_x, real_x)
            loss_features = criterion(fake_features, real_features)
            e_loss = loss_x + kappa * loss_features
            e_loss.backward()
            optimizer_E.step()
            if i % opt.n_critic == 0:
                print(f"[Epoch {epoch:{padding_epoch}}/{opt.n_epochs}] "
                      f"[Batch {i:{padding_i}}/{len(dataloader)}] "
                      f"[E loss: {e_loss.item():3f}]")
                batches_done += opt.n_critic
    torch.save(encoder.state_dict(), "results/encoder")
