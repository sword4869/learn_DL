import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        values = {"train_loss": loss}
        self.log_dict(values, prog_bar=True)
        # <<< return loss
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        # <<< dont return loss, instead log loss
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        # <<< dont return loss, instead log loss
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


transform = transforms.ToTensor()
train_set = datasets.MNIST(os.getcwd(), download=True, train=True, transform=transform)
# split the train set into two
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size
train_set, valid_set = random_split(train_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(42))
test_set = datasets.MNIST(os.getcwd(), download=True, train=False, transform=transform)
train_loader = DataLoader(train_set, num_workers=16)
valid_loader = DataLoader(valid_set, num_workers=16)
test_loader = DataLoader(test_set, num_workers=16)

autoencoder = LitAutoEncoder(Encoder(), Decoder())

early_stopping = EarlyStopping(monitor="val_loss", verbose=True)
trainer = pl.Trainer(
    devices=1,
    num_nodes=1,
    callbacks=[early_stopping],
)
# trian the model
trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=valid_loader)
# test the model
trainer.test(model=autoencoder, dataloaders=test_loader)