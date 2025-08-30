import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from pathlib import Path


DS_PATH = Path(Path(__file__).parent.parent, "samples")

class CIFAR10DataModule(pl.LightningDataModule):
	def __init__(self, data_dir=DS_PATH, batch_size=64, num_workers=2):
		super().__init__()
		self.data_dir = data_dir
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.train_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.RandomHorizontalFlip(),
			transforms.RandomResizedCrop(32, (0.5, 1))
		])
		self.val_transform = transforms.Compose([
			transforms.ToTensor(),
		])

	def setup(self, stage=None):
		self.train_ds = datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=self.train_transform)
		self.val_ds = datasets.CIFAR10(root=self.data_dir, train=False, download=True, transform=self.val_transform)

	def train_dataloader(self):
		return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

	def val_dataloader(self):
		return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

class SimpleCNNNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
			nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Flatten(),
			nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
			nn.Linear(128, 10)
		)

		self.head = nn.LogSoftmax(dim=1)

	def forward(self, x):
		return self.head(self.net(x))

class SimPL(pl.LightningModule):
	def __init__(self):
		super().__init__()
		self.model = SimpleCNNNet()
		self.loss = nn.NLLLoss()

	def forward(self, x):
		return self.model(x)

	def training_step(self, batch, batch_idx):
		x, y = batch
		logits = self(x)
		loss = self.loss(logits, y)
		acc = (logits.argmax(dim=1) == y).float().mean()
		self.log("train_loss", loss)
		self.log("train_acc", acc)
		return loss

	def validation_step(self, batch, batch_idx):
		x, y = batch
		logits = self(x)
		loss = self.loss(logits, y)
		acc = (logits.argmax(dim=1) == y).float().mean()
		self.log("val_loss", loss, prog_bar=True)
		self.log("val_acc", acc, prog_bar=True)
		return {"val_loss": loss, "val_acc": acc}

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=0.0005)

def main():
	data_module = CIFAR10DataModule()
	model = SimPL()
	tb_logger = TensorBoardLogger("lightning_logs", name="cifar10")
	trainer = pl.Trainer(max_epochs=50, logger=tb_logger)
	trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
	main()
