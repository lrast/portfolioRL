import torch
import pytorch_lightning as pl

from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class functionLearner(pl.LightningModule):
    """a simple function learner to check neural network architectures"""
    def __init__(self, target, inRange=[0, 1], numSamples=1024):
        super(functionLearner, self).__init__()
        self.target = target
        self.inRange = inRange
        self.numSamples = numSamples

        hiddenSize = 256

        self.network = nn.Sequential(
                                    nn.Linear(1, hiddenSize),
                                    nn.LayerNorm((hiddenSize,)),
                                    nn.Sigmoid(),
                                    nn.Linear(hiddenSize, hiddenSize),
                                    nn.LayerNorm((hiddenSize,)),
                                    nn.ReLU(),
                                    nn.Linear(hiddenSize, 1),
                                    )

        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.network(x-0.5)

    def training_step(self, batch, batchidx):
        xs, ys = batch
        return self.loss(self.forward(xs), ys)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1E-3)

    def setup(self, stage=None):
        """make dataset"""
        xs = (self.inRange[0] + 
              (self.inRange[1] - self.inRange[0]) * torch.rand(self.numSamples)
              )[:, None]
        ys = self.target(xs)
        self.data = TensorDataset(xs, ys)

    def train_dataloader(self):
        return DataLoader(self.data, batch_size=10, shuffle=True)


class discreteLearner(pl.LightningModule):
    """Constant learner with one-hot encoding of time values"""
    def __init__(self, middleDimension=6, dataSeed=None):
        super(discreteLearner, self).__init__()

        self.dataSeed = dataSeed
        if middleDimension == 0:
            self.network = nn.Sequential( 
                                nn.Embedding(10, 2),
                                )
        else:
            self.network = nn.Sequential(
                                nn.Embedding(10, middleDimension),
                                nn.Linear(middleDimension, 2),
                                )

        self.loss = nn.MSELoss()

    def forward(self, x):
        return torch.abs(self.network(x))

    def training_step(self, batch, batchidx):
        targets = batch[0]
        predictions = self.forward(torch.arange(0, 10))
        predictions = predictions.repeat((targets.shape[0], 1, 1))

        return self.loss(predictions, targets)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1E-3)

    def setup(self, stage=None):
        """Make Dataset"""
        if self.dataSeed:
            torch.manual_seed(self.dataSeed)

        self.targets = torch.distributions.Exponential(0.01).sample((10, 2))
        self.datapoints = 5*torch.randn((1000, 10, 2)) + self.targets

    def train_dataloader(self):
        return DataLoader(TensorDataset(self.datapoints), batch_size=32)
