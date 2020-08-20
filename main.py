"""
pytorch

=======


"""

import pandas as pd
import numpy as np
import torch
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as torch_optim

# from torchsummary import summary
from tensorboardX import SummaryWriter

writer = SummaryWriter('./summary')


# override Dataset from torch.utils.data
class CompanyDataset(Dataset):
    def __init__(self, X, Y):
        self.x = X.copy().astype(np.float32)
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# architecture of the neural network
class FlagModel(nn.Module):
    def __init__(self, inp, out):
        super().__init__()

        self.lin1 = nn.Linear(inp, 200)
        self.lin2 = nn.Linear(200, 70)
        self.lin3 = nn.Linear(70, out)

        self.bn2 = nn.BatchNorm1d(200)
        self.bn3 = nn.BatchNorm1d(70)

        self.drops = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.drops(x)

        x = self.bn2(x)
        x = F.relu(self.lin2(x))
        x = self.drops(x)
        x = self.bn3(x)
        x = self.lin3(x)
        #         x=torch.sigmoid(x)
        return x


class Average(object):

    def __init__(self):
        self.sum = 0
        self.count = 0

    def __str__(self):
        return '{:.6f}'.format(self.average)

    @property
    def average(self):
        return self.sum / self.count

    def update(self, value, number):
        self.sum += value * number
        self.count += number


class Accuracy(object):

    def __init__(self):
        self.correct = 0
        self.count = 0

    def __str__(self):
        return '{:.2f}%'.format(self.accuracy * 100)

    @property
    def accuracy(self):
        return self.correct / self.count

    def update(self, output, target):
        with torch.no_grad():
            pred = output.argmax(dim=1)
            correct = pred.eq(target).sum().item()

        self.correct += correct
        self.count += output.size(0)


class Trainer(object):

    def __init__(self, model, optimizer, train_loader, test_loader, device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def fit(self, epochs, optimizer):
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train()
            test_loss, test_acc = self.evaluate()

            print(
                'Epoch: {}/{},'.format(epoch, epochs),
                'train loss: {}, train acc: {},'.format(train_loss, train_acc),
                'test loss: {}, test acc: {}.'.format(test_loss, test_acc),
            )

            writer.add_scalars('train_loss', {optimizer: train_loss.average}, epoch)
            writer.add_scalars('test_loss', {optimizer: test_loss.average}, epoch)
            writer.add_scalars('train_acc', {optimizer: train_acc.accuracy}, epoch)
            writer.add_scalars('test_acc', {optimizer: test_acc.accuracy}, epoch)

    def train(self):
        self.model.train()

        train_loss = Average()
        train_acc = Accuracy()

        for data, target in self.train_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            target = target.long()

            output = self.model(data)
            loss = F.cross_entropy(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), data.size(0))
            train_acc.update(output, target)

        return train_loss, train_acc

    def evaluate(self):
        self.model.eval()

        test_loss = Average()
        test_acc = Accuracy()

        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                target = target.long()

                output = self.model(data)
                loss = F.cross_entropy(output, target)

                test_loss.update(loss.item(), data.size(0))
                test_acc.update(output, target)

        return test_loss, test_acc


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = pd.read_csv(
        args.path)  # a DataFrame without nan,including x and y which named 'flag'=================================
    train_X, val_X, train_y, val_y = train_test_split(
        data.drop(columns='flag').values  # numpy array stands for x
        , data['flag'].values  # numpy array stands for y
        , test_size=0.30
        , random_state=0
        , )
    model = FlagModel(data.shape[1] - 1, data['flag'].unique().shape[0])
    model.to(device)

    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2),
                                     weight_decay=args.weight_decay)

    # train_X, val_X ,train_y, val_y are all numpy array
    train_ds = CompanyDataset(train_X, train_y)
    valid_ds = CompanyDataset(val_X, val_y)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=True)

    trainer = Trainer(model, optimizer, train_loader, test_loader, device)
    trainer.fit(args.epochs, args.optimizer)
    torch.save(model.state_dict(), "neural.npy")


#     model.load_state_dict(torch.load("neural.npy"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='beta1 for adam')
    parser.add_argument('--beta2', default=0.999, type=float,
                        help='beta2 for adam')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup steps for adam')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    args = parser.parse_args()

    print(args)
    run(args)


if __name__ == '__main__':
    main()
    writer.close()
