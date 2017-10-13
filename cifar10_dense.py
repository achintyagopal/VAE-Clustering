from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import cv2 as cv

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-level', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--sample-size', type=int, default=100, metavar='N',
                    help='number of samples to generate (should be perfect square)')
parser.add_argument('--eval-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument("--load-model", type=str,
        help="The file containing already trained model.")
parser.add_argument("--save-model", default="dense_mnist", type=str,
        help="The file containing already trained model.")
parser.add_argument("--mode", type=str, default="train-eval", choices=["train", "eval", "train-eval"],
                        help="Operating mode: train and/or test.")
args = parser.parse_args()


torch.manual_seed(args.seed)

kwargs = {}

if "train" in args.mode:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

if "eval" in args.mode:
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

if args.mode == "eval":
    if not args.load_model:
        raise ValueError("Need which model to evaluate")
    args.epoch = 1
    args.eval_interval = 1


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'mine1': [32,'M2',32,'M2',32,'M2'],
    'mine2': [32,'M8'],
}


# class VGG(nn.Module):
#     def __init__(self, vgg_name):
#         super(VGG, self).__init__()
#         self.features = self._make_layers(cfg[vgg_name])
#         self.classifier = nn.Linear(512, 10)
#         self.softmax = nn.Softmax()

#     def forward(self, x):
#         out = self.features(x)
#         out = out.view(out.size(0), -1)
#         out = self.softmax(self.classifier(out))
#         return out

#     def _make_layers(self, cfg):
#         layers = []
#         in_channels = 3
#         for x in cfg:
#             if x == 'M2':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             elif x == 'M4':
#                 layers += [nn.MaxPool2d(kernel_size=4, stride=4)]
#             elif x == 'M8':
#                 layers += [nn.MaxPool2d(kernel_size=8, stride=8)]
#             else:
#                 layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                            nn.BatchNorm2d(x),
#                            nn.ReLU(inplace=True)]
#                 in_channels = x
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
#         return nn.Sequential(*layers)

# model = VGG('mine2')

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # self.fc1 = nn.Sequential(
        #     # nc * 32 * 32
        #     nn.Conv2d(3, 32, 4, 4, 1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),

        #     # 32 x 8 x 8
        #     nn.Conv2d(32, 32, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.2, inplace=True)

        #     # 32 x 4 x 4
        # )

        # self.fc2 = nn.Linear(512, 10)

        self.fc1 = nn.Sequential(
            # state size. (ndf) x 32 x 32
            nn.Conv2d(3, 16 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (16*2) x 16 x 16
            nn.Conv2d(16 * 2, 16 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (16*4) x 8 x 8
            nn.Conv2d(16 * 4, 16 * 8, 4, 2, 1, bias=False)
        )

        self.fc2 = nn.Linear(2048, 10)
        # self.fc3 = nn.Linear(512, 10)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def encode(self, x):
        h1 = self.fc1(x)
        h2 = h1.view(-1, 2048)
        return self.softmax(self.fc2(h2))

        # h2 = self.relu(self.fc2(h2))
        # return self.softmax(self.fc3(h2))

    def forward(self, x):
        return self.encode(x)


model = VAE()


def loss_function(y_class, y_pred):
    # categorical loss KL(p||q) + H(q)
    return torch.sum(-Variable(torch.from_numpy(y_class).type(torch.FloatTensor)).mul(y_pred.add(1e-9).log()))
    # return torch.sum(y_pred.mul(y_pred.log())) + torch.sum(-Variable(torch.from_numpy(y_class).type(torch.FloatTensor)).mul(y_pred.log()))
    # return torch.sum(y_pred.mul(y_pred.log().sub_(y_pred.add(1e-9).log())))
    # return torch.sum(y_pred.mul(y_pred.log().sub_(Variable(torch.from_numpy(y_class).type(torch.FloatTensor)).mul(y_pred.log()))))
    # return torch.sum(y_pred.mul(y_pred.log())) + torch.sum(-y_pred.mul(Variable(torch.from_numpy(y_class).type(torch.FloatTensor)).add_(0.5).log()))

optimizer = optim.Adam(model.parameters(), lr=1e-2)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, y_class) in enumerate(train_loader):
        y_class = np.eye(10)[y_class.numpy()]

        data = Variable(data)
        optimizer.zero_grad()
        # repeat model(data) multiple times, mu and logvar won't change, recon_batch will, it's like batch 
        y_pred = model(data)
        loss = loss_function(y_class, y_pred)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if args.log_level == 1:
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, z_class) in enumerate(test_loader):
        y_class = np.eye(10)[z_class.numpy()]


        data = Variable(data, volatile=True)
        y_pred = model(data)
        test_loss += loss_function(y_class, y_pred)
        z = y_pred.data.cpu().numpy()
        for i, row in enumerate(z):
            pred = np.argmax(row)
            if pred == z_class[i]:
                correct += 1
        total += len(z_class)

    test_loss /= len(test_loader.dataset)
    print('Correct: ' + str(correct))
    print('Total: ' + str(total))
    print('====> Test set loss: ' + str(test_loss.data.cpu().numpy()[0]))

def stack(ra):
    num_per_row = int(np.sqrt(len(ra)))
    rows = [np.concatenate(tuple(ra[i* num_per_row : i*num_per_row + num_per_row]), axis=1) 
            for i in range(num_per_row)]
    img = np.concatenate(tuple(rows), axis=0)
    return img

if args.load_model:
    model = torch.load(args.load_model)

for epoch in range(1, args.epochs + 1):
    
    if "train" in args.mode:
        train(epoch)
    if "eval" in args.mode:
        test(epoch)

    if epoch % args.save_interval == 0:
        torch.save(model, args.save_model + "_" + str(epoch))

