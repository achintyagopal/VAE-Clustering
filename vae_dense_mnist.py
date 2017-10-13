from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import cv2 as cv
import random

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--sample-size', type=int, default=10, metavar='N',
                    help='number of samples to generate (should be perfect square)')
parser.add_argument('--eval-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-interval', type=int, default=100000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument("--load-model", type=str,
        help="The file containing already trained model.")
parser.add_argument("--save-model", default="cvae_dense", type=str,
        help="The file containing already trained model.")
parser.add_argument("--save-image", default="cvae_dense", type=str,
        help="The file containing already trained model.")
parser.add_argument("--temperature", default=1, type=float,
        help="The file containing already trained model.")
parser.add_argument("--mode", type=str, default="train-eval", choices=["train", "eval", "train-eval"],
                        help="Operating mode: train and/or test.")
parser.add_argument("--num-samples", default=1, type=int,
        help="The number of samples to draw from distribution")

args = parser.parse_args()


torch.manual_seed(args.seed)

kwargs = {}

if "train" in args.mode:
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

if "eval" in args.mode:
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

if args.mode == "eval":
    if not args.load_model:
        raise ValueError("Need which model to evaluate")
    args.epoch = 1
    args.eval_interval = 1


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.z_size = 5
        self.cat_size = 10

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, self.z_size)
        self.fc22 = nn.Linear(400, self.z_size)
        self.fc23 = nn.Linear(400, self.cat_size)
        self.fc23b = nn.Linear(self.cat_size, self.cat_size)

        self.fc3 = nn.Linear(self.encoder_size(), 400)
        self.fc4 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def encoder_size(self):
        return self.z_size + self.cat_size

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1), self.fc23(h1)

    def reparametrize_normal(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        eps = eps.mul(std)
        return eps.mul(std).add_(mu)

    def sample_gumbel(self, size):
        eps = torch.FloatTensor(size).uniform_(0,1)
        eps = eps.add_(1e-9).log().mul_(-1).add_(1e-9).log().mul_(-1)
        eps = Variable(eps)
        return eps

    def reparametrize_gumbel(self, categorical, hard=False):

        temperature = args.temperature
        noise = self.sample_gumbel(categorical.size())
        x = (categorical + noise)/temperature
        x = F.softmax(x)

        if hard:
            max_val, _ = torch.max(x, x.dim() - 1, keepdim=True)
            x_hard = x == max_val.expand_as(x)
            tmp  = x_hard.float() - x
            tmp2 = tmp.clone()
            tmp2.detach_()
            x = tmp2 + x

        return x.view_as(categorical)

    def decode(self, z, c):
        c = self.fc23b(c)
        z = torch.cat([z,c], 1)
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def sampleAndDecode(self, mu, logvar, categorical):
        z = self.reparametrize_normal(mu, logvar)
        c = self.reparametrize_gumbel(categorical)
        return self.decode(z, c), mu, logvar, categorical

    def forward(self, x):
        mu, logvar, categorical = self.encode(x.view(-1, 784))
        return self.sampleAndDecode(mu, logvar, categorical)

model = VAE()

reconstruction_function = nn.MSELoss()
reconstruction_function.size_average = False


def loss_function(recon_xs, x, mu, logvar, categorical, y_class,epoch):
    BCE = 0
    for recon_x in recon_xs:
        BCE += 0.01*reconstruction_function(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # print(KLD)
    # add categorical loss function here
    # print(categorical)

    # print(torch.sum(categorical.exp().mul(categorical.exp().mul(model.cat_size).add_(1e-9).log())))
    # KLD += torch.sum(categorical.exp().mul(categorical.exp().mul(model.cat_size).add_(1e-9).log()))
    c = F.softmax(categorical)
    # KLD += torch.sum(c.mul(c.mul(model.cat_size).add_(1e-9).log()))


    # KLD += torch.sum(c.mul(c.log())) + torch.sum(-c.mul(Variable(torch.from_numpy(y_class).type(torch.FloatTensor)).add_(0.5).log()))
    KLD += torch.sum(-Variable(torch.from_numpy(y_class).type(torch.FloatTensor)).mul(c.add(1e-9).log()))


    # KLD += torch.sum(c.mul(ccategorical.mul(model.cat_size).add_(1e-9).log()))
    return BCE + KLD


optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, y_class) in enumerate(train_loader):
        y_class = np.eye(10)[y_class.numpy()]

        data = Variable(data)
        optimizer.zero_grad()

        total_batch = []
        recon_batch, mu, logvar, categorical = model(data)
        total_batch.append(recon_batch)
        for _ in range(args.num_samples - 1):
            recon_batch, _, _, _ = model.sampleAndDecode(mu, logvar, categorical)
            total_batch.append(recon_batch)

        loss = loss_function(total_batch, data, mu, logvar, categorical, y_class,epoch)

        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

zs = []
for _ in range(args.sample_size):
    z = torch.FloatTensor(1,model.z_size).normal_()
    z = Variable(z)
    zs.append(z)

cs = []
for i in range(10):
    c = np.zeros((1,10))
    c[0][i] = 1
    c = torch.from_numpy(c).type(torch.FloatTensor)
    c = Variable(c)
    cs.append(c)

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, z_class) in enumerate(test_loader):
        y_class = np.eye(10)[z_class.numpy()]

        data = Variable(data, volatile=True)
        recon_batch, mu, logvar, y_pred = model(data)

        test_loss += loss_function([recon_batch], data, mu, logvar, y_pred, y_class,epoch)
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

    if epoch % args.eval_interval == 0:
        imgs = []
        for z in zs: 
            for c in cs:
                model.eval()
                x = model.decode(z, c)

                imgFile = np.resize((x.data).cpu().numpy(), (28,28))
                imgs.append(imgFile)

        imgFile = stack(imgs)
        imgFile = imgFile * 255 / np.max(imgFile)
        imgFileName = args.save_image + "_" + str(epoch) + ".png"
        cv.imwrite(imgFileName, imgFile)

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

