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
import math
import pickle
from scipy.stats import multivariate_normal

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
parser.add_argument("--save-model", default="gvae/gvae_mnist", type=str,
        help="The file containing already trained model.")
parser.add_argument("--save-image", default="gvae/gvae_mnist", type=str,
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
    train_loader2 = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=60000, **kwargs)

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

        self.z_size = 20

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, self.z_size)
        self.fc22 = nn.Linear(400, self.z_size)

        self.fc3 = nn.Linear(self.encoder_size(), 400)
        self.fc4 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def encoder_size(self):
        return self.z_size

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize_normal(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        eps = eps.mul(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def sampleAndDecode(self, mu, logvar):
        z = self.reparametrize_normal(mu, logvar)
        return self.decode(z), mu, logvar

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        return self.sampleAndDecode(mu, logvar)

model = VAE()

reconstruction_function = nn.MSELoss()
reconstruction_function.size_average = False

mu_clusters = []
sigma_clusters = []

def loss_function(recon_xs, x, mu, logvar, y_class, epoch):
    BCE = 0
    for recon_x in recon_xs:
        BCE += reconstruction_function(recon_x, x)
    BCE /= len(recon_xs)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # 0.5 * sum ((mu - mu_clusters) * 1/sigma_clusters * (mu - mu-clusters) + 1/sigma_clusters * logVar.exp() + logVar - log det sigma_clusters)
    y_class = torch.from_numpy(y_class).type(torch.FloatTensor)
    mu2 = Variable(y_class.matmul(mu_clusters))
    sigma2 = Variable(y_class.matmul(sigma_clusters))
    KLD_element = mu.sub(mu2).pow(2).div(sigma2)
    KLD_element = KLD_element.add_(logvar.exp().div(sigma2)).mul_(-1).add_(logvar).sub_(sigma2.log()) * max(epoch, (2.*model.z_size)) / (2.*model.z_size)

    # KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return BCE + KLD


optimizer = optim.Adam(model.parameters(), lr=1e-2)

def train(epoch):
    # update_clusters(epoch)
    for _ in range(1):
        train_vae(epoch)
    # return initialize()
    return update_clusters(epoch)

def update_clusters(epoch):
    model.eval()
    mu_tot = []
    sigma_tot = []
    for data, z_class in train_loader2:
        y_class = np.eye(10)[z_class.numpy()]
        data = Variable(data)
        recon_batch, mu, logvar = model(data)
        mu_data = mu.data.numpy()
        sigma_data = logvar.exp().data.numpy()
        for c in range(10):
            a = z_class.numpy() == c
            if True not in a:
                continue
            mu_data_2 = (mu_data[a,:])
            sigma_data_2 = sigma_data[a,:]
            mu = np.sum(mu_data_2, axis=0) / len(y_class)
            sigma = np.sum(sigma_data_2, axis=0) / len(y_class)
            sigma = sigma + np.sum(np.square(mu_data - mu), axis=0) / len(y_class)
            mu_tot.append(mu)
            sigma_tot.append(sigma)
        break
    # _, sigma_tot = initialize()
    return torch.from_numpy(np.array(mu_tot)).type(torch.FloatTensor), torch.from_numpy(np.array(sigma_tot)).type(torch.FloatTensor)
    # return torch.from_numpy(np.array(mu_tot)).type(torch.FloatTensor), sigma_tot

def train_vae(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, y_class) in enumerate(train_loader):
        y_class = np.eye(10)[y_class.numpy()]

        data = Variable(data)
        optimizer.zero_grad()

        total_batch = []
        recon_batch, mu, logvar = model(data)
        total_batch.append(recon_batch)
        for _ in range(args.num_samples - 1):
            recon_batch, _, _ = model.sampleAndDecode(mu, logvar)
            total_batch.append(recon_batch)

        loss = loss_function(total_batch, data, mu, logvar, y_class, epoch)

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

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    imgs = []
    for i in range(10):
        mu_cluster = mu_clusters[i]
        sigma_cluster = sigma_clusters[i]

        model.eval()
        x = model.decode(z)

        imgFile = np.resize((x.data).cpu().numpy(), (28,28))
        imgs.append(imgFile)

    imgFile = stack(imgs)
    imgFile = imgFile * 255 / np.max(imgFile)
    imgFileName = args.save_image + "_" + str(epoch) + ".png"
    cv.imwrite(imgFileName, imgFile)

    for batch_idx, (data_t, z_class) in enumerate(test_loader):
        y_class = np.eye(10)[z_class.numpy()]

        data = Variable(data_t, volatile=True)
        recon_batch, mu, logvar = model(data)

        test_loss += loss_function([recon_batch], data, mu, logvar, y_class, epoch)
        mu = mu.data.numpy()
        logvar = logvar.data.numpy()
        for i in range(len(z_class)):
            mu1, logvar1 = mu[i], logvar[i]
            max_index = 0
            min_distance = 0
            for j in range(10):
                mu2 = mu_clusters[j]
                sigma2 = sigma_clusters[j]
                distance = kl_divergence(mu1, logvar1, mu2, sigma2)
                # distance = normal_pdf(mu1, mu2, sigma2)
                if j == 0 or distance < min_distance:
                    min_distance = distance
                    max_index = j
            # print(max_index)
            if max_index == z_class[i]:
                correct += 1
            total += 1


    test_loss /= len(test_loader.dataset)
    print('Correct: ' + str(correct))
    print('Total: ' + str(total))
    print('====> Test set loss: ' + str(test_loss.data.cpu().numpy()[0]))

def normal_pdf(mu1, mu2, sigma2):
    mu2 = mu2.numpy()
    sigma2 = sigma2.numpy()
    # print(np.eye(model.z_size) * sigma2)
    # print(mu2)
    # print(mu1)
    # a = multivariate_normal(mean=mu2, cov=np.eye(model.z_size) * sigma2)
    # return a(mu1)

    size = model.z_size
    sigma = np.eye(model.z_size) * sigma2
    det = np.linalg.det(sigma)
    norm_const = - np.log(math.pow(det,1.0/2) )
    # print(norm_const)
    x_mu = (mu1 - mu2)
    inv = np.linalg.inv(sigma)
    # print(x_mu.size())
    # print(inv.size())
    # print(np.dot(x_mu, np.dot(inv, x_mu.T)))
    result = -0.5 *np.dot(x_mu, np.dot(inv, x_mu.T))
    return - norm_const - result

def kl_divergence(mu1, logsigma1, mu2, sigma2):
    # 0.5 * sum ((mu - mu_clusters) * 1/sigma_clusters * (mu - mu-clusters) 
    # + 1/sigma_clusters * logVar.exp() + logVar - log det sigma_clusters)
    # mu1 = mu1
    # logsigma1 = logsigma1

    mu2 = mu2.numpy()
    sigma2 = sigma2.numpy()
    return 0.5*np.sum(np.square(mu2 - mu1) / sigma2 + np.exp(logsigma1) / sigma2 + np.log(sigma2) - logsigma1)

def initialize():
    mu = []
    sigma = []
    for i in range(10):
        mu.append([0] * model.z_size)
        # mu[i][0] = i
        sigma.append([1] * model.z_size)
    return torch.from_numpy(np.array(mu)).type(torch.FloatTensor), torch.from_numpy(np.array(sigma)).type(torch.FloatTensor)

def stack(ra):
    num_per_row = int(np.sqrt(len(ra)))
    rows = [np.concatenate(tuple(ra[i* num_per_row : i*num_per_row + num_per_row]), axis=1) 
            for i in range(num_per_row)]
    img = np.concatenate(tuple(rows), axis=0)
    return img

# zs = []
# for _ in range(args.sample_size ** 2):
#     z = torch.FloatTensor(1,model.z_size).normal_()
#     z = Variable(z)
#     zs.append(z)

if args.load_model:
    model = torch.load(args.load_model)
    mu_clusters = pickle.load(open(args.load_model + '_mu', 'rb'))
    sigma_clusters = pickle.load(open(args.load_model + '_sigma', 'rb'))
    # load mu and sigma from pickle file
else:
    mu_clusters, sigma_clusters = initialize()

for epoch in range(1, args.epochs + 1):
    # if "eval" in args.mode and epoch % 1 == 0:
        # test(epoch)
    if "train" in args.mode:
        mu_clusters, sigma_clusters = train(epoch)
    if "eval" in args.mode and epoch % 1 == 0:
        test(epoch)

    if epoch % args.save_interval == 0:
        torch.save(model, args.save_model + "_" + str(epoch))
        # save mu and sigma to pickle file

torch.save(model, args.save_model + "_" + str(epoch))
pickle.dump(mu_clusters, open(args.save_model + "_" + str(epoch) + "_mu", 'wb'))
pickle.dump(sigma_clusters, open(args.save_model + "_" + str(epoch) + "_sigma", 'wb'))
# save mu and sigma to pickle file
