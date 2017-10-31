# TODO Write loss function with theta
# TODO Categorical KL Divergence
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
parser.add_argument('--save-interval', type=int, default=1, metavar='N',
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
        return eps.mul(std).add_(mu)

    def sample_gumbel(self, size):
        eps = torch.FloatTensor(size).uniform_(0,1)
        eps = eps.add_(1e-9).log().mul_(-1).add_(1e-9).log().mul_(-1)
        eps = Variable(eps)
        return eps

    def reparametrize_gumbel(self, categorical, hard=True):

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
mu_clusters = []
sigma_clusters = []
theta_clusters = []

def loss_function(recon_xs, x, mu, logvar, y_class, categorical, epoch):
    BCE = 0
    for recon_x in recon_xs:
        # BCE += reconstruction_function(recon_x, x) * max(math.sqrt(model.z_size / float(epoch ** 2)), 1.)
        BCE += reconstruction_function(recon_x, x) * math.sqrt(model.z_size)
    # BCE /= len(recon_xs)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # 0.5 * sum ((mu - mu_clusters) * 1/sigma_clusters * (mu - mu-clusters) + 1/sigma_clusters * logVar.exp() + logVar - log det sigma_clusters)
    y_class = torch.from_numpy(y_class).type(torch.FloatTensor)
    mu2 = Variable(y_class.matmul(mu_clusters))
    sigma2 = Variable(y_class.matmul(sigma_clusters))
    theta2 = Variable(y_class.matmul(theta_clusters))

    KLD_element = mu.sub(mu2).pow(2).div(sigma2)
    # print(logvar)
    # print(torch.sum(theta2.mul(theta2.log()),dim=1).view((args.batch_size,-1)).expand_as(logvar))
    KLD_element = KLD_element.add_(logvar.exp().div(sigma2)).mul_(-1).add_(1).add_(logvar).mul((-torch.sum(theta2.mul(theta2.log()),dim=1).view((logvar.size()[0],-1)).expand_as(logvar))) #.sub_(sigma2.log()) #/ (2*model.z_size)
    KLD = torch.sum(KLD_element).mul_(-0.5) * model.z_size/2.
    # KLD *= torch.sum(theta2.mul(theta2.log()),dim=1)

    # print(KLD)
    # KLD_element2 = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    # KLD2 = torch.sum(KLD_element2).mul_(-0.5) * model.z_size
    # print(KLD2)

    c = F.softmax(categorical)
    
    # KLD += kl_loss(categorical, theta2)
    KLD += torch.sum(c.mul(c.add(1e-9).log() - (theta2.add(1e-9)).log()))
    # KLD += kl_loss(theta2, categorical)



    # do something here

    return BCE + KLD


optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(epoch):
    # update_clusters(epoch)
    for _ in range(2):
        print('train vae')
        train_vae(epoch)
    # return initialize()
    return update_clusters(epoch)

def update_clusters(epoch):
    model.eval()
    mu_tot = []
    sigma_tot = []
    theta_tot = []
    theta_sum = [0] * (model.cat_size)
    for data, z_class in train_loader2:
        y_class = np.eye(10)[z_class.numpy()]
        data = Variable(data)
        recon_batch, mu, logvar, categorical = model(data)
        mu_data = mu.data.numpy()
        sigma_data = logvar.exp().data.numpy()
        theta_data = F.softmax(categorical).data.numpy()

        for c in range(10):
            a = z_class.numpy() == c
            if True not in a:
                continue
            mu_data_2 = (mu_data[a,:])
            sigma_data_2 = sigma_data[a,:]
            theta_data_2 = theta_data[a,:]
            mu = np.sum(mu_data_2, axis=0) / len(y_class)
            sigma = np.sum(sigma_data_2, axis=0) / len(y_class)
            theta = np.sum(theta_data_2, axis=0) / len(y_class)
            # print((theta))
            # theta_sum += theta
            theta_sum[c] = np.sum(theta)
            sigma = sigma + np.sum(np.square(mu_data - mu), axis=0) / len(y_class)
            mu_tot.append(mu)
            sigma_tot.append(sigma)
            theta_tot.append(theta)
        break
    for c in range(10):
        # theta_tot[c] /= theta_sum
        theta_tot[c] /= theta_sum[c]
    # this part basically says don't update sigma, 
    # _, sigma_tot, _ = initialize()

    # update theta tensors
    return convert_list(mu_tot), convert_list(sigma_tot), convert_list(theta_tot)
    # return torch.from_numpy(np.array(mu_tot)).type(torch.FloatTensor), sigma_tot

def train_vae(epoch):
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

        loss = loss_function(total_batch, data, mu, logvar, y_class, categorical, epoch)

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

    for batch_idx, (data_t, z_class) in enumerate(test_loader):
        y_class = np.eye(10)[z_class.numpy()]

        data = Variable(data_t, volatile=True)
        recon_batch, mu, logvar, categorical = model(data)

        test_loss += loss_function([recon_batch], data, mu, logvar, y_class, categorical, epoch)
        mu = mu.data.numpy()
        logvar = logvar.data.numpy()
        # theta = F.softmax(categorical).data.numpy()
        # theta = categorical.data.numpy()
        theta = categorical

        for i in range(len(z_class)):
            mu1, logvar1, theta1 = mu[i], logvar[i], theta[i]
            max_index = 0
            min_distance = 0
            for j in range(10):
                mu2 = mu_clusters[j]
                sigma2 = sigma_clusters[j]
                theta2 = theta_clusters[j]

                distance = gaussian_kl_divergence(mu1, logvar1, mu2, sigma2)
                cat_distance = categorical_kl_divergence(theta1, theta2).data.numpy()[0] * 300
                # print(j, distance, cat_distance)
                distance += cat_distance
                # print(theta1, theta2, categorical_kl_divergence(theta1, theta2).data.numpy()[0])
                # distance += categorical_kl_divergence(theta2, theta1).data.numpy()[0] * model.z_size

                # print(distance)
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

def categorical_kl_divergence(theta1, theta2):
    return F.kl_div(theta1, theta2)

def gaussian_kl_divergence(mu1, logsigma1, mu2, sigma2):
    # 0.5 * sum ((mu - mu_clusters) * 1/sigma_clusters * (mu - mu-clusters) 
    # + 1/sigma_clusters * logVar.exp() + logVar - log det sigma_clusters)
    # mu1 = mu1
    # logsigma1 = logsigma1

    mu2 = mu2.numpy()
    sigma2 = sigma2.numpy()
    return 0.5*np.sum(np.square(mu2 - mu1) / sigma2 + np.exp(logsigma1) / sigma2 + np.log(sigma2) - logsigma1 - model.z_size)

def convert_list(a):
    return torch.from_numpy(np.array(a)).type(torch.FloatTensor)

def initialize():
    mu = []
    sigma = []
    theta = []
    for i in range(10):
        mu.append([0] * model.z_size)
        # mu[i][0] = i
        sigma.append([1] * model.z_size)
        theta.append([1./model.cat_size] * model.cat_size)
    return convert_list(mu), convert_list(sigma), convert_list(theta)

if args.load_model:
    model = torch.load(args.load_model)
    mu_clusters = pickle.load(open(args.load_model + '_mu', 'rb'))
    sigma_clusters = pickle.load(open(args.load_model + '_sigma', 'rb'))
    theta_clusters = pickle.load(open(args.load_model + '_theta', 'rb'))
    # load mu and sigma from pickle file
else:
    mu_clusters, sigma_clusters, theta_clusters = initialize()

def stack(ra):
    num_per_row = int(np.sqrt(len(ra)))
    rows = [np.concatenate(tuple(ra[i* num_per_row : i*num_per_row + num_per_row]), axis=1) 
            for i in range(num_per_row)]
    img = np.concatenate(tuple(rows), axis=0)
    return img

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

for epoch in range(1, args.epochs + 1):
    # if "eval" in args.mode:
    # test(epoch)
    if "train" in args.mode:
        mu_clusters, sigma_clusters, theta_clusters = train(epoch)
        print(theta_clusters)
    if "eval" in args.mode and epoch % 1 == 0:
        test(epoch)

    if epoch % args.save_interval == 0:
        torch.save(model, args.save_model + "_" + str(epoch))
        # save mu and sigma to pickle file

torch.save(model, args.save_model + "_" + str(epoch))
pickle.dump(mu_clusters, open(args.save_model + "_" + str(epoch) + "_mu", 'wb'))
pickle.dump(sigma_clusters, open(args.save_model + "_" + str(epoch) + "_sigma", 'wb'))
pickle.dump(theta_clusters, open(args.save_model + "_" + str(epoch) + "_theta", 'wb'))
# save mu and sigma to pickle file
