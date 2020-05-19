# k_means聚类量化
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import cv2
from copy import deepcopy
import math
from torchvision import datasets, transforms





print("python 脚本执行kmeans 量化")

def k_means_cpu(data:np.array, k, iteration = 50):
    #
    print("对数据进行聚类， 返回中心点和与data形状相同的label")
    #输入数据为torch tensor
    org_shape = data.shape
    data = data.reshape(-1, 1)
    
    #聚类
    c = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, iteration, 0.5)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _ , labels, centers = cv2.kmeans(data, k, None, c,  10, flags = flags)

    #形状变化
    labels = labels.reshape(org_shape)
    return torch.from_numpy(centers), torch.from_numpy(labels).int()
def reconstruct_data(data, quant_bit = 4, bias = False, iteration = 50):
    
    k = 2**int(quant_bit)
    #寻找数据的中心点和标签
    data = data.cpu().detach().numpy()
    centers, labels = k_means_cpu(data, k, iteration)
    
    #放入cpu
    centers = centers.reshape(-1)

    #创建新的权重
    new_weight = torch.zeros_like(labels).float()

    #使用聚类结果重构数据
    for cls, cen in enumerate(centers):
        new_weight[labels == cls] = cen.item()
        
    return new_weight


class QuantLinear(nn.Linear):
    def __init__(self,in_features, out_features, bias=False):
        super(QuantLinear, self).__init__( in_features, out_features, bias)
        self.quant_flag = False
        self.quant_bias = bias
        
        #对权重和bis进行量化
    def kmeans_quant(self, bias, quant_bit=4, iteration=50):
        self.quant_flag = True
        self.quant_bias = bias
        self.weight.data = reconstruct_data(self.weight, quant_bit, self.bias, iteration)

class QuantConv2d(nn.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, groups = 1, bias = False):
            super(QuantConv2d, self).__init__( in_channels, out_channels, kernel_size, stride, padding=0, groups = 1, bias = False)
            
            #设置标志位
            self.quant_flag = False
            self.quant_bias = bias
            #对权重和bias进行重构
        def kmeans_quant(self,bias=False, quant_bit=4, iteration=50):
            #标志为置为True
            self.quant_flag = True
            self.quant_bias = bias
            #重构weight
            self.weight.data = reconstruct_data(self.weight, quant_bit, bias, iteration)
            #重构bias
            

            
            
            
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        #
        self.conv1 = QuantConv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = QuantConv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3 = QuantConv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.linear1 = QuantLinear(576, 10)
        
    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = self.relu3(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        return out

    def kmeans_quant(self, bias=False, quantize_bit=4):
        # Should be a less manual way to quantize
        # Leave it for the future
        self.conv1.kmeans_quant(bias, quantize_bit)
        self.conv2.kmeans_quant(bias, quantize_bit)
        self.conv3.kmeans_quant(bias, quantize_bit)
        self.linear1.kmeans_quant(bias, quantize_bit)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total += len(data)
        progress = math.ceil(batch_idx / len(train_loader) * 50)
        print("\rTrain epoch %d: %d/%d, [%-51s] %d%%" %
              (epoch, total, len(train_loader.dataset),
               '-' * progress + '>', progress * 2), end='')

def test():
    pass


def main():
    epochs = 1
    batch_size = 64
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../datasets', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../datasets', train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1000, shuffle=True)

    model = ConvNet().to(device)
    optimizer = torch.optim.Adadelta(model.parameters())
    
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        _, acc = test(model, device, test_loader)
    
    quant_model = deepcopy(model)
    print('=='*10)
    print('2 bits quantization')
    quant_model.kmeans_quant(bias=False, quantize_bit=2)
    _, acc = test(quant_model, device, test_loader)
        
    return model, quant_model

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest: average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct / len(test_loader.dataset)
if __name__ == "__main__":
#     x = torch.randn(10, 10)
#     print("x的值", x)
#     #k_means_cpu(x, 2)
#     print("重构数据", reconstruct_data(x, 2))

    model, quant_model = main()
