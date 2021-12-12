import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import time
import os
import torch.backends.cudnn as cudnn


import torch.optim as optim 
from torchsummary import summary 
from tensorboardX import SummaryWriter 

os.environ["CUDA_VISIBLE_DEVICES"] = '6'                # GPU Number 
start_time = time.time()
batch_size = 32
learning_rate = 0.01
root_dir = 'drive/app/cifar10/'
default_directory = 'drive/app/torch/save_models'

default_directory1 =  'drive/app/torch/1'
default_directory2 =  'drive/app/torch/2'
default_directory3 =  'drive/app/torch/3'
default_directory4 =  'drive/app/torch/4'
default_directory5 =  'drive/app/torch/5'
default_directory6 =  'drive/app/torch/6'
default_directory7 =  'drive/app/torch/7'

writer = SummaryWriter('runs/graph') 
start_time = time.time()

transform_test = transforms.Compose([
    transforms.ToTensor(),                              # change [0,255] Int value to [0,1] Float value
    transforms.Normalize(mean=(0.4914, 0.4824, 0.4467), # RGB Normalize MEAN
                         std=(0.2471, 0.2436, 0.2616))  # RGB Normalize Standard Deviation
])

test_dataset = datasets.CIFAR10(root=root_dir,
                                train=False,
                                transform=transform_test)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,            # at Test Procedure, Data Shuffle = False
                                          num_workers=0)            # CPU loader number

class BottleNeck(nn.Module): 
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
            )
            
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x  

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

model1 = ResNet(BottleNeck, [3,4,6,3])
model2 = ResNet(BottleNeck, [3,4,6,3])
model3 = ResNet(BottleNeck, [3,4,6,3])
model4 = ResNet(BottleNeck, [3,4,6,3])
model5 = ResNet(BottleNeck, [3,4,6,3])
model6 = ResNet(BottleNeck, [3,4,6,3])
model7 = ResNet(BottleNeck, [3,4,6,3])

criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss()
criterion3 = nn.CrossEntropyLoss()
criterion4 = nn.CrossEntropyLoss()
criterion5 = nn.CrossEntropyLoss()
criterion6 = nn.CrossEntropyLoss()
criterion7 = nn.CrossEntropyLoss()

if torch.cuda.device_count() > 0:
    print("USE", torch.cuda.device_count(), "GPUs!")
    model1 = nn.DataParallel(model1).cuda()
    model2 = nn.DataParallel(model2).cuda()
    model3 = nn.DataParallel(model3).cuda()
    model4 = nn.DataParallel(model4).cuda()
    model5 = nn.DataParallel(model5).cuda()
    model6 = nn.DataParallel(model6).cuda()
    model7 = nn.DataParallel(model7).cuda()
    cudnn.benchmark = True
else:
    print("USE ONLY CPU!")


def test(epoch):
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    model6.eval()
    model7.eval()
    
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if torch.cuda.is_available():
            data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
            data, target = Variable(data), Variable(target)

        outputs1 = model1(data)
        loss1 = criterion1(outputs1, target)
        
        outputs2 = model2(data)
        loss2 = criterion2(outputs2, target)
        
        outputs3 = model3(data)
        loss3 = criterion3(outputs3, target)
        
        outputs4 = model4(data)
        loss4 = criterion4(outputs4, target)
        
        outputs5 = model5(data)
        loss5 = criterion5(outputs5, target)

        outputs6 = model6(data)
        loss6 = criterion6(outputs6, target)
        
        outputs7 = model7(data)
        loss7 = criterion7(outputs7, target)
        
        loss = ( loss1+loss2+loss3+loss4+loss5+loss6+loss7 )/ 7
        outputs = (outputs1 + outputs2 + outputs3 + outputs4+outputs5+outputs6+outputs7)/7

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
        
        writer.add_scalar('test loss', test_loss / (batch_idx + 1), epoch * len(test_loader)+ batch_idx)
        writer.add_scalar('test accuracy', 100. * correct / total, epoch * len(test_loader)+ batch_idx)
        
    print('# TEST : Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
          .format(test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def load_checkpoint(directory, filename='latest.tar.gz'):

    model_filename = os.path.join(directory, filename)
    if os.path.exists(model_filename):
        print("=> loading checkpoint")
        state = torch.load(model_filename)
        return state
    else:
        return None

start_epoch = 0

checkpoint1 = load_checkpoint(default_directory1)
checkpoint2 = load_checkpoint(default_directory2)
checkpoint3 = load_checkpoint(default_directory3)
checkpoint4 = load_checkpoint(default_directory4)
checkpoint5 = load_checkpoint(default_directory5)
checkpoint6 = load_checkpoint(default_directory6)
checkpoint7 = load_checkpoint(default_directory7)

model1.load_state_dict(checkpoint1['state_dict'])
model2.load_state_dict(checkpoint2['state_dict'])
model3.load_state_dict(checkpoint3['state_dict'])
model4.load_state_dict(checkpoint4['state_dict'])
model5.load_state_dict(checkpoint5['state_dict'])
model6.load_state_dict(checkpoint6['state_dict'])
model7.load_state_dict(checkpoint7['state_dict'])

for epoch in range(start_epoch, 1):
    test(epoch)  

now = time.gmtime(time.time() - start_time)
print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))