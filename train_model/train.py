import torch.optim
import torchvision
import torchvision.models as models
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchgen.model import Return
from torchvision import transforms
from model import *

train_data = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True,
                                          transform=transforms.ToTensor())

test_data = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True,
                                         transform=transforms.ToTensor(),)

train_data_size = len(train_data)       #50000
test_data_size = len(test_data)         #10000

# print(train_data_size)
# print(test_data_size)

train_loader = DataLoader(train_data, batch_size=50,shuffle=True)
test_loader = DataLoader(test_data, batch_size=50)

#创建网络模型
train = train()
train = train.cuda()

#损失函数
loss_train = nn.CrossEntropyLoss()
loss_train = loss_train.cuda()

#优化器
learning_rate = 1e-4
optimizer = torch.optim.Adam(train.parameters(), lr=learning_rate,weight_decay=0.0005)


#设置训练参数
total_train_step = 0
#测试次数
total_test_step = 0
#训练轮数
epoch = 100

#添加tensorborad
writer = SummaryWriter("test_CIFAR10_model")

for i in range(epoch):
    print(f"第{i+1}轮训练开始")


    #训练开始
    train.train()
    total_train_loss = 0
    train_accuracy = 0
    for data in train_loader:
        imgs,targets = data
        imgs = imgs.cuda()
        optimizer.zero_grad()
        targets = targets.cuda()
        outputs = train(imgs)
        loss = loss_train(outputs, targets)
        total_train_loss += loss.item()
        #优化器优化模型
        # optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train.parameters(), 1)

        optimizer.step()

        total_train_step += 1
        accuracy = (outputs.argmax(1) == targets).sum()

        train_accuracy += accuracy

        if total_train_step % 100 == 0:
            print(f"训练次数：{total_train_step}，loss:{loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    #测试步骤
    train.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs,targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = train(imgs)

            loss = loss_train(outputs, targets)
            total_test_loss += loss.item()

            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f"整体测试的loss:{total_test_loss}")
    print(f"整体测试正确率：{total_accuracy/test_data_size}")
    writer.add_scalars("loss", {'test_loss':total_test_loss,'train_loss':total_train_loss}, total_test_step)
    writer.add_scalars("accuracy", {'test':total_accuracy/test_data_size,'train':train_accuracy/train_data_size}, total_test_step)
    total_test_step += 1

writer.close()