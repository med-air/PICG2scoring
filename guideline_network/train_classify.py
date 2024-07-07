import torch
import torch.nn.functional as F   # 激励函数的库
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from train_classify_dataset import Prostate_lesionDataset_public
from torch.optim import lr_scheduler

# 定义全局变量
n_epochs = 200     # epoch 的数目
batch_size = 20  # 决定每次读取多少图片

# 定义训练集个测试集，如果找不到数据，就下载
train_data = Prostate_lesionDataset_public(5,"train")
test_data = Prostate_lesionDataset_public(5,"test")
# 创建加载器
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, num_workers = 0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, num_workers = 0)


# 建立一个四层感知机网络
class MLP(torch.nn.Module):   # 继承 torch 的 Module
    def __init__(self):
        super(MLP,self).__init__()    # 
        # 初始化三层神经网络 两个全连接的隐藏层，一个输出层
        self.fc1 = torch.nn.Linear(4096*3,1024)  # 第一个隐含层  
        self.fc2 = torch.nn.Linear(1024,512)  # 第二个隐含层
        self.fc3 = torch.nn.Linear(512,5)   # 输出层
        
    def forward(self,din):
        # 前向传播， 输入值：din, 返回值 dout
        # din = din.view(-1,28*28)       # 将一个多行的Tensor,拼接成一行
        # bat = din.size(0)
        # din = din.view(bat, -1)
        dout = F.relu(self.fc1(din))   # 使用 relu 激活函数
        dout = F.relu(self.fc2(dout))
        dout = F.softmax(self.fc3(dout), dim=1)  # 输出层使用 softmax 激活函数
        # 10个数字实际上是10个类别，输出是概率分布，最后选取概率最大的作为预测值输出
        return dout

# 训练神经网络
def train():
    #定义损失函数和优化器
    device = torch.device('cuda')
    lossfunc = torch.nn.CrossEntropyLoss()
    lossfunc = lossfunc.to(device)
    optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.01)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # 开始训练
    for epoch in range(n_epochs):
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()   # 清空上一步的残余更新参数值
            data = data.to(device).squeeze()
            # data = data.to(device)
            target = target.to(device)
            # print(data.shape)
            # print(target)
            output = model(data)    # 得到预测值
            # print(output)

            loss = lossfunc(output,target)  # 计算两者的误差
            loss.backward()         # 误差反向传播, 计算参数更新值
            optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
            train_loss += loss.item()*data.size(0)
            # _, predicted = torch.max(output.data, 1)
            # total += target.size(0)
            # correct += (predicted == target).sum().item()

    
        train_loss = train_loss / len(train_loader.dataset)
        # scheduler.step()
        print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
        # 每遍历一遍数据集，测试一下准确率
        # print('Accuracy of the network on the test images: %d %%' % (
        # 100 * correct / total))
        test()

# 在数据集上测试神经网络
def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 训练集中不需要反向传播
        for datas, target in test_loader:

            datas = datas.to(device).squeeze()
            target = target.to(device)
            outputs = model(datas)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
    return 100.0 * correct / total

# 声明感知器网络
model = MLP()
device = torch.device('cuda')
model = model.to(device)

if __name__ == '__main__':
    train()

