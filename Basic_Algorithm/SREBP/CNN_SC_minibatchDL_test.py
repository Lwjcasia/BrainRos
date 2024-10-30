import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from CNN_0402 import CNN_MNIST
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import datasets
import seaborn as sns


class LinearClassifier(nn.Module):
    def __init__(self, num_dim, num_classes=10):
        super().__init__()
        self.indim = num_dim
        self.outdim = num_classes
        self.fc = nn.Linear(self.indim, self.outdim)

    def forward(self, x):
        x = self.fc(x)
        return x


out_dim = 32
classifier = LinearClassifier(num_dim=out_dim, )
classifier_loss = nn.CrossEntropyLoss()
classifier_optimizer = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.5)


def train_classifier(epoch):

    for data in train_loader:  # 遍历数据集中的每一个batch
        classifier_optimizer.zero_grad()
        images, labels = data  # 保存测试的输入和输出
        images, labels = images.to(device), labels.to(device)
        outputs0 = model(images)
        outputs0 = torch.reshape(outputs0, (batch_size, 32))
        outputs = classifier(outputs0)
        loss = classifier_loss(outputs, labels)
        loss.backward()
        classifier_optimizer.step()


def test2():
    with torch.no_grad():  # 不需要计算梯度
        tally = torch.zeros((10, 10))
        for data in test_loader:  # 遍历数据集中的每一个batch
            images, labels = data  # 保存测试的输入和输出
            images, labels = images.to(device), labels.to(device)
            outputs0 = model(images)  # model = CNN
            outputs0 = torch.reshape(outputs0, (batch_size, 32))
            outputs = classifier(outputs0)  # classifier = 线性分类器
            maxval, predicted = torch.max(outputs.data, dim=1)
            for i in range(len(labels)):
                if abs(maxval[i]) > 1e-3:
                    tally[labels[i], predicted[i]] += 1

    plt.figure()
    sns.heatmap(tally)
    plt.title('SimCLR_batch size=' + str(batch_size) + ' ; dim=' + str(out_dim))
    plt.show()

    accuracies = np.array(tally.diag()) / np.array(tally.sum(dim=1))

    accuracy = (tally.diag().sum() / tally.sum()).item()
    plt.figure()
    plt.bar(np.linspace(start=0, stop=9, num=10), accuracies)
    plt.title(f'Prediction ACC (Average={round(accuracy, 3)} ; sample={tally.sum()})')
    plt.savefig('./test.jpg')
    plt.show()


batch_size = 64

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])
data_path = "D:/ChenFei/SPNET/pythonProject/STDP_SOM_program/NMNIST_data/"
train_dataset = datasets.MNIST(root=data_path,
                               train=True,
                               download=True,
                               transform=transform
                               )
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size,
                          drop_last=True
                          )
test_dataset = datasets.MNIST(root=data_path,
                              train=False,
                              download=True,
                              transform=transform
                              )
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size,
                         drop_last=True
                         )

model_path = 'D:/program/model/epoch0_99_CNN32_D15_SimCLR'
model = CNN_MNIST()
model.load_state_dict(torch.load(model_path))
model.eval()
# 在这里加入两行代码，将数据送入GPU中计算！！！
device = torch.device("cpu")
model.to(device)  # 将模型的所有内容放入cuda中
for epoch in range(0, 100):
    print('epoch:', epoch)
    train_classifier(epoch)
    if epoch % 10 == 9:
        test2()
        print('Finished')
