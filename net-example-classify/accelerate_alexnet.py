import torch
# nn块
from torch import nn
# data.DataLoader
from torch.utils.data import DataLoader
# torchvision.datasets.FashionMNIST
import torchvision
# 修改数据集格式
from torchvision import transforms
from accelerate import Accelerator


# -----------参数-----------
accelator = Accelerator()
device = accelator.device
print(device)
batch_size = 128
lr = 3e-2
num_epochs=10


# 列表
trans = [
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
]
# 转化列表为torchvision.transforms.transforms.Compose对象, 这样就能写 transform=trans
trans = transforms.Compose(trans)
mnist_train_totensor = torchvision.datasets.FashionMNIST(
    root="../data",
    train=True,
    download=True,
    transform=trans
)
mnist_test_totensor = torchvision.datasets.FashionMNIST(
    root="../data",
    train=False,
    download=True,
    transform=trans
)


# shuffle, 打乱
# num_workers, 使用4个进程来读取数据
train_iter = DataLoader(
    mnist_train_totensor, batch_size, shuffle=True, num_workers=4)
test_iter = DataLoader(
    mnist_test_totensor, batch_size, shuffle=True, num_workers=4)


net = nn.Sequential(
    # 这里，我们使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    
    nn.Conv2d(384, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    
    nn.Conv2d(384, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    
    nn.Linear(6400, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 10)
).to(device)


X = torch.rand(size=(1, 1, 224, 224), dtype=torch.float32).to(device)
for layer in net:
    X = layer(X)
    print(f'output shape: {layer.__class__.__name__: <15}{X.shape}')


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


net.apply(init_weights)
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()

net, train_iter, test_iter, optimizer = accelator.prepare(net, train_iter, test_iter, optimizer)


def train_loop(train_iter, net, loss, optimizer):
    # 共有几批
    num_batchs = len(train_iter)
    # 总平均loss
    total_train_loss = 0
    for batch, (X, y) in enumerate(train_iter):
        # move to device
        X, y = X.to(device), y.to(device)
        # 该批的推断结果
        y_hat = net(X)
        
        train_loss = loss(y_hat, y)
        total_train_loss += train_loss.item()

        # Backpropagation
        optimizer.zero_grad()
        # - train_loss.backward()
        accelator.backward(train_loss)
        optimizer.step()

        # --------打印进度        
        print(f"\r[{batch+1:>8d}/{num_batchs:>8d}]  ", end='')
    
    return total_train_loss / num_batchs

# ---------训练
def main():
    for epoch in range(num_epochs):
        total_train_loss = train_loop(train_iter, net, loss, optimizer)
        print(f'epoch {epoch + 1}, total_train_loss {total_train_loss:f}')