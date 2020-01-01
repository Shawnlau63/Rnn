import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dataset
import torch.utils.data as data

batch = 1000


class RnnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_layer = nn.LSTM(28, 64, 1, batch_first=True)  # LSTM自带激活函数
        self.out_layer = nn.Linear(64, 10)

    def forward(self, x):
        # x的格式为NCHW，要转换为NS(C*H)V(W)(此为横切，竖切时格式为NS(C*W)V(H))
        input = x.view(-1, 28, 28)

        # LSTM中h0和c0可以不定义，默认为0矩阵
        h0 = torch.zeros(1, batch, 64)
        c0 = torch.zeros(1, batch, 64)

        outputs, (hn, cn) = self.rnn_layer(input, (h0, c0))

        output = outputs[:,-1,:]# 只要NSV的最后一个S的数据，output形状为N1V，要转成NV，不用管，pytorch中会直接将空维度降维
        output = self.out_layer(output)

        return output

if __name__ == '__main__':
    train_dataset = dataset.MNIST(root=r'E:\AI\numrec\datasets',train=True,download=False,transform=transforms.ToTensor())
    train_dataloader = data.DataLoader(train_dataset,batch_size=batch,shuffle=True)

    test_dataset = dataset.MNIST(root=r'E:\AI\numrec\datasets',train=False,download=False,transform=transforms.ToTensor())
    test_dataloader = data.DataLoader(train_dataset,batch_size=batch,shuffle=True)

    net = RnnNet()
    opt = torch.optim.Adam(net.parameters())

    loss_fun = nn.CrossEntropyLoss()

    for epoch in range(10000):
        for i,(x,y) in enumerate(train_dataloader):
            output = net(x)
            loss = loss_fun(output,y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 10 == 0:
                print('epoch:{}-{}-loss:{}'.format(epoch, i, loss.item()))
                for xs,ys in test_dataloader:
                    out = net(xs)
                    test_out = torch.argmax(out,dim=1)
                    acc = torch.mean(torch.eq(test_out,ys).float()).item()
                    print('acc',acc)
                    break