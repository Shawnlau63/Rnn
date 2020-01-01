import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dataset

batch = 1000
class RnnNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义RNN网络层,LSTM自带激活
        self.rnn_layer = nn.LSTM(input_size=28, hidden_size=64, num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(64,10)

    def forward(self, x):
        # MNIST数据是NCHW格式，要转换成NS(C*H)V(W)格式
        input = x.view(-1,28,28)

        #LSTM中h0和c0可以不给，默认为0
        h0 = torch.zeros(1,batch,64)
        c0 = torch.zeros(1,batch,64)

        outputs, (h0, c0) = self.rnn_layer(input, (h0, c0))

        output = outputs[:,-1,:]# 只要NSV的最后一个S的数据

        output = self.output_layer(output)
        return output

if __name__ == '__main__':
    train_dataset = dataset.MNIST(root=r'D:\AI\numrec\datasets',train=True,download=False,transform=transforms.ToTensor())
    train_dataloader = data.DataLoader(train_dataset,batch_size=batch,shuffle=True)

    test_dataset = dataset.MNIST(root=r'D:\AI\numrec\datasets',train=False,download=False,transform=transforms.ToTensor())
    test_dataloader = data.DataLoader(train_dataset,batch_size=batch,shuffle=True)

    net = RnnNet()
    opt = torch.optim.Adam(net.parameters())

    loss_fun = nn.CrossEntropyLoss()

    for epoch in range(100000):
        for i, (x,y) in enumerate(test_dataloader):
            output = net(x)
            loss = loss_fun(output,y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 10 == 0:
                print('loss',loss.item())
                for xs,ys in test_dataloader:
                    out = net(xs)
                    test_out = torch.argmax(out,dim=1)
                    acc = torch.mean(torch.eq(test_out,ys).float()).item()
                    print('acc',acc)
                    break