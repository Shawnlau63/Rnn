import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from RNNdemo.MyDatas import MyDataset


img_path = 'data'
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCH = 100
save_path = r'model/test.pkl'

class RnnNet(nn.Module):
    # 定义网络
    def __init__(self):
        super().__init__()
        # 加一层线性层，为了将形状多做一次进行处理
        self.fc1 = nn.Sequential(
            nn.Linear(180, 128),    # [batch_size*120,128]
            nn.BatchNorm1d(num_features=128),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True)
        self.out = nn.Linear(128,62)

    def forward(self, x):
        # 图片格式NCHW(N,3,60,120)先转成(N,3*60,120)再转成NSV(N,120,3*60)，竖着切
        x = x.view(-1, 180, 120).permute(0, 2, 1)   # [batch_size,120,180]
        # 传入全连接层的图像格式为NV结构
        x = x.contiguous().view(-1, 180)    # [batch_size*120,180]
        fc1 = self.fc1(x)   # [batch_size*120,180]
        fc1 = fc1.view(-1, 120, 128)    # [batch_size*120,180]
        lstm, (h_n, h_c) = self.lstm(fc1)   # [batch_size*120,180]
        out = lstm[:, -1, :]

        out = out.view(-1, 1, 128)
        out = out.expand(BATCH_SIZE, 4, 128)    # [batch_size,4,128]
        lstm, (h_n, h_c) = self.lstm(out)   # [batch_size,4,128]
        # .contiguous()是在调用.view()前使用防止报错的方法，若用reshape则可不调用该方法
        y1 = lstm.contiguous().view(-1, 128)    # [batch_size*4,128]
        out = self.out(y1)                      # [batch_size*4,62]
        out = out.view(-1, 4, 62)               # [batch_size,4,62]
        output = F.softmax(out,dim=2)

        return out,output

if __name__ == '__main__':
    net = RnnNet()
    opt = torch.optim.Adam(net.parameters())
    loss_fun = nn.MSELoss()

    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))

    train_data = MyDataset(root='data')
    train_loader = data.DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True,drop_last=True,num_workers=NUM_WORKERS)

    for epoch in range(EPOCH):
        for i, (x, y) in enumerate(train_loader):
            batch_x = x
            batch_y = y.float()

            decoder = net(batch_x)
            out, output = decoder[0], decoder[1]
            loss = loss_fun(out, batch_y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 5 == 0:
                test_y = torch.argmax(y,2).data.numpy()
                pred_y = torch.argmax(output,2).cpu().data.numpy()
                accuracy = np.mean(np.all(pred_y == test_y,axis=1))
                print('epoch:', epoch, '  |  ', 'i:', i, '  |  ', 'loss:', '%.4f' % loss.item(), '  |  ', 'accuracy:', '%.2f%%' % (accuracy * 100))
                print('test_y:',[chr(int(x)) for x in test_y[0]])
                print('pred_y:',[chr(int(x)) for x in pred_y[0]])

        torch.save(net.state_dict(), save_path)