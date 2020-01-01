import os
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms


# 将图片转化为tensor格式并归一化
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


class MyDataset(data.Dataset):
    def __init__(self,root):
        self.transform = data_transforms
        self.list = []

        # 遍历文件名
        for filenames in os.listdir(root):
            # 获取文件路径
            x = os.path.join(root,filenames)
            # 文件名按照.分割，取前半部分
            ys = filenames.split('.')
            # 对文件名进行one-hot处理
            y = self.one_hot(ys[0])
            self.list.append([x,np.array(y)])


    def __len__(self):
        # 返回数据集长度
        return len(self.list)


    def __getitem__(self, index):
        # 返回图片路径和标签
        img_path,label = self.list[index]
        # 打开图片
        img = Image.open(img_path)
        # 将图片转化成tensor格式并归一化
        img = self.transform(img)
        # 将标签也转化成tensor格式
        label = torch.from_numpy(label)

        return img,label


    def one_hot(self,x):
        # 每个验证码有4个文字，每个文字由62个可能，因此标签的形状为（4,62）
        z = np.zeros(shape=[4,62])
        li1 = [chr(i) for i in range(48, 58)]
        li2 = [chr(i) for i in range(65, 91)]
        li3 = [chr(i) for i in range(97, 123)]
        li = li1 + li2 + li3
        for i in range(4):
            index = int(li.index(x[i]))
            z[i][index] += 1
        return z


# if __name__ == '__main__':
#     mydata = Mydataset('data')
#     dataloader = data.DataLoader(dataset=mydata,batch_size=100,shuffle=True)
#     for i,(x,y) in enumerate(dataloader):
#         print(i)
#         print(x.size())
#         print(y.size())