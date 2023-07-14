import torch
import torch.nn as nn
import re
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec


def get_text_vectors(batch_text):
    word2vec_model = Word2Vec.load('word2vec_model.bin')
    word_vectors=[]
    max_length = 30

    for text in batch_text:
        # 分割文本为单词
        words = text.split()
        if len(words) < max_length:  # 补齐文本序列
            words += [''] * (max_length - len(words))

        # 将每个单词转换为词向量
        vectors = []
        for word in words:
            if word in word2vec_model.wv:
                vector = torch.from_numpy(word2vec_model.wv[word].copy())
                vectors.append(vector)
        # 对于未登录词，可以选择忽略或设置为全零向量
        # 将词向量组合成一个矩阵
        if vectors:
            text_vectors = torch.stack(vectors)
        else:
            # 如果文本中没有有效的单词，则设置该文本的向量为全零向量
            text_vectors = torch.zeros((max_length, word2vec_model.vector_size))
        output_tensor=torch.zeros((30,100))
        for i,tensor in enumerate(text_vectors):
            rows=tensor.size(0)
            output_tensor[:rows,:]=tensor
        #print(output_tensor.shape)
        # 将文本向量添加到列表中
        word_vectors.append(output_tensor)

    # 将列表转换为张量对象
    text_tensor = torch.stack(word_vectors)
    #print(text_tensor.shape)  # 输出张量的形状
    return text_tensor

# 定义文本处理部分的模型
class TextModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TextModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.rnn(x)
        out = self.fc(h_n[-1])
        return out

class ImageModel(nn.Module):#LeNet
    def __init__(self, num_classes):
        super(ImageModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # 输入通道数为1，输出通道数为6，卷积核大小为5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 输入通道数为6，输出通道数为16，卷积核大小为5x5
        self.fc1 = nn.Linear(59536, 120)  # 全连接层，输入维度为16x4x4，输出维度为120
        self.fc2 = nn.Linear(120, 84)  # 全连接层，输入维度为120，输出维度为84

        self.fc = nn.Linear(84, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))  # 卷积层1后接ReLU激活函数
        x = nn.functional.max_pool2d(x, 2)  # 最大池化层，池化窗口大小为2x2
        x = nn.functional.relu(self.conv2(x))  # 卷积层2后接ReLU激活函数
        x = nn.functional.max_pool2d(x, 2)  # 最大池化层，池化窗口大小为2x2
        x = x.view(x.size(0), -1)  # 将特征图展平成一维向量
        #print(x.shape)
        x = nn.functional.relu(self.fc1(x))  # 全连接层1后接ReLU激活函数
        x = nn.functional.relu(self.fc2(x))  # 全连接层2后接ReLU激活函数

        out = self.fc(x)
        return out


# 定义整体的多模态情感分析模型
class MultimodalModel(nn.Module):
    def __init__(self, text_input_size, image_input_size, hidden_size, num_classes):
        super(MultimodalModel, self).__init__()
        self.text_model = TextModel(text_input_size, hidden_size, num_classes)
        self.image_model = ImageModel(num_classes)
        self.fc = nn.Linear(num_classes * 2, num_classes)

    def forward(self, text_x, image_x):
        text_out = self.text_model(text_x)
        image_out = self.image_model(image_x)
        out = torch.cat((text_out, image_out), dim=1)
        out = self.fc(out)
        return out

class MultimodalDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data_dict = data_dict
        self.transform = transform
        self.keys = list(data_dict.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        # 获取文本、图片和标签
        label, text, image = self.data_dict[key]
        # 图片预处理
        if self.transform:
            image = self.transform(image)

        return text, image, label


if __name__ == '__main__':
    mydata = {}
    with open('test_without_label.txt', 'r') as file:
        next(file)
        for line in file:
            line = line.strip()  # 去除行尾的换行符和空格
            guid, tag = line.split(',')  # 使用逗号进行分割
            mydata[guid] = [tag]  # 将标签信息添加到字典中
    file.close()
    for key in mydata:
        txt_filepath = './data/' + key + '.txt'
        with open(txt_filepath, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read().strip()
            cleaned_text = re.sub(r'[^\w\s]', '', text)  # 移除非单词字符和空白字符
            lowercase_text = cleaned_text.lower()
            mydata[key].append(lowercase_text)
        f.close()
        img_filepath = './data/' + key + '.jpg'
        image = Image.open(img_filepath)
        # 在此处进行所需的图像预处理操作，例如调整大小、裁剪、色彩转换等
        new_size = (1024, 1024)  # 新的目标尺寸为 1024x1024
        resized_image = image.resize(new_size)
        mydata[key].append(resized_image)
    # print(mydata)
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),  # LeNet
        transforms.ToTensor(),
    ])

    text_input_size = 100  # 文本输入的特征大小
    image_input_size = 3  # 图片输入的特征大小
    hidden_size = 128  # 隐藏层大小
    num_classes = 3  # 类别数
    label_mapping = {'positive': 0, 'neutral': 1, 'negative': 2}

    model = MultimodalModel(text_input_size, image_input_size, hidden_size, num_classes)
    dataset = MultimodalDataset(mydata, image_transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    model.load_state_dict(torch.load('multimodel61625.bin'))

    model.eval()
    all_labels=[]
    with torch.no_grad():

        for batch_text, batch_image, batch_label in dataloader:
            text_vectors = get_text_vectors(batch_text)
            text_outputs = model.text_model(text_vectors)
            image_outputs = model.image_model(batch_image)
            outputs = torch.cat((text_outputs, image_outputs), dim=1)
            logits = model.fc(outputs)
            #print(outputs)

            predicted = (torch.softmax(logits.data, dim=1))
            predicted = torch.argmax(predicted,dim=1)
            #print(predicted)
            labels = [key for value in predicted.numpy() for key, val in label_mapping.items() if val == value]
            #print(labels)
            #print(batch_text)
            all_labels.append(labels)
    mylabels=[]
    for labels in all_labels:
        for label in labels:
            mylabels.append(label)
    i=0
    for key in mydata:
        #print(mydata[key][0])
        mydata[key][0]=mylabels[i]
        i=i+1
        #print(mydata[key])
    #print(mydata)

    result = "guid,tag\n"
    for key, value in mydata.items():
        tag=value[0]
        result+= f"{key},{tag}\n"

    with open("result.txt","w") as file:
        file.write(result)
    file.close()
