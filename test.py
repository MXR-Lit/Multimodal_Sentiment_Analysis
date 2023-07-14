from PIL import Image
from gensim.models import Word2Vec
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import random
import re


def split_dict(dictionary):
    keys = list(dictionary.keys())  # 获取字典的所有键
    random.shuffle(keys)  # 打乱键的顺序

    split_index = int(len(keys) * 0.8)  # 计算分割点索引位置

    dict_1 = {}  # 第一个字典
    dict_2 = {}  # 第二个字典

    for i, key in enumerate(keys):
        if i < split_index:
            dict_1[key] = dictionary[key]
        else:
            dict_2[key] = dictionary[key]

    return dict_1, dict_2


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


# 定义图片处理部分的模型
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

def batch_text_labels_to_tensor(labels):
    label_mapping = {'positive': 0, 'neutral': 1, 'negative': 2}
    label_indices = [label_mapping[label] for label in labels]
    tensor = torch.tensor(label_indices).view(-1, 1)
    return tensor


if __name__ == '__main__':
    mydata={}
    with open('train.txt', 'r') as file:
        next(file)
        for line in file:
            line = line.strip()  # 去除行尾的换行符和空格
            guid, tag = line.split(',')  # 使用逗号进行分割
            mydata[guid] = [tag]  # 将标签信息添加到字典中
    file.close()
    for key in mydata:
        txt_filepath='./data/'+key+'.txt'
        with open(txt_filepath, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read().strip()
            cleaned_text = re.sub(r'[^\w\s]', '', text)  # 移除非单词字符和空白字符
            lowercase_text = cleaned_text.lower()
            mydata[key].append(lowercase_text)
        f.close()
        img_filepath='./data/'+key+'.jpg'
        image = Image.open(img_filepath)
        # 在此处进行所需的图像预处理操作，例如调整大小、裁剪、色彩转换等
        new_size = (1024, 1024)  # 新的目标尺寸为 1024x1024
        resized_image = image.resize(new_size)
        mydata[key].append(resized_image)
    #print(mydata)

    train_data, test_data=split_dict(mydata)
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),#LeNet
        transforms.ToTensor(),
    ])
    train_dataset = MultimodalDataset(train_data, image_transform)
    test_dataset = MultimodalDataset(test_data, image_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    text_input_size = 100  # 文本输入的特征大小
    image_input_size = 3  # 图片输入的特征大小
    hidden_size = 128  # 隐藏层大小
    num_classes = 3  # 类别数

    model=MultimodalModel(text_input_size,image_input_size,hidden_size,num_classes)
    #model.load_state_dict(torch.load('multimodel.bin'))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    num_epochs=30
    accuracy=0
    for epoch in range(num_epochs):
        model.train()
        total_loss=0
        print("train")
        for batch_text, batch_image, batch_label in train_dataloader:
            optimizer.zero_grad()

            #print(batch_image)
            #print(batch_image.shape)

            text_vectors=get_text_vectors(batch_text)
            label_tensors = (batch_text_labels_to_tensor(batch_label))

            text_outputs = model.text_model(text_vectors)
            #image_outputs = model.image_model(batch_image) #AlexNet
            image_outputs = model.image_model(batch_image)    #Lenet
            outputs=torch.cat((text_outputs, image_outputs), dim=1)
            logits = model.fc(outputs)

            #predictions=(torch.argmax(logits,dim=1))
            predictions = torch.softmax(logits, dim=1)

            #print("logits",logits)
            #print("predictions",torch.argmax(predictions,dim=1).view(-1))
            #print("label_tensors",label_tensors.view(-1))
            #print("text_output",torch.argmax(text_outputs,dim=1).view(-1))
            #print("img_output",torch.argmax(image_outputs,dim=1).view(-1))

            loss = criterion(logits, torch.squeeze(label_tensors))
            loss.backward()
            optimizer.step()
            total_loss=total_loss+loss

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, total_loss.item()))

        model.eval()  # 将模型设置为评估模式
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_text, batch_image, batch_label in test_dataloader:
                text_vectors = get_text_vectors(batch_text)
                text_outputs = model.text_model(text_vectors)
                image_outputs = model.image_model(batch_image)
                outputs = torch.cat((text_outputs, image_outputs), dim=1)
                logits = model.fc(outputs)
                #print(outputs)

                #predicted = torch.argmax(logits.data,dim=1)    #60125

                predicted = (torch.softmax(logits.data, dim=1))
                predicted = torch.argmax(predicted,dim=1)
                # 返回每个样本最大值的索引

                label_tensors = (batch_text_labels_to_tensor(batch_label)).view(-1)

                #print(predicted)
                #print(label_tensors)

                total += label_tensors.size(0)  # 计算总样本数
                print("total: ",total)
                correct += (predicted == label_tensors).sum().item()  # 统计预测正确的样本数
                print("correct: ", correct)

        if correct/total>accuracy:
            torch.save(model.state_dict(), 'multimodel.bin')
            accuracy = correct / total  # 计算准确率
        print(f"在测试集上的准确率为: {correct/total:.2%}")

    print(accuracy)