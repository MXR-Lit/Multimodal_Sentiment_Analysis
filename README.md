# 多模态情感分析

This is the official repository of ICML 2022 paper *Finding Global Homophily in Graph Neural Networks When Meeting Heterophily*.

## Setup

这个实现是基于 Python3 的。要运行这段代码，你需要以下依赖项：

- gensim==4.3.1
- Pillow==9.2.0
- torch==1.13.1
- torchvision==0.14.1

你可以简单地运行以下命令来安装这些依赖项：

```
pip install -r requirements.txt
```



## Repository structure

代码库结构

```
- data/                    # 存储数据集的目录
- train.txt                # 训练集数据
- test_without_label.txt   # 测试集数据
- models/                  # 存储训练好的模型
- README.md                # 项目说明文件
- requirements.txt         # 依赖项清单
- main.py                  # 进行消融实验
- test.py                  # 训练模型的脚本
- predict.py               # 预测模型的脚本
- word2vec.py              # 训练词向量模型的脚本
```



## Run 

训练word2vec模型保存

```bash
python word2vec.py
```

训练模型并评估，最终保存在验证集上准确率最高的模型

```bash
python test.py
```

预测

```bash
python predict.py
```

进行消融实验

```bash
python main.py
```

