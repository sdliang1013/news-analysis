# 全部流程代码

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils import data
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast

CACHE_DIR = 'D:/data/huggingface/cache'
DS_DIR = 'D:/data/huggingface/datasets'
MODEL_DIR = 'D:/data/huggingface/models'

model_path = f'{MODEL_DIR}/bert-base-chinese'
token_length = 256
hidden_size = 768

labels_dict = {
    "时尚": 0,
    "汽车": 1,
    "财经": 2,
    "科技": 3,
    "健康": 4,
    "体育": 5,
    "教育": 6,
    "文化": 7,
    "军事": 8,
    "娱乐": 9,
}
num_labels = len(labels_dict)
col_label = 'classify'
col_content = 'content'


class Dataset(data.Dataset):
    def __init__(self, df: pd.DataFrame):
        # 标签
        self.labels = [np.int64(labels_dict[label]) for label in df[col_label]]
        # 文本向量化
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=token_length,
                                truncation=True,
                                return_tensors="pt")
                      for text in df[col_content]]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        ary_texts = self.get_texts(idx)
        ary_label = self.get_labels(idx)
        return ary_texts, ary_label


# 构建模型
class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_labels)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

    def save_model(self, output_dir: str):
        self.bert.save_pretrained(output_dir)


# 从上面的代码可以看出，BERT Classifier 模型输出了两个变量：
# 1. 在上面的代码中命名的第一个变量_包含sequence中所有 token 的 Embedding 向量层。
# 2. 命名的第二个变量pooled_output包含 [CLS] token 的 Embedding 向量。对于文本分类任务，使用这个 Embedding 作为分类器的输入就足够了。
# 然后将pooled_output变量传递到具有ReLU激活函数的线性层。在线性层中输出一个维度大小为 5 的向量，每个向量对应于标签类别（运动、商业、政治、 娱乐和科技）。


def train(model: nn.Module, train_data: pd.DataFrame, val_data: pd.DataFrame, learning_rate: float, epochs: int):
    # 通过Dataset类获取训练和验证集
    ds_train, ds_val = Dataset(train_data), Dataset(val_data)
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_loader = data.DataLoader(ds_train, batch_size=100, shuffle=True)
    val_loader = data.DataLoader(ds_val, batch_size=100)
    # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    # 开始进入训练循环
    for epoch_num in range(epochs):
        # 定义两个变量，用于存储训练集的准确率和损失
        total_acc_train = 0
        total_loss_train = 0
        # 进度条函数tqdm
        for train_input, train_label in tqdm(train_loader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            # 通过模型得到输出
            output = model(input_id, mask)
            # 计算损失
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            # 计算精度
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            # 模型更新
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        # ------ 验证模型 -----------
        # 定义两个变量，用于存储验证集的准确率和损失
        total_acc_val = 0
        total_loss_val = 0
        # 不需要计算梯度
        with torch.no_grad():
            # 循环获取数据集，并用训练好的模型进行验证
            for val_input, val_label in val_loader:
                # 如果有GPU，则使用GPU，接下来的操作同训练
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_data): .3f} 
              | Train Accuracy: {total_acc_train / len(train_data): .3f} 
              | Val Loss: {total_loss_val / len(val_data): .3f} 
              | Val Accuracy: {total_acc_val / len(val_data): .3f}''')

    # 我们对模型进行了 5 个 epoch 的训练，我们使用 Adam 作为优化器，而学习率设置为1e-6。


# 测试模型
def evaluate(model: nn.Module, test_data: pd.DataFrame):
    ds_test = Dataset(test_data)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=100)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_loader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')


if __name__ == '__main__':
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertClassifier()
    # 因为本案例中是处理多类分类问题，则使用分类交叉熵作为我们的损失函数。
    EPOCHS = 5
    LR = 1e-6
    # 数据集准备
    # 拆分训练集、验证集和测试集 8:1:1
    news_texts = pd.read_csv(f'{DS_DIR}/news/sogou-news.txt.csv')
    # bbc_text_df.head()
    news_df = pd.DataFrame(news_texts)
    np.random.seed(112)
    df_train, df_val, df_test = np.split(news_df.sample(frac=1, random_state=42),
                                         [int(0.8 * len(news_df)), int(0.9 * len(news_df))])
    print(len(df_train), len(df_val), len(df_test))
    # 训练
    train(model, df_train, df_val, LR, EPOCHS)
    # 用测试数据集进行测试
    evaluate(model, df_test)
    # 保存
    model.save_model(f'{MODEL_DIR}/tuning/news/')
