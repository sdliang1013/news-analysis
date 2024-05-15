"""test python"""
from typing import Tuple, MutableMapping

import numpy
import torch
from datasets import load_dataset, Dataset
from transformers import BertTokenizer, PreTrainedTokenizer, PreTrainedModel, BertForSequenceClassification, \
    BertTokenizerFast, TrainingArguments, Trainer

BASE_DIR = 'D:/data/huggingface'
CACHE_DIR = f'{BASE_DIR}/cache'
DS_DIR = f'{BASE_DIR}/datasets'
MODEL_DIR = f'{BASE_DIR}/models'
RESULT_DIR = f'{BASE_DIR}/results'
LOG_DIR = f'{BASE_DIR}/logs'
token_length = 128
col_label = 'classify'
col_content = 'content'

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


def classify_label(classify: str) -> int:
    return labels_dict.get(classify, 0)
    # return numpy.array(labels_dict.get(classify, 0))


def train_test_split(ds: Dataset, test_ratio: float) -> Tuple[Dataset, Dataset]:
    ds_split = ds.train_test_split(test_size=test_ratio)
    return ds_split.get('train'), ds_split.get('test')


def conv2input(tokenizer: PreTrainedTokenizer, data: MutableMapping, max_length: int) -> MutableMapping:
    # 内容
    token = tokenizer(text=data.get(col_content), add_special_tokens=True, max_length=max_length,
                      padding='max_length', truncation=True, return_tensors="pt",
                      return_token_type_ids=True, return_attention_mask=True, )
    data.update({
        "input_ids": token.get('input_ids'),
        "attention_mask": token.get('attention_mask'),
        "token_type_ids": token.get('token_type_ids'),
    })
    # 分类
    labels = data.get(col_label, [])
    if labels:
        data.update({
            "label": list(map(classify_label, labels)),
        })
    return data


def tuning_news():
    """
    新闻分类预模型训练
    :return:
    """
    # load model
    model_path = f'{MODEL_DIR}/bert-base-chinese'
    model: PreTrainedModel = BertForSequenceClassification.from_pretrained(model_path,
                                                                           classifier_dropout=0.5,
                                                                           num_labels=num_labels)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    print(model)
    # load data
    # df = pd.read_csv(filepath_or_buffer=f'{DS_DIR}/news/sogou-news.txt.csv')
    ds = load_dataset(path='csv', data_files=[f'{DS_DIR}/news/sogou-news.txt.csv'], cache_dir=CACHE_DIR)
    # 构建输入向量
    ds_train, ds_test = train_test_split(ds=ds.get("train"), test_ratio=0.1)
    ds_train = ds_train.map(function=lambda data: conv2input(tokenizer=tokenizer, data=data, max_length=token_length),
                            batched=True, batch_size=100)
    ds_test = ds_test.map(function=lambda data: conv2input(tokenizer=tokenizer, data=data, max_length=token_length),
                          batched=True, batch_size=100)
    # 选择列
    ds_train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    ds_test.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    print(ds_train)
    print(ds_test)
    # 定义优化器,损失函数
    batch_size = 8
    epochs = 5
    warmup_steps = 100
    weight_decay = 0.01
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir='./logs',
        optim="adamw_torch",  # 修复告警
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
    )
    # 训练
    trainer.train()
    # 评估
    trainer.evaluate()
    # 保存
    trainer.save_model(f'{MODEL_DIR}/tuning/news/')


def predict_news():
    """
    新闻分类模型预测
    :return:
    """
    # load model
    model_path = f'{MODEL_DIR}/tuning/news'
    model: PreTrainedModel = BertForSequenceClassification.from_pretrained(model_path,
                                                                           classifier_dropout=0.5,
                                                                           num_labels=num_labels)
    tokenizer = BertTokenizerFast.from_pretrained(f'{MODEL_DIR}/bert-base-chinese')
    print(model)
    # load data
    ds_predict = load_dataset(path='csv', data_files=[f'{DS_DIR}/news/predict-news.csv'], cache_dir=CACHE_DIR)
    # 构建输入向量
    ds_predict = ds_predict.get('train').map(
        function=lambda data: conv2input(tokenizer=tokenizer, data=data, max_length=token_length),
        batched=True, batch_size=100)
    # 选择列
    ds_predict.set_format('torch', columns=['input_ids', 'attention_mask'])
    print(ds_predict)
    # 定义优化器,损失函数
    batch_size = 8
    epochs = 5
    warmup_steps = 100
    weight_decay = 0.01
    training_args = TrainingArguments(
        output_dir=RESULT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=LOG_DIR,
        optim="adamw_torch",  # 修复告警
    )
    trainer = Trainer(
        model=model,
        args=training_args,
    )
    output = trainer.predict(ds_predict)
    print(output)


def pad_list(data: list, length: int, pad: any) -> list:
    left = length - len(data)
    if left <= 0:
        return data
    return data + [pad for i in range(left)]


def test_bert():
    model_path = f'{MODEL_DIR}/bert-base-chinese'
    token_bert = BertTokenizer.from_pretrained(model_path)
    token_bert.add_tokens(new_tokens=['今年', '巴基斯坦', '以色列', '入侵'])
    tokens = token_bert.tokenize('we love you, 我爱你, 今年五月份巴基斯坦被以色列入侵')
    print(tokens)


def np2torch():
    np_data = numpy.arange(6).reshape((2, 3))
    torch_data = torch.from_numpy(np_data)
    tensor2array = torch_data.numpy()
    print('numpy array:', np_data)
    print('torch tensor:', torch_data)
    print('tensor to array:', tensor2array)


def tokenize(tokenizer: PreTrainedTokenizer, txt: str, ) -> dict:
    length = 50
    tokens_ognl = ['[CLS]'] + tokenizer.tokenize(text=txt) + ['SEP']
    len_tokens = len(tokens_ognl)
    tokens = pad_list(data=tokens_ognl, length=length, pad='[PAD]')
    # token_ids
    token_ids = tokenizer.convert_tokens_to_ids(tokens_ognl)
    token_ids = pad_list(data=token_ids, length=length, pad=0)
    token_ids = torch.tensor(token_ids).unsqueeze(0)
    # attention_mask
    masks = [1 for i in range(len_tokens)]
    masks = pad_list(data=masks, length=length, pad=0)
    seg_ids = [0 for i in range(length)]
    return {
        "tokens": tokens,
        "token_ids": token_ids,
        "seg_ids": seg_ids,
        "attention_mask": masks,
    }


if __name__ == '__main__':
    # test_bert()
    # np2torch()
    tuning_news()
    # predict_news()
