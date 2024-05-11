"""test python"""
import numpy
import torch
from datasets import load_dataset, Dataset
from transformers import BertTokenizer, PreTrainedTokenizer, PreTrainedModel, BertForSequenceClassification, \
    BertTokenizerFast, BatchEncoding, TrainingArguments, Trainer

CACHE_DIR = 'D:/data/huggingface/cache'
DS_DIR = 'D:/data/huggingface/datasets'
MODEL_DIR = 'D:/data/huggingface/models'


def pad_list(data: list, length: int, pad: any) -> list:
    left = length - len(data)
    if left <= 0:
        return data
    data = data + [pad for i in range(left)]
    return data + [pad for i in range(left)]


def test_bert():
    model_path = f'{MODEL_DIR}/bert-base-multilingual-uncased'
    # model_bert = BertModel.from_pretrained(model_path,
    #                                        output_hidden_states=True,
    #                                        output_attentions=True)
    token_bert = BertTokenizer.from_pretrained(model_path)
    token_bert.add_tokens(new_tokens=['今年', '巴基斯坦', '以色列', '入侵'])
    tokens = token_bert.tokenize('we love you, 我爱你, 今年五月份巴基斯坦被以色列入侵')
    tokens = ['[CLS]'] + tokens + ['SEP']
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


def conv_classify(classify: str) -> int:
    if classify == '汽车':
        return 1
    elif classify == '财经':
        return 2
    elif classify == '科技':
        return 3
    elif classify == '健康':
        return 4
    elif classify == '体育':
        return 5
    elif classify == '教育':
        return 6
    elif classify == '文化':
        return 7
    elif classify == '军事':
        return 8
    elif classify == '娱乐':
        return 9
    elif classify == '时尚':
        return 10
    return 0


def train_test_split(ds: Dataset, test_ratio: float) -> (Dataset, Dataset):
    ds_split = ds.train_test_split(test_size=test_ratio)
    return ds_split.get('train'), ds_split.get('test')


def conv2input(tokenizer: PreTrainedTokenizer, data: dict) -> BatchEncoding:
    encode = tokenizer.encode_plus(text=data["content"], add_special_tokens=True, max_length=128,
                                   return_token_type_ids=True, return_attention_mask=True, padding=True,
                                   truncation=True)
    return encode


def tuning_news():
    # load model
    model_path = f'{MODEL_DIR}/bert-base-multilingual-uncased'
    model: PreTrainedModel = BertForSequenceClassification.from_pretrained(model_path)
    print(model)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    # load data
    # df = pd.read_csv(filepath_or_buffer=f'{DS_DIR}/news/sogou-news.txt.csv')
    ds = load_dataset(path='csv', data_files=[f'{DS_DIR}/news/sogou-news.txt.csv'], cache_dir=CACHE_DIR)
    ds_train, ds_test = train_test_split(ds=ds.get("train"), test_ratio=0.1)
    print(ds_train)
    print(ds_test)
    # 构架输入向量
    data = conv2input(tokenizer=tokenizer, data={"content": "这是一个东西"})
    # ds_train.set_format('torch', columns=['input_ids', 'attention_mask', 'classify'])
    # ds_train.set_format('torch', columns=['input_ids', 'attention_mask', 'classify'])
    # 定义优化器,损失函数
    batch_size = 8
    epochs = 2
    warmup_steps = 500
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
        eval_dataset=ds_test
    )
    # 训练
    trainer.train()
    # 评估
    trainer.evaluate()


if __name__ == '__main__':
    # test_bert()
    # np2torch()
    tuning_news()
    ...
