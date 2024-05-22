"""test python"""
from typing import Tuple, MutableMapping

from datasets import load_dataset, Dataset
from modelscope import AutoTokenizer, AutoModelForSequenceClassification
from swift import LoRAConfig, Swift, TrainingArguments, Trainer
from transformers import default_data_collator

from tuning.constants import DATA_DIR

BASE_DIR = f"{DATA_DIR}/modelscope"
CACHE_DIR = f"{BASE_DIR}/cache"
DS_DIR = f"{BASE_DIR}/datasets"
MODEL_DIR = f"{BASE_DIR}/models"
RESULT_DIR = f"{BASE_DIR}/results"
LOG_DIR = f"{BASE_DIR}/logs"
token_length = 128
col_label = "classify"
col_content = "content"

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
    return ds_split.get("train"), ds_split.get("test")


def conv2input(
        tokenizer: AutoTokenizer, data: MutableMapping, max_length: int
) -> MutableMapping:
    # 内容
    tokens = tokenizer(
        text=data.get(col_content),
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=True,
        return_attention_mask=True,
    )
    data.update(tokens)
    # 分类
    labels = data.get(col_label, [])
    if labels:
        data.update(
            {
                "label": list(map(classify_label, labels)),
            }
        )
    return data


def tuning_model():
    """
    预模型训练
    :return:
    """
    # load model
    model_path = f"{MODEL_DIR}/nlp_roberta_backbone_base_std"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, classifier_dropout=0.5, num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    lora_config = LoRAConfig(target_modules=["query", "key", "value"])
    model = Swift.prepare_model(model=model, config=lora_config)
    # load data
    # df = pd.read_csv(filepath_or_buffer=f'{DS_DIR}/news/sogou-news.txt.csv')
    ds = load_dataset(
        path="csv",
        data_files=[f"{DS_DIR}/news/sogou-news.txt.csv"],
        cache_dir=CACHE_DIR,
    )
    # 构建输入向量
    ds_train, ds_eval = train_test_split(ds=ds.get("train"), test_ratio=0.1)
    ds_train = ds_train.map(
        function=lambda data: conv2input(
            tokenizer=tokenizer, data=data, max_length=128
        ),
        batched=True,
        batch_size=100,
    )
    ds_eval = ds_eval.map(
        function=lambda data: conv2input(
            tokenizer=tokenizer, data=data, max_length=128
        ),
        batched=True,
        batch_size=100,
    )
    # 选择列
    ds_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    ds_eval.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    print(ds_train)
    print(ds_eval)
    batch_size = 16
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
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=default_data_collator,
    )
    # 训练
    trainer.train()
    # 评估
    trainer.evaluate()
    # 保存
    trainer.save_model(f"{MODEL_DIR}/tuning/news/")


def predict_classify():
    """
    新闻分类模型预测
    :return:
    """
    # load model
    model_path = f"{MODEL_DIR}/tuning/news"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, classifier_dropout=0.5, num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = Swift.from_pretrained(model=model)
    # load data
    output = model(
        **tokenizer(
            "近日，理想汽车被投资者集体诉讼至法院，投资者指控该公司及其部分高管虚假陈述，违反证券法令投资者受损，并向法院申请判令被告赔偿因其违法行为导致股价下跌给投资者造成的损失。",
            return_tensors="pt",
        )
    )
    print(output)


if __name__ == "__main__":
    tuning_model()
    # predict_classify()
