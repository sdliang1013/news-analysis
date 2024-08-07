"""test python"""
from typing import Tuple, MutableMapping

from datasets import load_dataset, Dataset
from modelscope import MsDataset, pipeline, Exporter
from modelscope.preprocessors import TextClassificationTransformersPreprocessor
from modelscope.trainers import build_trainer

from ai_env.constants import DATA_DIR

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


def conv2input(data: MutableMapping) -> MutableMapping:
    # 分类
    labels = data.get(col_label, [])
    if labels:
        data.update(
            {
                "label": list(map(classify_label, labels)),
            }
        )
    return data


# 这个方法在trainer读取configuration.json后立即执行，先于构造模型、预处理器等组件
def cfg_modify_fn(cfg):
    batch_size = 16  # 与cpu核数相当
    cfg.task = "text-classification"
    cfg.pipeline = {"type": "text-classification"}
    cfg.preprocessor = {
        "train": {
            # 配置预处理器名字
            "type": "bert-seq-cls-tokenizer",
            # 配置句子1的key
            "first_sequence": col_content,
            # 配置label
            "label": "label",
            # 配置mode
            "mode": "train",
            "label2id": labels_dict,
        },
        "val": {
            # 配置预处理器名字
            "type": "bert-seq-cls-tokenizer",
            # 配置句子1的key
            "first_sequence": col_content,
            # 配置label
            "label": "label",
            "mode": "eval",
            "label2id": labels_dict,
        },
    }
    cfg.model["num_labels"] = num_labels
    cfg["train"] = {
        "work_dir": RESULT_DIR,
        "max_epochs": 5,
        "dataloader": {
            # batch_size 16
            "batch_size_per_gpu": batch_size,
            "workers_per_gpu": 0,
        },
        "optimizer": {
            # optimizer信息
            "type": "AdamW",
            "lr": 0.01,
            "options": {"grad_clip": {"max_norm": 2.0}},
        },
        "lr_scheduler": {
            # lr_scheduler信息，注意torch版本是否包含该lr_scheduler
            "type": "LinearLR",
            "start_factor": 1.0,
            "end_factor": 0.0,
            "options": {"by_epoch": False},
        },
        "hooks": [
            {
                "type": "CheckpointHook",
                "interval": 1,
                "by_epoch": False,
            },
            {
                "type": "EvaluationHook",
                "interval": 1,
                "by_epoch": False,
            },
        ],
    }
    cfg["evaluation"] = {
        "dataloader": {
            # batch_size 16
            "batch_size_per_gpu": batch_size,
            "workers_per_gpu": 0,
            "shuffle": False,
        },
        "metrics": [
            {
                "type": "seq-cls-metric",
                "label_name": "labels",
                "logit_name": "logits",
            }
        ],
    }
    cfg.train.lr_scheduler.total_iters = int(train_len / cfg.train.dataloader.batch_size_per_gpu) * cfg.train.max_epochs
    return cfg


def tuning_model():
    """
    预模型训练
    PS: 实际运行时, 没有看到train步骤, 只有test步骤
    :return:
    """
    # load model
    model_path = f"{MODEL_DIR}/nlp_roberta_backbone_base_std"
    preprocessor = TextClassificationTransformersPreprocessor(
        model_dir=model_path,
        first_sequence="content",
        padding=True,
        max_length=128,
        use_fast=True,
    )
    # load data
    # df = pd.read_csv(filepath_or_buffer=f'{DS_DIR}/news/sogou-news.txt.csv')
    ds = load_dataset(
        path="csv",
        data_files=[f"{DS_DIR}/news/sogou-news.txt.csv"],
        cache_dir=CACHE_DIR,
    )
    # 构建输入向量
    ds_train, ds_test = train_test_split(ds=ds.get("train"), test_ratio=0.1)
    ds_train = ds_train.map(
        function=lambda data: conv2input(data=data), batched=True, batch_size=100
    )
    ds_test = ds_test.map(
        function=lambda data: conv2input(data=data), batched=True, batch_size=100
    )
    # 选择列
    ds_train.set_format("torch", columns=[col_content, "label"])
    ds_test.set_format("torch", columns=[col_content, "label"])

    global train_len
    train_len = len(ds_train)
    print(ds_train)
    print(ds_test)

    training_args = dict(
        model=model_path,
        train_dataset=MsDataset.to_ms_dataset(ds_train),
        eval_dataset=MsDataset.to_ms_dataset(ds_test),
        cfg_modify_fn=cfg_modify_fn,
        cfg_file=f"{model_path}/configuration.json",
        device="cpu",
    )
    trainer = build_trainer(default_args=training_args)
    # 训练
    trainer.train()
    # 评估
    trainer.evaluate()
    # 保存
    output_files = Exporter.from_model(model_path).export_onnx(
        opset=13,
        output_dir=f"{MODEL_DIR}/tuning/news/",
    )
    print(output_files)


def predict_classify():
    """
    新闻分类模型预测
    :return:
    """
    # load model
    model_path = f"{MODEL_DIR}/tuning/news"
    pl = pipeline(task="text-classification", model=model_path, device="cpu")
    print(pl.model)
    # load data
    output = pl(
        "百度在AI时代开启自我革命，用AI重构搜索和传统业务，同时也在自动驾驶等新业务上展现领先实力。尽管过程中遭遇冷眼和嘲讽，百度坚持AI发展，并在ChatGPT出现后走在中文互联网前列。百度借助AI对自身进行深度改造，展现了巨大的发展潜能和勇气，焕发新生。"
    )
    print(output)


if __name__ == "__main__":
    train_len = 700
    tuning_model()
    # predict_classify()
