"""test python"""
from typing import Tuple, MutableMapping

import numpy
import torch
from datasets import load_dataset, Dataset
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    BertForSequenceClassification,
    BertTokenizerFast,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerBase,
)

from tuning.constants import DATA_DIR

BASE_DIR = f"{DATA_DIR}/hugginface"
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
        tokenizer: PreTrainedTokenizer, data: MutableMapping, max_length: int
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


def tuning_news():
    """
    新闻分类预模型训练
    :return:
    """
    # load model
    model_path = f"{MODEL_DIR}/bert-base-chinese"
    model: PreTrainedModel = BertForSequenceClassification.from_pretrained(
        model_path, classifier_dropout=0.5, num_labels=num_labels
    )
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    print(model)
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
        function=lambda data: conv2input(
            tokenizer=tokenizer, data=data, max_length=token_length
        ),
        batched=True,
        batch_size=100,
    )
    ds_test = ds_test.map(
        function=lambda data: conv2input(
            tokenizer=tokenizer, data=data, max_length=token_length
        ),
        batched=True,
        batch_size=100,
    )
    # 选择列
    ds_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    ds_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    print(ds_train)
    print(ds_test)
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
        train_dataset=ds_train,
        eval_dataset=ds_test,
    )
    # 训练
    trainer.train()
    # 评估
    trainer.evaluate()
    # 保存
    trainer.save_model(f"{MODEL_DIR}/tuning/news/")


def predict_news():
    """
    新闻分类模型预测
    :return:
    """
    # load model
    model_path = f"{MODEL_DIR}/tuning/news"
    model: PreTrainedModel = BertForSequenceClassification.from_pretrained(
        model_path, classifier_dropout=0.5, num_labels=num_labels
    )
    tokenizer = BertTokenizerFast.from_pretrained(f"{MODEL_DIR}/bert-base-chinese")
    print(model)
    # load data
    ds_predict = load_dataset(
        path="csv", data_files=[f"{DS_DIR}/news/predict-news.csv"], cache_dir=CACHE_DIR
    )
    # 构建输入向量
    ds_predict = ds_predict.get("train").map(
        function=lambda data: conv2input(
            tokenizer=tokenizer, data=data, max_length=token_length
        ),
        batched=True,
        batch_size=100,
    )
    # 选择列
    ds_predict.set_format("torch", columns=["input_ids", "attention_mask"])
    print(ds_predict)
    # 预测
    trainer = Trainer(
        model=model,
    )
    output = trainer.predict(ds_predict)
    print(output)


def pad_list(data: list, length: int, pad: any) -> list:
    left = length - len(data)
    if left <= 0:
        return data
    return data + [pad for i in range(left)]


def test_tokens():
    txt = "今年５月，一则“吴宇森罹患喉癌 ”的传闻震动电影圈，虽然老友兼制片人张家振第一 时间辟谣，但因导演本人一直未能现身，还是让不少媒体和影迷心怀疑虑。７月１９日，康复回京的吴宇森首次露面，宣布明年开拍《生死恋》之前会先拍一部西片《独行杀手》（即美版《喋血双雄》），同时对当下中国电影的现状进行了一番点评，最后感叹：“我们还没有真正的华语大片。”＃笼Ｓ械牡佳蔟Ａ分镜头都不会２煞弥校吴宇森澄清了自己的病情并透露新片近况。“在这里要感谢北京的几位医生，是他们发现我颈部有肿瘤，还好是良性，但如果再晚一步的话，就可能恶化了。后来我的治疗因为迁就工作的方便去了美国，经过几个月的治疗和休养，现在身体基本恢复健康了。再次谢谢所有网友和影迷对我的关怀，我现在复原真的很好，我会以后更当心一点。”＝几年，从好莱坞回来的吴宇森耳闻目睹了华语电影圈很多怪现象，一向温和的他也忍不住吐槽，“我听说有些年轻导演很容易得了机会，不知道自己要什么，也不知道镜头摆在哪里。我也听说，有些导演是副导演帮他分镜头。这些情况以前是不会发生的，我们那一代被训练下来，动作戏都是我设计镜头，我跟动作指导在场地中央，两个人拿着枪（是玩具枪），怎么样打怎么样跑怎么跳怎么开枪，我镜头怎么拍出周润发拿枪姿势的美感，就是张彻那样的大导演，也是自己分镜头的，这是一种工作态度，我们到现场就清楚，我的风格是什么！”＃慢Ｎ颐腔姑挥幸徊魁Ｕ嬲的华语大片８粘龅谰透过吴宇森的陈可辛导演曾经说过：“提起Ｊｏｈｎ　Ｗｏｏ（吴宇森英文名），就是拍大片的，这是大家的共识。”正因如此，当听到在好莱坞拍过《变脸》和在内地导过《赤壁》的吴宇森说“我们还没有一部真正的华语大片”时，震撼之余，也多了几分当事人检讨的诚意。！安豢推的讲，我觉得到现在为止我们还没有一部真正的华语大片。其实有一部，可能很多人都不同意的，就是张艺谋的《活着》，有大片的格局，是一个大时代里的故事，有大片那份感觉。现在的大片，以为千军万马都是大片，我的《赤壁》也犯同样的毛病，剧本比较薄弱，太多的篇幅放在场面上，故事缺乏一种浓厚的人文精神，就是功力不足。我不好意思讲太多别人的作品，我只是觉得，到现在我们还没有一部像《七武士》、《阿拉伯的劳伦斯》那样的大片，我们的电影圈没有拍大片的气候，我们的工作团队没有拍大片那种熟练的技术，所以《赤壁》拍得很辛苦。”！按蠹颐挥泄那种经验，在取景，在镜头角度方面，花很多时间去拍摄，不好再来过，换过一个角度再去拍，没有大片细密的设计和计算的精准，制片的筹备功夫还是不够，应该有胡金铨那种画面的工整精确，还有导演创作者团队达不成大片的共识。除了技术，一些演员的演出，比较夸张，表演比较公式化，我怎样看都不像人家的大片来得那么自然，真实的感觉不够。但是大家在努力，年轻人不断加入进来，花更多的钱，更多的时间，让大家学习了解怎么去配合一部大片。当然，这纯粹是个人感觉，我并不处在一个批评的角度，我期待真正的大片出现。我们创作的环境和条件还是不够，制作方面还是有很多改进的空间。”Ｗ魑专业电影人，吴宇森最后表达了和很多业内人士同样的忧虑，至于怎样解决，他没有说，对于一位已经六十六岁的影坛宿将而言，讲再多，不如拍好下一部电影来得实际。"
    model_path = f"{MODEL_DIR}/bert-base-chinese"
    token_bert = BertTokenizerFast.from_pretrained(model_path)
    token_bert.add_tokens(new_tokens=["今年", "吴宇森", "电影", "导演"])
    tokens = token_bert.tokenize(text=txt)
    print(tokens)


def np2torch():
    np_data = numpy.arange(6).reshape((2, 3))
    torch_data = torch.from_numpy(np_data)
    tensor2array = torch_data.numpy()
    print("numpy array:", np_data)
    print("torch tensor:", torch_data)
    print("tensor to array:", tensor2array)


def tokenize(tokenizer: PreTrainedTokenizerBase, txt: str) -> dict:
    tokens = ["[CLS]"] + tokenizer.tokenize(text=txt) + ["SEP"]
    max_length = len(tokens)
    # tokens = pad_list(data=tokens, length=max_length, pad='[PAD]')
    # input_ids
    input_ids = tokenizer.encode(text=txt)
    input_ids = pad_list(data=input_ids, length=max_length, pad=0)
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    # attention_mask
    masks = [1 for i in range(max_length)]
    masks = pad_list(data=masks, length=max_length, pad=0)
    token_type_ids = [0 for i in range(max_length)]
    return {
        "tokens": tokens,
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": masks,
    }


if __name__ == "__main__":
    # test_tokens()
    # np2torch()
    tuning_news()
    # predict_news()
