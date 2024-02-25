from math import floor
from transformers import (
    AutoTokenizer,
    AutoModelForNextSentencePrediction,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from os import listdir
import numpy as np
import evaluate

metric = evaluate.load("accuracy")

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


def loadFile(file):
    data = np.load("data/" + file)
    s = ""
    for note in data:
        s += ",".join(map(str, note)) + "\n"
    return {"text": s}


def tokenize_function(s):
    return tokenizer(s, padding="max_length", truncation=True)


rawData = list(map(loadFile, listdir("data")))

train_dataset = Dataset.from_list(rawData[: floor(len(rawData) * 0.8)]).map(
    tokenize_function, batched=True
)

test_dataset = Dataset.from_list(rawData[floor(len(rawData) * 0.8) :]).map(
    tokenize_function, batched=True
)


model = AutoModelForNextSentencePrediction.from_pretrained(
    "google-bert/bert-base-cased"
)

training_args = TrainingArguments(
    output_dir="test_trainer", evaluation_strategy="epoch"
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print(dataset["train"][0])
