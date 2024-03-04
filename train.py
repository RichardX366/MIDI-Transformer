from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
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
    return tokenizer(s["text"], padding="max_length", truncation=True)


rawData = []

for file in listdir("data"):
    text = loadFile(file)
    text.update(tokenize_function(text))
    rawData.append(text)

unsplit_dataset = Dataset.from_list(rawData)

dataset = unsplit_dataset.train_test_split(test_size=0.2)

model = AutoModelForSequenceClassification.from_pretrained(
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
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
