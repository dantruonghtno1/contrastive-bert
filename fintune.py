from __future__ import print_function
from transformers import Trainer
from transformers import AutoModel
import torch
import torch.nn.functional as FF
import torch.nn as nn
import typing as tp
from transformers import TrainingArguments
from config import Param


    
from datasets import load_dataset, load_metric
from transformers import TrainingArguments, Trainer, AutoTokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
import warnings
import numpy as np

# from SupCsTrainer import SupCsTrainer

warnings.filterwarnings('ignore')

def main():
    param = Param()
    args = param.args

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained('./contrastive_phobert_base')
    for param in model.base_model.parameters():
        param.requires_grad = False

    dataset = load_dataset(args.data_path)
    # dataset = dataset.rename_column("labels", "label")

    def preprocess_function(examples):
            # Tokenize the texts
            result = tokenizer(examples["text"], padding="max_length", max_length=256, truncation=True)
            return result
    dataset = dataset.map(
                preprocess_function,
                batched=True,           
    )


    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    
    if args.sampled_for_model_test:
        print("process data for model test!!!!")
        train_dataset = train_dataset.select(range(1000))
        eval_dataset = eval_dataset.select(range(1000))

    # metric
    metric = load_metric("f1")
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
        acc = accuracy_score(labels, preds)
        return {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall
        }

    args = TrainingArguments(
            output_dir = './results',
            save_total_limit = 1,
            num_train_epochs=5,
            per_device_train_batch_size=28,  
            per_device_eval_batch_size=64,
            evaluation_strategy = 'epoch',
            logging_steps = 200,
            learning_rate = 1e-04,
            eval_steps = 200,
            warmup_steps=50, 
            report_to ='wandb',
            weight_decay=0.01,               
            logging_dir='./logs',
        )

    trainer = Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
    trainer.train()
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
