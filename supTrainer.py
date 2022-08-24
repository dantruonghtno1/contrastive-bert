from __future__ import print_function
from transformers import Trainer
from transformers import AutoModel
import torch
import torch.nn.functional as FF
import torch.nn as nn
import typing as tp
from transformers import TrainingArguments
from config import Param


class SupCsTrainer(Trainer):
    def __init__(
        self,
        w_drop_out: tp.Optional[tp.List[float]] = [0.0,0.05,0.2],
        temperature: tp.Optional[float] = 0.05,
        def_drop_out: tp.Optional[float]=0.1,
        pooling_strategy: tp.Optional[str]='pooler',
        **kwargs
        ):

    
        super().__init__(**kwargs)
        self.w_drop_out = w_drop_out
        self.temperature_s = temperature 
        self.def_drop_out = def_drop_out
        self.pooling_strategy = pooling_strategy
        if pooling_strategy == 'pooler':
            print('# Employing pooler ([CLS]) output.')
        else:
            print('# Employing mean of the last hidden layer.')
        
    def compute_loss(
        self,
        model: nn,
        inputs: tp.Dict,
        return_outputs: tp.Optional[bool]=False,
        )-> tp.Tuple[float, torch.Tensor]:

        labels = inputs.pop("labels")
        
        # ----- Default p = 0.1 ---------#
        output = model(**inputs)
        if self.pooling_strategy == 'pooler':
            try:
                logits = output.pooler_output.unsqueeze(1) 
            except:
                logits = output.last_hidden_state.mean(dim=1, keepdim=True)
        else:
            logits = output.last_hidden_state.mean(dim=1, keepdim=True)
        
        # ---- iteratively create dropouts -----#
        for p_dpr in self.w_drop_out:
            # -- Set models dropout --#
            if p_dpr != self.def_drop_out:
                model = self.set_dropout_mf(model, w=p_dpr)
            # ---- concat logits ------#
            if self.pooling_strategy == 'pooler':
                # --------- If model does offer pooler output --------#
                try:
                    logits = torch.cat((logits, model(**inputs).pooler_output.unsqueeze(1)), 1)
                except:
                    logits = torch.cat((logits, model(**inputs).last_hidden_state.mean(dim=1, keepdim=True)), 1)
            else:
                logits = torch.cat((logits, model(**inputs).last_hidden_state.mean(dim=1, keepdim=True)), 1)
            
        # ---- L2 norm ---------#
        logits = FF.normalize(logits, p=2, dim=2)
        
        #----- Set model back to dropout = 0.1 -----#
        if p_dpr != self.def_drop_out: model = self.set_dropout_mf(model, w=0.1)
        
        
        # SupContrast
        loss_fn = SupConLoss(temperature=self.temperature_s) # temperature=0.1

        loss = loss_fn(logits, labels) # added rounding for stsb
        
        return (loss, output) if return_outputs else loss
    
    def set_dropout_mf(
        self, 
        model:nn, 
        w:tp.List[float]
        ):

        # ------ set hidden dropout -------#
        if hasattr(model, 'module'):
            model.module.embeddings.dropout.p = w
            for i in model.module.encoder.layer:
                i.attention.self.dropout.p = w
                i.attention.output.dropout.p = w
                i.output.dropout.p = w        
        else:
            model.embeddings.dropout.p = w
            for i in model.encoder.layer:
                i.attention.self.dropout.p = w
                i.attention.output.dropout.p = w
                i.output.dropout.p = w
            
        return model

    
class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
from datasets import load_dataset, load_metric
from transformers import TrainingArguments, Trainer, AutoTokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel
from sklearn.metrics import classification_report
import warnings
import numpy as np

# from SupCsTrainer import SupCsTrainer

warnings.filterwarnings('ignore')

def main():
    param = Param()
    args = param.args

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = RobertaModel.from_pretrained(args.model_path)

    dataset = load_dataset(args.data_path)
    dataset = dataset.rename_column("labels", "label")

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
        print("process data for model test !!!!")
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


    CL_args = TrainingArguments(
            output_dir = './results_contrastive',
            save_total_limit = 1,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,  
            per_device_eval_batch_size=args.batch_size,
            evaluation_strategy = 'no',
            logging_steps = 200,
            learning_rate = 5e-05,
            warmup_steps=50, 
            report_to ='wandb',
            weight_decay=0.01,               
            logging_dir='./logs',
        )

    if args.pooling_strategy_type == 1:
        pooling_strategy = 'pooler'
    else:
        pooling_strategy = 'mean'

    SupCL_trainer = SupCsTrainer(
                w_drop_out=[0.0,0.05],
                temperature= 0.05,
                def_drop_out=0.1,
                pooling_strategy=pooling_strategy,
                model = model,
                args = CL_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics
            )

    SupCL_trainer.train()
    SupCL_trainer.save_model('./contrastive_phobert_base')

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
