
# GPU 하나만 사용하는 환경에서 사용함

import os
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import Config
from utils import compute_loss
from checkpoint import save_checkpoint, load_checkpoint
from Data.tokenizer_bert import CustomTokenizer
from collections import deque
import argparse
import bert_models


import models
import optim
from utils import set_seeds, get_device

class CodeDataset(Dataset):
    def __init__(self, data_file, tokenizer, project_dir, max_len, pipeline=[]):
        """
        Args:
            data_file (str): Path to the data file containing file paths.
            tokenizer (CustomTokenizer): An instance of CustomTokenizer.
            project_dir (str): Directory for ByteTok.
            max_len (int): Maximum sequence length.
            pipeline (list): List of preprocessing functions.
        """
        self.file_paths = []
        self.tokenizer = tokenizer
        self.project_dir = project_dir
        self.max_len = max_len
        self.pipeline = pipeline

        with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                file_path = line.strip()
                if not file_path:
                    continue
                self.file_paths.append(file_path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as token_file:
            tokens_line = token_file.read().strip()

        
        tokens = tokens_line.split()

        
        instance = (tokens, tokens)

        
        for proc in self.pipeline:
            instance = proc(instance)

        src_input_ids, src_attention_mask, labels = instance

        
        src_input_ids = torch.tensor(src_input_ids, dtype=torch.long)
        src_attention_mask = torch.tensor(src_attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            'input_ids': src_input_ids,
            'attention_mask': src_attention_mask,
            'labels': labels
        }

class Pipeline():
    """ Pre-process Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError
    
class Preprocess4Seq2Seq(Pipeline):
    
    def __init__(self, max_len, tokenizer):
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __call__(self, instance):
        src_tokens, tgt_tokens = instance

        
        src_tokens = src_tokens[:self.max_len - 2]
        tgt_tokens = tgt_tokens[:self.max_len - 2]

        
        src_tokens = [self.tokenizer.bos_token] + src_tokens + [self.tokenizer.eos_token]
        tgt_tokens = [self.tokenizer.bos_token] + tgt_tokens + [self.tokenizer.eos_token]

        
        src_input_ids = self.tokenizer.convert_tokens_to_ids(src_tokens)
        tgt_input_ids = self.tokenizer.convert_tokens_to_ids(tgt_tokens)

        
        src_attention_mask = [1] * len(src_input_ids)
        tgt_attention_mask = [1] * len(tgt_input_ids)

        
        src_padding_length = self.max_len - len(src_input_ids)
        tgt_padding_length = self.max_len - len(tgt_input_ids)

        src_input_ids += [self.tokenizer.pad_token_id] * src_padding_length
        tgt_input_ids += [self.tokenizer.pad_token_id] * tgt_padding_length

        src_attention_mask += [0] * src_padding_length
        tgt_attention_mask += [0] * tgt_padding_length

        return (src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask)
        


class SpanMasking(Pipeline):
    
    def __init__(self, mask_prob=0.15, max_span_length=3, tokenizer=None):
        
        self.mask_prob = mask_prob
        self.max_span_length = max_span_length
        self.tokenizer = tokenizer

    def __call__(self, instance):
        src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask = instance

        
        maskable_indices = [
            i for i, token_id in enumerate(src_input_ids)
            if src_attention_mask[i] == 1 and
               token_id != self.tokenizer.bos_token_id and
               token_id != self.tokenizer.eos_token_id and
               token_id != self.tokenizer.pad_token_id
        ]

        maskable_length = len(maskable_indices)
        total_mask = max(1, int(maskable_length * self.mask_prob))

        mask_indices = []
        attempts = 0  
        max_attempts = maskable_length * 2

        while total_mask > 0 and attempts < max_attempts:
            span_length = random.randint(1, min(self.max_span_length, total_mask))
            possible_starts = [
                i for i in maskable_indices
                if i not in mask_indices and
                   all(j not in mask_indices for j in range(i, min(i + span_length, len(src_input_ids))))
            ]

            if not possible_starts:
                break 

            start_idx = random.choice(possible_starts)
            end_idx = min(start_idx + span_length, len(src_input_ids))
            new_mask = list(range(start_idx, end_idx))
            mask_indices.extend(new_mask)
            total_mask -= len(new_mask)
            attempts += 1

        
        mask_indices = sorted(list(set(mask_indices)))

        
        masked_src_input_ids = src_input_ids.copy()
        labels = tgt_input_ids.copy()

        for idx in mask_indices:
            masked_src_input_ids[idx] = self.tokenizer.mask_token_id
            
            labels[idx] = src_input_ids[idx]

        
        for i in range(len(labels)):
            if i not in mask_indices:
                labels[i] = -100  

        return (
            masked_src_input_ids,  
            src_attention_mask,    
            labels                 
        )

class BERTModelForPretrain(nn.Module):
    def __init__(self, cfg):
        
        super().__init__()
        self.bert = bert_models.BERT(cfg)
    
    def forward(self, input_ids, attention_mask):
        encoded_layers, pooled_output, logits = self.bert(
            src_input_ids=input_ids,
            src_mask=attention_mask
        )
        
        return logits




def evaluate(model, validation_loader, device):
    model.eval()
    total_loss = 0.0
    total_all_tokens = 0
    total_correct = 0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    with torch.no_grad():
        for batch in tqdm(validation_loader, desc="Validation"):
            
            masked_input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask
            )  # logits: (batch_size, seq_len, vocab_size)

            
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

            
            preds = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)
            mask = labels != -100  
            correct = (preds == labels) & mask
            total_correct += correct.sum().item()
            total_all_tokens += attention_mask.sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / len(validation_loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    model.train()
    return avg_loss, accuracy, total_correct, total_tokens, total_all_tokens

def main(
    data_file='', # put your train input paths
    project_dir='', # you dont need to use this variable
    train_cfg='Model/config/train.json',
    model_cfg='Model/config/model.json',
    model_file=None,
    vocab_file='Data/vocab.txt', 
    save_dir='', # put your save directory path
    GPUs='4', # put your GPU number
    test_mode=False,  
    predict_mode=False,  
    input_file=None,  
    output_file=None  
):
    import os
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from torch.utils.tensorboard import SummaryWriter
    

    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPUs
    device = get_device()
    
    
    cfg = Config.from_json(train_cfg)
    model_cfg = models.Config.from_json(model_cfg)
    
    
    set_seeds(cfg.seed)
    
    
    tokenizer = CustomTokenizer(vocab_file=vocab_file)
    
    if not test_mode and not predict_mode:
        
        pipeline = [Preprocess4Seq2Seq(model_cfg.max_len, tokenizer),
                    SpanMasking(mask_prob=0.15, max_span_length=5, tokenizer=tokenizer)]
        dataset = CodeDataset(
            data_file=data_file,
            tokenizer=tokenizer,
            project_dir=project_dir,
            max_len=model_cfg.max_len,
            pipeline=pipeline
        )
        data_iter = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        
        
        model = BERTModelForPretrain(model_cfg)
        model.to(device)
        
        
        total_steps = len(data_iter) * cfg.n_epochs
        optimizer, scheduler = optim.build_optimizer_and_scheduler(
            cfg, model, total_steps
        )
        
        의
        pad_token_id = tokenizer.pad_token_id
    
        writer = SummaryWriter(log_dir=save_dir)
    
        validation_dataset = CodeDataset(
            data_file='',  # put your data path
            tokenizer=tokenizer,
            project_dir=project_dir,
            max_len=model_cfg.max_len,
            pipeline=pipeline
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
        if model_file:
            print(f"Loading checkpoint from {model_file}")
            checkpoint = torch.load(model_file, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            global_step = checkpoint.get('step', 0)
            start_epoch = checkpoint.get('epoch', -1) + 1
            print(f"Resuming training from epoch {start_epoch}, global_step {global_step}")
        else:
            
            start_epoch, global_step = load_checkpoint(save_dir, model, optimizer, scheduler, device)
    
        
        model.train()
        gradient_accumulation_steps = cfg.gradient_accumulation_steps
        
        for epoch in range(start_epoch, cfg.n_epochs):
            epoch_loss = 0
            progress_bar = tqdm(data_iter, desc=f"Epoch {epoch+1}/{cfg.n_epochs}")
            optimizer.zero_grad()
    
            for batch in progress_bar:
                
                
                src_input_ids = batch['input_ids'].to(device)
                src_attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                
                
                lm_logits = model(
                    input_ids=src_input_ids,
                    attention_mask=src_attention_mask
                )
        
                
                loss = compute_loss(lm_logits, labels)
                loss = loss / gradient_accumulation_steps
                loss.backward()
    
                
                if (global_step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
    
                
                epoch_loss += loss.item() * gradient_accumulation_steps
                global_step += 1
        
                
                progress_bar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})
        
                
                if global_step % cfg.save_steps == 0:
                    save_checkpoint(model, optimizer, scheduler, cfg, global_step, save_dir)
        
            
            avg_epoch_loss = epoch_loss / len(data_iter)
            print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")
            writer.add_scalar('Loss/epoch_avg', avg_epoch_loss, epoch)
            
            
            val_loss, val_accuracy, total_correct, total_tokens = evaluate(model, validation_loader, device)
            print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Accuracy/validation', val_accuracy, epoch)
        
        
            
            save_checkpoint(model, optimizer, scheduler, cfg, global_step, save_dir, epoch)
            model.train()
        
        writer.close()
    elif test_mode:
        
        
        if not model_file:
            raise ValueError("please define save point")
        
        
        print(f"Loading checkpoint from {model_file}")
        checkpoint = torch.load(model_file, map_location=device)
        model = BERTModelForPretrain(model_cfg)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        
        pipeline = [Preprocess4Seq2Seq(model_cfg.max_len, tokenizer),
                    SpanMasking(mask_prob=0.15, max_span_length=5, tokenizer=tokenizer)]
        test_dataset = CodeDataset(
            data_file='put your test data path',  # put your test data path
            tokenizer=tokenizer,
            project_dir=project_dir,
            max_len=model_cfg.max_len,
            pipeline=pipeline
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loss, test_accuracy, total_correct, total_tokens, total_all_tokens = evaluate(model, test_loader, device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Total correct: {total_correct}, Total mask tokens: {total_tokens}, Total tokens: {total_all_tokens}")

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training, test")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='mode: train, test')
    parser.add_argument('--model_file', type=str, help='checkpoint path')
    args = parser.parse_args()

    if args.mode == 'train':
        main()
    elif args.mode == 'test':
        if not args.model_file:
            raise ValueError("please define model save point.")
        main(model_file=args.model_file, test_mode=True)