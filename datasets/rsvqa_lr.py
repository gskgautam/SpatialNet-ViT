import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
from transformers import BertTokenizer
from collections import defaultdict

class RSVQALRDataset(Dataset):
    """
    PyTorch Dataset for RSVQA-LR.
    Supports VQA (presence, comparison, urban/rural) and counting tasks.
    """
    def __init__(self, jsonl_path, tokenizer=None, img_transform=None, task_type='vqa_presence', answer2idx=None):
        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained('bert-base-uncased')
        self.img_transform = img_transform
        self.task_type = task_type
        self.answer2idx = answer2idx or defaultdict(lambda: 0)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['image']
        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"[Error] Could not load image {img_path}: {e}")
            raise RuntimeError(f"Image loading failed for {img_path}")
        if self.img_transform:
            image = self.img_transform(image)
        question = sample['question']
        answer = sample['answer']
        q_tokens = self.tokenizer(question, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
        if self.task_type == 'count':
            try:
                answer_val = float(answer)
            except:
                answer_val = 0.0
            return {
                'image': image,
                'question': q_tokens['input_ids'].squeeze(0),
                'attention_mask': q_tokens['attention_mask'].squeeze(0),
                'count': torch.tensor(answer_val, dtype=torch.float),
                'task': 'count',
            }
        else:
            answer_idx = self.answer2idx[answer]
            return {
                'image': image,
                'question': q_tokens['input_ids'].squeeze(0),
                'attention_mask': q_tokens['attention_mask'].squeeze(0),
                'answer': torch.tensor(answer_idx, dtype=torch.long),
                'task': self.task_type,
            } 