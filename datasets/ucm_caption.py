import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
from transformers import BertTokenizer

class UCMCaptionDataset(Dataset):
    """
    PyTorch Dataset for UCM-Caption.
    Supports both captioning and land-use classification tasks.
    """
    def __init__(self, processed_json, split, tokenizer=None, img_transform=None):
        with open(processed_json, 'r') as f:
            data = json.load(f)
        self.samples = [x for x in data if x['split'] == split]
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained('bert-base-uncased')
        self.img_transform = img_transform
        self.img_dir = os.path.dirname(processed_json) + '/images_256/'
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.img_dir, sample['image'])
        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"[Error] Could not load image {img_path}: {e}")
            raise RuntimeError(f"Image loading failed for {img_path}")
        if self.img_transform:
            image = self.img_transform(image)
        captions = sample['captions']
        # Tokenize all 5 captions
        tokens = self.tokenizer(captions, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
        # Placeholder: class label (should be provided in processed JSON for classification)
        label = sample.get('class', 0)
        return {
            'image': image,
            'captions': tokens['input_ids'],
            'attention_mask': tokens['attention_mask'],
            'label': torch.tensor(label, dtype=torch.long),
            'task': 'caption',
            'landuse_label': torch.tensor(label, dtype=torch.long),
        } 