import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer
from datasets.ucm_caption import UCMCaptionDataset
from datasets.rsvqa_lr import RSVQALRDataset
from models.spatialnet_vit import SpatialNetViT
from train.metrics import compute_bleu, compute_meteor, compute_rouge, compute_accuracy, compute_mae
from utils.answer_mapping import get_answer2idx

# Configuration
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
L2_REG = 0.01
TASK_LOSS_WEIGHTS = {
    'landuse': 1.0,
    'caption': 1.0,
    'vqa_presence': 1.0,
    'vqa_comparison': 1.0,
    'vqa_urbanrural': 1.0,
    'count': 1.0,
}

# Paths
UCM_PROCESSED_JSON = 'UCM_dataset/RSICD_optimal-master/ucm_processed.json'
RSVQA_JSONL = {
    'train': 'RSVQA-LR Dataset/jsonl/train.jsonl',
    'val': 'RSVQA-LR Dataset/jsonl/val.jsonl',
    'test': 'RSVQA-LR Dataset/jsonl/test.jsonl',
}

# Tokenizer and transforms
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
img_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Build answer2idx mapping from all answers in RSVQA-LR train split
all_answers = []
with open(RSVQA_JSONL['train'], 'r') as f:
    for line in f:
        entry = eval(line)
        all_answers.append(entry['answer'])
answer2idx = get_answer2idx(all_answers)

# DataLoaders for all tasks
ucm_train = UCMCaptionDataset(UCM_PROCESSED_JSON, 'train', tokenizer, img_transform)
ucm_val = UCMCaptionDataset(UCM_PROCESSED_JSON, 'val', tokenizer, img_transform)

rsvqa_train_presence = RSVQALRDataset(RSVQA_JSONL['train'], tokenizer, img_transform, task_type='vqa_presence', answer2idx=answer2idx)
rsvqa_train_comparison = RSVQALRDataset(RSVQA_JSONL['train'], tokenizer, img_transform, task_type='vqa_comparison', answer2idx=answer2idx)
rsvqa_train_urbanrural = RSVQALRDataset(RSVQA_JSONL['train'], tokenizer, img_transform, task_type='vqa_urbanrural', answer2idx=answer2idx)
rsvqa_train_count = RSVQALRDataset(RSVQA_JSONL['train'], tokenizer, img_transform, task_type='count', answer2idx=answer2idx)

train_loaders = {
    'caption': DataLoader(ucm_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
    'landuse': DataLoader(ucm_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
    'vqa_presence': DataLoader(rsvqa_train_presence, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
    'vqa_comparison': DataLoader(rsvqa_train_comparison, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
    'vqa_urbanrural': DataLoader(rsvqa_train_urbanrural, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
    'count': DataLoader(rsvqa_train_count, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
}

# Model
model = SpatialNetViT(embed_dim=512, num_classes=21, vocab_size=30522)
model = model.cuda() if torch.cuda.is_available() else model

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)

# Loss functions
loss_fns = {
    'landuse': nn.CrossEntropyLoss(),
    'caption': nn.CrossEntropyLoss(),
    'vqa_presence': nn.CrossEntropyLoss(),
    'vqa_comparison': nn.CrossEntropyLoss(),
    'vqa_urbanrural': nn.CrossEntropyLoss(),
    'count': nn.MSELoss(),
}

def get_next_batch(loaders):
    """Round-robin sampling from each loader."""
    iters = {k: iter(v) for k, v in loaders.items()}
    while True:
        for k in loaders:
            try:
                yield k, next(iters[k])
            except StopIteration:
                continue

def train():
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for task, batch in get_next_batch(train_loaders):
            optimizer.zero_grad()
            if task == 'caption':
                images = batch['image']
                captions = batch['captions']
                attention_mask = batch['attention_mask']
                outputs = model(images, captions=captions, attention_mask=attention_mask, task=task)
                loss = loss_fns[task](outputs.view(-1, outputs.size(-1)), captions.view(-1))
            elif task == 'landuse':
                images = batch['image']
                labels = batch['landuse_label']
                outputs = model(images, task=task)
                loss = loss_fns[task](outputs, labels)
            elif task in ['vqa_presence', 'vqa_comparison', 'vqa_urbanrural']:
                images = batch['image']
                questions = batch['question']
                attention_mask = batch['attention_mask']
                answers = batch['answer']
                outputs = model(images, questions=questions, attention_mask=attention_mask, task=task)
                loss = loss_fns[task](outputs, answers)
            elif task == 'count':
                images = batch['image']
                questions = batch['question']
                attention_mask = batch['attention_mask']
                counts = batch['count']
                outputs = model(images, questions=questions, attention_mask=attention_mask, task=task)
                loss = loss_fns[task](outputs.squeeze(-1), counts)
            else:
                continue
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}")
        # Validation (to be implemented)
        # validate(model, val_loaders)

if __name__ == '__main__':
    print('Model, optimizer, and loss functions are set up. Ready for training loop implementation.')
    # Uncomment to run training loop when dataloaders are ready
    # train() 