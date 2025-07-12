import os
import json
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

def resize_image(img_path, out_path, size=(256, 256)):
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(size, Image.BICUBIC)
        img.save(out_path)
        return True
    except (FileNotFoundError, UnidentifiedImageError) as e:
        print(f"[Warning] Could not process image {img_path}: {e}")
        return False

def main():
    DATASET_JSON = 'UCM_dataset/RSICD_optimal-master/dataset_rsicd.json'
    IMAGES_DIR = 'UCM_dataset/RSICD_optimal-master/images/'
    RESIZED_DIR = 'UCM_dataset/RSICD_optimal-master/images_256/'
    SPLIT_DIR = 'UCM_dataset/RSICD_optimal-master/splits/'
    PROCESSED_JSON = 'UCM_dataset/RSICD_optimal-master/ucm_processed.json'
    os.makedirs(RESIZED_DIR, exist_ok=True)
    os.makedirs(SPLIT_DIR, exist_ok=True)
    with open(DATASET_JSON, 'r') as f:
        data = json.load(f)
    entries = data if isinstance(data, list) else data.get('images', data)
    if not isinstance(entries, list):
        entries = data
    samples = []
    for entry in entries:
        img_file = entry.get('filename')
        if not img_file:
            print(f"[Warning] Missing filename in entry: {entry}")
            continue
        img_path = os.path.join(IMAGES_DIR, img_file)
        resized_path = os.path.join(RESIZED_DIR, img_file)
        if not resize_image(img_path, resized_path):
            continue
        captions = [s.get('raw', '') for s in entry.get('sentences', []) if 'raw' in s]
        if not captions or len(captions) < 1:
            print(f"[Warning] No valid captions for image {img_file}")
            continue
        samples.append({'image': img_file, 'captions': captions, 'split': entry.get('split', 'train')})
    if not samples:
        print("[Error] No valid samples found. Exiting.")
        return
    train, temp = train_test_split(samples, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    splits = {'train': train, 'val': val, 'test': test}
    for split, items in splits.items():
        with open(os.path.join(SPLIT_DIR, f'{split}.json'), 'w') as f:
            json.dump(items, f, indent=2)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    processed = []
    for sample in samples:
        tokenized = [tokenizer.encode(c, add_special_tokens=True) for c in sample['captions']]
        processed.append({'image': sample['image'], 'captions': tokenized, 'split': sample['split']})
    with open(PROCESSED_JSON, 'w') as f:
        json.dump(processed, f, indent=2)
    print('UCM dataset preparation complete!')

if __name__ == '__main__':
    main() 