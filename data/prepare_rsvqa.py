import os
import json
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

def resize_images(IMAGES_DIR, RESIZED_DIR):
    for fname in tqdm(os.listdir(IMAGES_DIR), desc='Resizing images'):
        if fname.endswith('.tif'):
            img_path = os.path.join(IMAGES_DIR, fname)
            out_path = os.path.join(RESIZED_DIR, fname.replace('.tif', '.jpg'))
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((256, 256), Image.BICUBIC)
                img.save(out_path, 'JPEG')
            except (FileNotFoundError, UnidentifiedImageError) as e:
                print(f"[Warning] Could not process image {img_path}: {e}")

def parse_split(split, SPLIT_FILES, RESIZED_DIR, JSONL_DIR):
    with open(SPLIT_FILES[split]['questions'], 'r') as fq:
        questions = json.load(fq)
    with open(SPLIT_FILES[split]['answers'], 'r') as fa:
        answers = json.load(fa)
    answer_map = {a['id']: a['answer'] if 'answer' in a else a.get('text', '') for a in answers}
    out_path = os.path.join(JSONL_DIR, f'{split}.jsonl')
    with open(out_path, 'w', encoding='utf-8') as fout:
        for q in tqdm(questions, desc=f'Processing {split} split'):
            img_id = q.get('img_id')
            if img_id is None:
                print(f"[Warning] Missing img_id in question: {q}")
                continue
            image_file = f'{img_id}.jpg'
            question = q.get('question', '')
            answer_ids = q.get('answers_ids', [])
            if not answer_ids:
                print(f"[Warning] No answer_ids for question: {q}")
                continue
            answer = answer_map.get(answer_ids[0], '') if answer_ids else ''
            if not answer:
                print(f"[Warning] No answer found for answer_id {answer_ids[0]} in question: {q}")
                continue
            entry = {
                'image': os.path.join(RESIZED_DIR, image_file),
                'question': question,
                'answer': answer
            }
            fout.write(json.dumps(entry) + '\n')

def main():
    IMAGES_DIR = 'RSVQA-LR Dataset/Images_LR/Images_LR/'
    RESIZED_DIR = 'RSVQA-LR Dataset/Images_LR/images_256/'
    SPLITS = ['train', 'val', 'test']
    SPLIT_FILES = {
        'train': {
            'questions': 'RSVQA-LR Dataset/LR_split_train_questions.json',
            'answers': 'RSVQA-LR Dataset/LR_split_train_answers.json',
            'images': 'RSVQA-LR Dataset/LR_split_train_images.json',
        },
        'val': {
            'questions': 'RSVQA-LR Dataset/LR_split_val_questions.json',
            'answers': 'RSVQA-LR Dataset/LR_split_val_answers.json',
            'images': 'RSVQA-LR Dataset/LR_split_val_images.json',
        },
        'test': {
            'questions': 'RSVQA-LR Dataset/LR_split_test_questions.json',
            'answers': 'RSVQA-LR Dataset/LR_split_test_answers.json',
            'images': 'RSVQA-LR Dataset/LR_split_test_images.json',
        },
    }
    JSONL_DIR = 'RSVQA-LR Dataset/jsonl/'
    os.makedirs(RESIZED_DIR, exist_ok=True)
    os.makedirs(JSONL_DIR, exist_ok=True)
    resize_images(IMAGES_DIR, RESIZED_DIR)
    for split in SPLITS:
        parse_split(split, SPLIT_FILES, RESIZED_DIR, JSONL_DIR)
    print('RSVQA-LR dataset preparation complete!')

if __name__ == '__main__':
    main() 