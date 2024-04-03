import os
import h5py
import torch
from transformers import BertTokenizer, BertModel
import argparse

def encode_captions_with_bert(captions, model, tokenizer):
    # Tokenize and encode the captions
    encoded_inputs = tokenizer(captions, padding=True, truncation=True, return_tensors='pt', max_length=128)
    with torch.no_grad():
        outputs = model(**encoded_inputs)
    # Use the pooled output by default; you could also use mean pooling on the last hidden state
    embeddings = outputs.pooler_output
    return embeddings.cpu().numpy()

def save_caption_vectors_flowers(data_dir, model, tokenizer):
    img_dir = os.path.join(data_dir, 'flowers/jpg')
    image_files = [f for f in os.listdir(img_dir) if 'jpg' in f]
    image_captions = {img_file: [] for img_file in image_files}

    caption_dir = os.path.join(data_dir, 'flowers/text_c10')
    for i in range(1, 103):
        class_dir_name = 'class_%.5d' % i
        class_dir = os.path.join(caption_dir, class_dir_name)
        caption_files = [f for f in os.listdir(class_dir) if 'txt' in f]
        for cap_file in caption_files:
            with open(os.path.join(class_dir, cap_file)) as f:
                captions = f.read().strip().split('\n')
            img_file = cap_file[0:11] + ".jpg"
            image_captions[img_file].extend(captions[:5])

    encoded_captions = {}

    for i, img_file in enumerate(image_captions):
        print(f"Processing {img_file} {i+1}/{len(image_captions)}")
        encoded_captions[img_file] = encode_captions_with_bert(image_captions[img_file], model, tokenizer)

    h = h5py.File(os.path.join(data_dir, 'flower_tv.hdf5'), 'w')
    for img_file, emb in encoded_captions.items():
        h.create_dataset(img_file, data=emb)
    h.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='Data', help='Data directory')
    parser.add_argument('--data_set', type=str, default='flowers', help='Data Set: Flowers, MS-COCO')
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    if args.data_set == 'flowers':
        save_caption_vectors_flowers(args.data_dir, model, tokenizer)
    # Add more datasets as needed

if __name__ == '__main__':
    main()
