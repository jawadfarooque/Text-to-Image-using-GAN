import os
from os.path import join, isfile
import numpy as np
import h5py
import argparse
import torch
from transformers import BertTokenizer, BertModel

# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--caption_file', type=str, default='Data/sample_caption.txt',
                    help='caption file')
parser.add_argument('--data_dir', type=str, default='Data',
                    help='Data Directory')

# Load pre-trained model tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def bert_encode(texts):

    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    sentence_embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    normalized_embeddings = sentence_embeddings / np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
    return normalized_embeddings

def main():
    args = parser.parse_args()
    with open(args.caption_file) as f:
        captions = f.read().split('\n')
    captions = [cap for cap in captions if len(cap) > 0]
    caption_vectors = bert_encode(captions)
    hdf5_path = join(args.data_dir, 'sample_caption_vectors.hdf5')
    if os.path.isfile(hdf5_path):
        os.remove(hdf5_path)
    with h5py.File(hdf5_path, 'w') as h:
        h.create_dataset('vectors', data=caption_vectors)
    print(f"Caption vectors saved to {hdf5_path}")

if __name__ == '__main__':
    main()
