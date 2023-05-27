import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import torch
import os


def generate_data_batches(filename, chunk_dir):
    """
    Takes the data provided in filename and sequentially reads it in 10k-sized chunks.
    These chunks are then filtered for rows where the abstract has between 100 and 250 tokens and the title has between
    10 and 30 tokens. The real tokenized sequence length may still vary, as this depends on specific tokenizer which is
    used. Default is the one for the e5-base-v2 model (so maybe could be rerun for and with different tokenizers - this
    requires a little bit of refactoring of the code).
    On average, 40-50% of the data is eliminated, leaving chunks of 4k to 5k to be written into a single file.

    :param filename: str; Reference to data file
    :param chunk_dir: str; Directory to store the filtered chunks in
    """
    if not os.path.isdir(chunk_dir):
        os.mkdir(chunk_dir)

    # I use the e5-base-v2 tokenizer to determine the token length for any piece of text
    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
    i = 0
    for chunk in pd.read_json(filename, lines=True, chunksize=10000):
        chunk: pd.DataFrame  # this does nothing - it's just type annotation for the IDE
        chunk = chunk[["title", "abstract"]]
        title_batch_dict = tokenizer(chunk["title"].tolist(), max_length=512, padding=True, truncation=True,
                                     return_tensors='pt')
        abstract_batch_dict = tokenizer(chunk["abstract"].tolist(), max_length=512, padding=True,
                                        truncation=True, return_tensors='pt')
        title_token_occurrences = torch.sum(title_batch_dict["attention_mask"], dim=1).numpy()
        abstract_token_occurrences = torch.sum(abstract_batch_dict["attention_mask"], dim=1).numpy()
        min_title_length = 10
        max_title_length = 30
        min_abstract_length = 100
        max_abstract_length = 250
        mask = (np.logical_and(np.logical_and(title_token_occurrences > min_title_length,
                                              title_token_occurrences < max_title_length),
                               np.logical_and(abstract_token_occurrences > min_abstract_length,
                                              abstract_token_occurrences < max_abstract_length)))
        chunk.drop(chunk.loc[mask].index, inplace=True)
        chunk.reset_index(inplace=True, drop=True)
        chunk.to_csv(f"{chunk_dir}/chunk_{i}.csv")
        print(f"wrote chunk {i}")
        i += 1


if __name__ == "__main__":
    generate_data_batches("arxiv_all_data.json", "data_chunks")
