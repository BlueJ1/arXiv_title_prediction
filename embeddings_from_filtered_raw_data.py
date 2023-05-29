import torch
from transformers import AutoTokenizer, AutoModel

import pandas as pd
import numpy as np
from accelerate import Accelerator

from tqdm import tqdm
from time import time
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # suppresses huggingface warning


def embeddings_from_filtered_raw_data(data_chunks_dir, n_chunks=None, embedding_chunks_dir="abstract_embeddings",
                                      model_name="intfloat/e5-large-v2"):
    if n_chunks is None:
        chunk_filenames = [data_chunks_dir + "/chunk_" + str(i) for i in range(len(os.listdir(data_chunks_dir)) - 1)]
    else:
        chunk_filenames = [data_chunks_dir + "/chunk_" + str(i) + ".csv" for i in range(n_chunks)]

    print("Downloading...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    print("Downloaded.")
    if torch.backends.mps.is_available():
        model.to("mps")
    elif torch.cuda.is_available():
        print("converting model to cuda")
        model.to("cuda")
    model = Accelerator().prepare(model)

    if not os.path.exists(embedding_chunks_dir):
        os.mkdir(embedding_chunks_dir)

    for chunk_filename in chunk_filenames:
        t = time()
        chunk_df = pd.read_csv(chunk_filename)["abstract"].tolist()
        tokenized_dict = tokenizer(chunk_df, max_length=250, padding=True, truncation=True, return_tensors='pt')
        if torch.backends.mps.is_available():
            tokenized_dict.to("mps")
        elif torch.cuda.is_available():
            print("converting tokenized_dict to cuda")
            tokenized_dict.to("cuda")
        tokenized_dict = Accelerator().prepare(tokenized_dict)

        embeddings = torch.empty((len(chunk_df), 250, 1024), dtype=torch.float32)
        data_length = len(chunk_df)
        batch_size = 128
        n_batches = data_length // batch_size
        max_batches = min(n_batches, np.inf)

        for i in tqdm(range(max_batches)):
            batch_dict = {k: v[i * batch_size: (i + 1) * batch_size] for k, v in tokenized_dict.items()}

            outputs = model(**batch_dict)
            embedding = outputs.last_hidden_state.masked_fill(~batch_dict["attention_mask"][..., None].bool(), 0.0)
            embeddings[i * batch_size: (i + 1) * batch_size] = embedding.cpu().detach()

        np.save(embedding_chunks_dir + "/" + chunk_filename.split("/")[-1].split(".")[0] + ".npy", embeddings)
        print(f'total time for this chunk: {time() - t}')


if __name__ == "__main__":
    embeddings_from_filtered_raw_data("data_chunks", n_chunks=1, embedding_chunks_dir="abstract_embeddings")
