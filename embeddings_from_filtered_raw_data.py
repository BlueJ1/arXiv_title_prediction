import torch
from transformers import AutoTokenizer, AutoModel

import pandas as pd
import numpy as np

from tqdm import tqdm
import os


def embeddings_from_filtered_raw_data(data_chunks_dir, n_chunks=None, embedding_chunks_dir="abstract_embeddings",
                                      model_name="intfloat/e5-large-v2"):
    if n_chunks is None:
        chunk_filenames = [data_chunks_dir + "/" + filename for filename in sorted(os.listdir(data_chunks_dir))[1:]]
    else:
        chunk_filenames = [data_chunks_dir + "/chunk_" + str(i) + ".csv" for i in range(n_chunks)]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if torch.backends.mps.is_available():
        model.to("mps")
    elif torch.cuda.is_available():
        model.to("cuda")

    if not os.path.exists(embedding_chunks_dir):
        os.mkdir(embedding_chunks_dir)

    for chunk_filename in chunk_filenames:
        chunk_df = pd.read_csv(chunk_filename)["abstract"].tolist()
        tokenized_dict = tokenizer(chunk_df, max_length=250, padding=True, truncation=True, return_tensors='pt')
        if torch.backends.mps.is_available():
            tokenized_dict.to("mps")
        elif torch.cuda.is_available():
            tokenized_dict.to("cuda")

        embeddings = np.ndarray((len(chunk_df), 250, 1024), dtype=np.float16)
        data_length = len(chunk_df)
        batch_size = 8
        n_batches = data_length // batch_size
        max_batches = min(n_batches, np.inf)

        for i in tqdm(range(max_batches)):
            batch_dict = {k: v[i * batch_size: (i + 1) * batch_size] for k, v in tokenized_dict.items()}

            outputs = model(**batch_dict)
            embedding = outputs.last_hidden_state.masked_fill(~batch_dict["attention_mask"][..., None].bool(), 0.0)
            embeddings[i * batch_size: (i + 1) * batch_size] = embedding.cpu().detach().numpy()

        np.save(embedding_chunks_dir + "/" + chunk_filename.split("/")[-1].split(".")[0] + ".npy", embeddings)


if __name__ == "__main__":
    embeddings_from_filtered_raw_data("data_chunks", n_chunks=None, embedding_chunks_dir="abstract_embeddings")
