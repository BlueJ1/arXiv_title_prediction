import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

import pandas as pd
import numpy as np

from time import time
from tqdm import tqdm

import accelerate
accelerator = accelerate.Accelerator()


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


print("loading data...")
arxiv_df = pd.read_csv("arxiv_first_100k.csv")
print("loaded data")

# Each input text should start with "query: " or "passage: ".
# For tasks other than retrieval, you can simply use the "query: " prefix.
input_texts = ['query: how much protein should a female eat',
               'query: summit define',
               "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
               "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."]

tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
model = AutoModel.from_pretrained('intfloat/e5-base-v2')
model.to("mps")

outputs_list = []
data_length = arxiv_df.shape[0]
batch_size = 64
n_batches = data_length // batch_size
max_batches = min(n_batches, 10)

for i in tqdm(range(max_batches)):
    batch_data = arxiv_df["title"].iloc[i * batch_size: (i + 1) * batch_size].to_list()
    # print(len(batch_data))
    # t = time()
    # Tokenize the input texts
    batch_dict = tokenizer(batch_data, max_length=512, padding=True, truncation=True, return_tensors='pt')
    batch_dict.to("mps")
    # print(batch_dict["input_ids"].shape)

    outputs = model(**batch_dict)
    outputs_list.append(outputs.last_hidden_state.cpu().detach().numpy())
    # outputs_list.append(outputs)
    # print(outputs.last_hidden_state.shape)
    # embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # (Optionally) normalize embeddings
    # embeddings = F.normalize(embeddings, p=2, dim=1)
    # scores = (embeddings[:2] @ embeddings[2:].T) * 100
    # print(scores.tolist())
    # print(time() - t)

outputs_list = np.concatenate(outputs_list)
outputs_list = np.array(outputs_list)
print(outputs_list.shape)
