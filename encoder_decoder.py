import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from evaluate import load
from typing import List
bertscore = load("bertscore")


class CustomDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        return self.embeddings[index]


class EncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_heads, dropout):
        super(EncoderDecoder, self).__init__()

        self.encoder = nn.TransformerEncoderLayer(input_size, num_heads, dim_feedforward=hidden_size, dropout=dropout)
        self.decoder = nn.TransformerDecoderLayer(hidden_size, num_heads, dim_feedforward=hidden_size, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        x = x + self.positional_encoding
        x = x.permute(1, 0, 2)  # Reshape for transformer

        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output, x)
        output = self.fc(decoder_output)
        return output.permute(1, 0, 2)  # Reshape back to (batch_size, seq_len, hidden_size)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
embedding_dim = 768
hidden_dim = 256
output_dim = 30
batch_size = 32
num_epochs = 10
learning_rate = 0.001
num_layers = 2
num_heads = 4
dropout = 0.1

# Prepare data
embeddings = ...  # Your (N, 250, 768) embeddings data
dataset = CustomDataset(embeddings)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model and optimizer
model = EncoderDecoder(embedding_dim, hidden_dim, output_dim, num_layers, num_heads, dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def calculate_bert_score(predictions: List[str], references: List[str])
    results = bertscore.compute("predictions"=predictions, "references"=references, model_type="distilbert-base-uncased")
    return -torch.mean(results)


# Training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0

    for embeddings_batch in dataloader:
        embeddings_batch = embeddings_batch.to(device)

        optimizer.zero_grad()

        predictions = model(embeddings_batch)

        targets = embeddings_batch[:, -output_dim:, :]

        loss = calculate_bert_score(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "encoder_decoder_model.pt")
