import torch
import torch.nn as nn
import math
from typing import Dict
Tensor = torch.Tensor



class AttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttentionModel, self).__init__()
        self.name = "test_model_attention"
        self.n_dims = input_dim + config["embedding_dim"] - 1

        # embedding:
        self.embedding = nn.Embedding(config["categorical_size"], config["embedding_dim"])
        
        # Linear layer to process the input sequence into hidden features
        self.fc = nn.Linear(self.n_dims, self.n_dims)
        self.relu = nn.ReLU()
        

        self.attention = nn.MultiheadAttention(
            embed_dim=self.n_dims,
            num_heads=config["num_heads"],
            dropout=config["dropout"],
            batch_first=True,
        )

        # Attention scoring mechanism
        self.Linear1 = nn.Linear(self.n_dims, hidden_dim)

        # Output layer to map attention-weighted input to output classes
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_length, input_dim)
        """
        

        #embedding:
        position_categorical = 1

        x_cat = x[:, :, position_categorical].long()
        x_cat = self.embedding(x_cat)

        all_features = torch.arange(x.size(2))  # All feature indices [0, 1, 2, ..., 7]
        features_to_keep = all_features[all_features != position_categorical]
        x_num = torch.index_select(x, dim=2, index=features_to_keep)

        x = torch.cat((x_num, x_cat), dim=2)

        # Step 1: Transform input using the first layer
        x = self.fc(x)  # (batch_size, seq_length, hidden_dim)
        x = self.relu(x)

        x , _ = self.attention(x, x, x)
        

        # Step 2: Compute attention scores
        x = self.Linear1(x)  # (batch_size, seq_length, 1)
        x = self.relu(x)
        # scores = torch.softmax(scores, dim=1)  # Normalize scores over sequence length
        

        # Step 3: Compute the weighted sum of inputs using attention scores
        x = x[:, -1, :]
        # x = torch.mean(x, dim=1)
        # x.flatten(start_dim=1)

        # Step 4: Map the weighted sum to output classes
        output = self.output_layer(x)  # (batch_size, output_dim)
        output = torch.sigmoid(output)

        return output
