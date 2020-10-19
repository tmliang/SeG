import torch
import torch.nn as nn
import numpy as np
from .Embedding import Entity_Aware_Embedding
from .Encoder import PCNN, SAN

class SeG(nn.Module):
    def __init__(self, pre_word_vec, rel_num, lambda_pcnn=0.05, lambda_san=1.0, pos_dim=5, pos_len=100, hidden_size=230, dropout_rate=0.5):
        super(SeG, self).__init__()
        word_embedding = torch.from_numpy(np.load(pre_word_vec))
        word_dim = word_embedding.shape[-1]
        self.embedding = Entity_Aware_Embedding(word_embedding, pos_dim, pos_len)
        self.PCNN = PCNN(word_dim, pos_dim, lambda_pcnn, hidden_size)
        self.SAN = SAN(word_dim, pos_dim, lambda_san)
        self.fc1 = nn.Linear(3 * word_dim, 3 * word_dim)
        self.fc2 = nn.Linear(3 * word_dim, 3 * hidden_size)
        self.classifer = nn.Linear(3 * hidden_size, rel_num)
        self.dropout = nn.Dropout(dropout_rate)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.classifer.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.classifer.bias)

    def forward(self, X, X_Pos1, X_Pos2, X_Ent1, X_Ent2, X_Mask, X_Scope):
        # Embed
        Xp, Xe = self.embedding(X, X_Pos1, X_Pos2, X_Ent1, X_Ent2)
        # Encode
        S = self.PCNN(Xp, Xe, X_Mask)
        U = self.SAN(Xp, Xe)
        # Combine
        X = self.selective_gate(S, U, X_Scope)
        X = self.dropout(X)
        # Classifier
        X = self.classifer(X)
        return X

    def selective_gate(self, S, U, X_Scope):
        G = torch.sigmoid(self.fc2(torch.tanh(self.fc1(U))))
        X = G * S
        B = []  # Bag Output
        for s in X_Scope:
            B.append(X[s[0]:s[1]].mean(0))
        B = torch.stack(B)
        return B