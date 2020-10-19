import torch
import torch.nn as nn

class Entity_Aware_Embedding(nn.Module):
    def __init__(self, word_embedding, pos_dim=5, pos_len=100):
        super(Entity_Aware_Embedding, self).__init__()
        self.word_embedding = nn.Embedding.from_pretrained(word_embedding, freeze=False, padding_idx=-1)
        self.pos1_embedding = nn.Embedding(2 * pos_len + 1, pos_dim)
        self.pos2_embedding = nn.Embedding(2 * pos_len + 1, pos_dim)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.pos1_embedding.weight)
        nn.init.xavier_uniform_(self.pos2_embedding.weight)

    def forward(self, X, X_Pos1, X_Pos2, X_Ent1, X_Ent2):
        X = self.word_embedding(X)
        Xp = self.word_pos_embedding(X, X_Pos1, X_Pos2)
        Xe = self.word_ent_embedding(X, X_Ent1, X_Ent2)
        return Xp, Xe

    def word_pos_embedding(self, X, X_Pos1, X_Pos2):
        X_Pos1 = self.pos1_embedding(X_Pos1)
        X_Pos2 = self.pos2_embedding(X_Pos2)
        return torch.cat([X, X_Pos1, X_Pos2], -1)

    def word_ent_embedding(self, X, X_Ent1, X_Ent2):
        X_Ent1 = self.word_embedding(X_Ent1).unsqueeze(1).expand(X.shape)
        X_Ent2 = self.word_embedding(X_Ent2).unsqueeze(1).expand(X.shape)
        return torch.cat([X, X_Ent1, X_Ent2], -1)
