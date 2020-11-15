import torch
import torch.nn as nn


class PCNN(nn.Module):
    def __init__(self, word_dim, pos_dim, lam, hidden_size=230):
        super(PCNN, self).__init__()
        mask_embedding = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        self.mask_embedding = nn.Embedding.from_pretrained(mask_embedding)
        self.cnn = nn.Conv1d(3 * word_dim, hidden_size, 3, padding=1)
        self.fc1 = nn.Linear(3 * word_dim, 3 * word_dim)
        self.fc2 = nn.Linear(2 * pos_dim + word_dim, 3 * word_dim)
        self.hidden_size = hidden_size
        self.lam = lam
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.cnn.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.cnn.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, Xp, Xe, X_mask):
        A = torch.sigmoid((self.fc1(Xe / self.lam)))
        X = A * Xe + (1 - A) * torch.tanh(self.fc2(Xp))
        X = self.cnn(X.transpose(1, 2)).transpose(1, 2)
        X = self.pool(X, X_mask)
        X = torch.tanh(X)
        return X

    def pool(self, X, X_mask):
        X_mask = self.mask_embedding(X_mask)
        X = torch.max(torch.unsqueeze(X_mask, 2) * torch.unsqueeze(X, 3), 1)[0]
        return X.view(-1, self.hidden_size * 3)


class SAN(nn.Module):
    def __init__(self, word_dim, pos_dim, lam):
        super(SAN, self).__init__()
        self.fc1 = nn.Linear(3 * word_dim, 3 * word_dim)
        self.fc2 = nn.Linear(2 * pos_dim + word_dim, 3 * word_dim)
        self.fc1_att = nn.Linear(3 * word_dim, 3 * word_dim)
        self.fc2_att = nn.Linear(3 * word_dim, 3 * word_dim)
        self.lam = lam
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc1_att.weight)
        nn.init.xavier_uniform_(self.fc2_att.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc1_att.bias)
        nn.init.zeros_(self.fc2_att.bias)

    def forward(self, Xp, Xe):
        # embedding
        A = torch.sigmoid((self.fc1(Xe / self.lam)))
        X = A * Xe + (1 - A) * torch.tanh(self.fc2(Xp))
        # encoder
        A = self.fc2_att(torch.tanh(self.fc1_att(X)))
        P = torch.softmax(A, 1)
        X = torch.sum(P * X, 1)
        return X