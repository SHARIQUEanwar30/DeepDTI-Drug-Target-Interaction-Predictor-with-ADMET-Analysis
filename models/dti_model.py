import torch
import torch.nn as nn

class Attention(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim,1)

    def forward(self,x):
        weights = torch.softmax(self.fc(x), dim=0)
        output = (weights * x).sum(dim=0)
        return output, weights


class Final_DTI_Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.drug_fc = nn.Linear(1024,256)
        self.embed = nn.Embedding(20,128)

        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.attn = Attention(128)

        self.fc1 = nn.Linear(256+128,128)
        self.fc2 = nn.Linear(128,1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, drug, protein):

        d = self.relu(self.drug_fc(drug))

        p = self.embed(protein).unsqueeze(1)
        p = self.transformer(p)

        p,_ = self.attn(p)

        combined = torch.cat((d, p.squeeze()), dim=0)

        out = self.relu(self.fc1(combined))
        out = self.sigmoid(self.fc2(out))

        return out
