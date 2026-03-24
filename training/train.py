import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from data.dataset_loader import load_data
from preprocessing.smiles_to_fp import smiles_to_fp
from preprocessing.protein_encoder import encode_protein
from models.dti_model import Final_DTI_Model

def prepare_dataset(df):

    dataset = []

    for _,row in df.iterrows():

        d = smiles_to_fp(row['Drug'])
        p = encode_protein(row['Target'])

        if d is not None:
            dataset.append((d,p,row['interaction']))

    return dataset


def train():

    df = load_data()
    dataset = prepare_dataset(df)

    train_data, test_data = train_test_split(dataset, test_size=0.2)

    model = Final_DTI_Model()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):

        total_loss = 0

        for d,p,label in train_data:

            optimizer.zero_grad()

            pred = model(d,p)

            loss = criterion(pred, torch.tensor([label], dtype=torch.float))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved!")

if __name__ == "__main__":
    train()
