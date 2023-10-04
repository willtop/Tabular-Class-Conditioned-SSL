import torch
from torch.utils import DataLoader
from scarf import Contrastive_Model, NTXent


# preprocess your data and create your pytorch dataset
# train_ds = ...

# train the model
batch_size = 128
epochs = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

model = Contrastive_Model(
    input_dim=train_ds.shape[1],
    emb_dim=16,
    corruption_rate=0.5,
).to(device)
optimizer = Adam(model.parameters(), lr=0.001)
ntxent_loss = NTXent()

for epoch in range(1, epochs + 1):
  for anchor, positive in train_loader:
        anchor, positive = anchor.to(device), positive.to(device)

        # reset gradients
        optimizer.zero_grad()

        # get embeddings
        emb, emb_corrupted = model(anchor, positive)

        # compute loss
        loss = ntxent_loss(emb, emb_corrupted)
        loss.backward()

        # update model weights
        optimizer.step()