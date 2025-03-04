from genXORdataset import XORDataset
from getModel import Classifier
import torch
import torch.nn as nn
import torch.utils.data as data

def train_loop(model, optimizer, dataloader, loss_fn):
    model.train()
    size = len(dataloader.dataset)
    device = torch.device("cuda")
    for batch, (X, y) in enumerate(dataloader):
        X_cuda = X.to(device)
        y_cuda = y.to(device)
        pred = model(X_cuda)
        pred_ = pred.squeeze(dim=1)
        loss = loss_fn(pred_, y_cuda.float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        loss, current = loss.item(), batch * 128 + len(X_cuda)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



    






if __name__ == "__main__":
    # preparing dataset
    train_dataset = XORDataset(size=2500)
    train_dataloader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    test_dataset = XORDataset(size=500)
    test_dataloader = data.DataLoader(test_dataset, batch_size=128, shuffle=True)

    # device
    device = torch.device("cuda")
    print(f"Using {device} device")

    model = Classifier(num_inputs=2, num_hidden=4, num_outputs=1)
    model.to(device)
    print(model)


    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(model=model, optimizer=optimizer, dataloader=train_dataloader,loss_fn=loss_fn )









