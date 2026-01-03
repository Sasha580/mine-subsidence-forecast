import torch


def train(model, train_loader, optimizer, loss_fn, device, num_epochs, val_loader):
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, target in train_loader:
            inputs, target = inputs.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(inputs)

            loss = loss_fn(output, target)
            loss.backward()  # Compute the gradients of the loss
            optimizer.step()  # Adjust learning weights

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {epoch_loss:.5f} | Val Loss: {val_loss:.5f}"
        )
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
    return train_losses, val_losses


def evaluate(model, test_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    num_samples = 0

    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)
            total_loss += loss_fn(model(inputs), target).item()
            num_samples += 1

    return total_loss / num_samples


def predict(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    all_preds = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            preds = model(inputs)
            all_preds.append(preds.cpu())

    return torch.cat(all_preds, dim=0)  # tensor [pred_num, mine_len]