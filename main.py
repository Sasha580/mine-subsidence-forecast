from torch.utils.data import DataLoader
from utils import *
from data import *
from model import *
from engine import *
from plotting import *


def main():
    set_seed(42)

    device = get_device()
    path = "approximated_data.csv"

    n_hist = 5
    train_ds, val_ds, test_ds, scaler = load_data(
        path,
        n_hist=n_hist,
        val_steps=1,
        test_steps=1,
        start_index=2,
    )

    batch_size = 1
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model = NextProfile1D(n_hist=n_hist).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss()
    num_epochs = 24

    train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        num_epochs=num_epochs,
        val_loader=val_loader,
    )

    preds = predict(model, test_loader, device)
    preds = scaler.decode(preds)

    # Attach last prediction as a new column in the original df
    df = pd.read_csv(path, delimiter=";", decimal=",")
    df["Pred_last"] = preds[-1].cpu().numpy()

    # Plot and save
    fig = plot_profiles(df)
    fig.show()
    fig.savefig("profiles.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()

