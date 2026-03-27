from pathlib import Path
import yaml
import torch
import matplotlib.pyplot as plt

from src.utils import set_seed, ensure_dirs
from src.dataset import build_dataloaders
from src.model import SimpleMLP
from src.pipeline import train_one_epoch, evaluate


CONFIG_PATH = Path("configs/smoke_test.yaml")
LOG_PATH = Path("outputs/logs/smoke_test_log.txt")
FIG_PATH = Path("outputs/figures/loss_curve.png")
CKPT_PATH = Path("outputs/checkpoints/smoke_model.pt")


def main():
    ensure_dirs()

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader = build_dataloaders(
        n_samples=cfg["n_samples"],
        n_features=cfg["n_features"],
        n_classes=cfg["n_classes"],
        test_size=cfg["test_size"],
        batch_size=cfg["batch_size"],
        seed=cfg["seed"],
    )

    model = SimpleMLP(
        input_dim=cfg["n_features"],
        hidden_dim=cfg["hidden_dim"],
        num_classes=cfg["n_classes"],
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    train_losses = []
    test_losses = []
    test_accuracies = []

    log_lines = []
    log_lines.append("=== CSC4005 SMOKE TEST ===")
    log_lines.append(f"Device: {device}")
    log_lines.append(f"Config: {cfg}")

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        line = (
            f"Epoch {epoch}/{cfg['epochs']} | "
            f"train_loss={train_loss:.4f} | "
            f"test_loss={test_loss:.4f} | "
            f"test_acc={test_acc:.4f}"
        )
        log_lines.append(line)
        print(line)

    torch.save(model.state_dict(), CKPT_PATH)

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, cfg["epochs"] + 1), train_losses, label="train_loss")
    plt.plot(range(1, cfg["epochs"] + 1), test_losses, label="test_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Smoke Test Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_PATH)
    plt.close()

    log_lines.append(f"Checkpoint saved to: {CKPT_PATH}")
    log_lines.append(f"Figure saved to: {FIG_PATH}")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")

    print(f"\nSaved log to: {LOG_PATH}")
    print(f"Saved figure to: {FIG_PATH}")
    print(f"Saved checkpoint to: {CKPT_PATH}")


if __name__ == "__main__":
    main()
