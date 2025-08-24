"""Training script for the PriceRNN model.

Example usage:
    python policy/model/train_rnn.py --data /path/to/market --ts-len 30 --epochs 10
"""

import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import trange


def train(args: argparse.Namespace) -> None:
    try:  # pragma: no cover - allow running as script or module
        from ..datasets.HugeStockMarket import HugeStockMarketDataset
        from .rnn_model import PriceRNN
    except ImportError:  # pragma: no cover
        from policy.datasets.HugeStockMarket import HugeStockMarketDataset
        from rnn_model import PriceRNN

    dataset = HugeStockMarketDataset(args.data, ts_len=args.ts_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = PriceRNN(hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ranger = trange(1, args.epochs + 1, desc="Training")
    for epoch in ranger:
        model.train()
        total_loss = 0.0
        for batch in loader:
            x = batch["data"].to(device)
            y_cls = batch["class"].to(device)
            y_reg = batch["regr"].to(device)

            optimizer.zero_grad()
            pred_cls, pred_reg, _ = model(x)
            
            criterrino_cls  = criterion_cls(pred_cls, y_cls)
            criterrino_reg  = criterion_reg(pred_reg, y_reg)
            adaptive_weight = (criterrino_cls / criterrino_reg).detach()

            loss = criterrino_cls + criterrino_reg * adaptive_weight
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataset)
        ranger.set_description(f"Epoch {epoch} - Loss: {avg_loss:.4f}")
        # ranger.update(1)
        
        if epoch - 1 % 25 == 0:  # Save every 10 epochs
            output_path = Path(args.output).parent / f'epoch_{epoch}.pt'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            # print(f"Model saved to {output_path}")

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)
        # print(f"Model saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PriceRNN policy")
    parser.add_argument("--data", required=True, help="Path to dataset root")
    parser.add_argument("--ts_len", type=int, default=30, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--hidden_size", type=int, default=64, help="RNN hidden size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of RNN layers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", default="price_rnn.pt", help="Where to save trained weights")
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover - script entry point
    train(parse_args())

