import argparse

# Model
import lightning as L
from data.dataset import get_data
from model.autoencoder import get_model 
from model.litmodule import LitAutoencoder

# Callbacks & Tracking
from callbacks import checkpoint_callback 
from log.wandb_config import wandb_logger



def main(batch_size: int, epochs: int, device: str|int, lr: float):

    train_loader, valid_loader = get_data(batch_size=batch_size)
    autoencoder_model = get_model()
    model = LitAutoencoder(autoencoder_model, lr=lr) 

    trainer = L.Trainer(
        max_epochs=epochs, 
        callbacks=[checkpoint_callback], 
        logger=wandb_logger,
        devices=device
        )

    trainer.fit(
        model=model, 
        train_dataloaders=train_loader, 
        val_dataloaders=valid_loader
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--batch-size", type=int, default=8
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=3
    )
    parser.add_argument(
        "-d", "--device", default="auto"
    )
    parser.add_argument(
        "-l", "--learning-rate", type=float, default=0.005
    )
    args = parser.parse_args() 

    main(
        batch_size=args.batch_size, 
        epochs=args.epochs, 
        device=args.device, 
        lr=args.learning_rate
    )
