import torch
import lightning as L 
import torch.nn.functional as F

class LitAutoencoder(L.LightningModule):
    def __init__(self, autoencoder, lr):
        super().__init__()
        self.autoencoder = autoencoder
        self.lr = lr
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.autoencoder(x)
        loss = F.mse_loss(output, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.autoencoder(x)
        val_loss = F.mse_loss(output, y)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer 

    def on_epoch_end(self):
        train_loss = self.trainer.logged_metrics['loss']
        print(f"Epoch {self.current_epoch}: Train Loss - {train_loss}")


if __name__ == "__main__":
    from lightning.pytorch.utilities.model_summary import ModelSummary
    from autoencoder import get_model

    autoencoder_model = get_model()
    model = LitAutoencoder(autoencoder_model)
    summary = ModelSummary(model)
    print(summary)