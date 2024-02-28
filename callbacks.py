from lightning.pytorch.callbacks import ModelCheckpoint 


checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoint',
    filename='{epoch}-{loss:.2f}', 
    monitor="val_loss", 
    save_on_train_epoch_end=True,
    save_top_k=1,
    mode='min'
)