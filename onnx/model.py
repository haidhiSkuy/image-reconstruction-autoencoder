from torch import nn 
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential( 
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.bottleneck = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.latent_space = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, padding=1)

        
    def forward(self, x):
        x = self.encoder(x) 
        encoded = F.relu(self.bottleneck(x))
        latent_space = F.sigmoid(self.latent_space(encoded)) #for visualization
        return encoded, latent_space

    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x
    

class AutoEncoder(nn.Module):
    def __init__(self, encoder:nn.Module, decoder:nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        x, _ = self.encoder(x)
        x = self.decoder(x)
        return x 
    

def get_model(): 
    encoder = Encoder()
    decoder = Decoder()
    autoencoder = AutoEncoder(encoder, decoder)
    return autoencoder
