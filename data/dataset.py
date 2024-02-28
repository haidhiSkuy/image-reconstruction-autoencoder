import os
import pandas as pd
from torchvision import transforms
from .torch_dataset import CatsDataset
from torch.utils.data import DataLoader 
from sklearn.model_selection import train_test_split

inputs = []
outputs = [] 

features_dir = os.path.join(os.getcwd(), "data/cats/x")
labels_dir = os.path.join(os.getcwd(), "data/cats/y")

x = sorted(os.listdir(features_dir))
y = sorted(os.listdir(labels_dir))


for i, o in zip(x, y): 
    input_img = os.path.join(features_dir, i)
    output_img = os.path.join(labels_dir, o)
    
    inputs.append(input_img)
    outputs.append(output_img)

dataset = pd.DataFrame({"input":inputs, "output":outputs})
train, valid = train_test_split(dataset, test_size=0.2)

to_tensor = transforms.Compose([ 
    transforms.Resize((256,256)),
    transforms.ToTensor()
])


def get_data(batch_size: int = 16):
    train_dataset = CatsDataset(train, to_tensor)
    valid_dataset = CatsDataset(valid, to_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader
    