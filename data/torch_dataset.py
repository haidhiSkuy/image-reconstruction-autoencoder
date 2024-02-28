from PIL import Image
from torch.utils.data import Dataset 


class CatsDataset(Dataset): 
    def __init__(self, image_dataframe, transform): 
        self.image_dataframe = image_dataframe
        self.transform = transform 
    
    def __len__(self):
        return len(self.image_dataframe)
    
    def __getitem__(self, idx):
        input_image = self.image_dataframe.iloc[idx, 0]
        x = Image.open(input_image)
        
        output_image = self.image_dataframe.iloc[idx, 1]
        y = Image.open(output_image)
        
        if self.transform: 
            x = self.transform(x)
            y = self.transform(y)
        
        return x, y