from PIL import Image
from torchvision import transforms


to_tensor = transforms.Compose([ 
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

def preprocess(image_path: str): 
    image = Image.open(image_path)
    image_tensor = to_tensor(image) 
    return image_tensor.unsqueeze(dim=0)

if __name__ == "__main__": 
    sample = preprocess("cat.png")
    print(sample.size())
