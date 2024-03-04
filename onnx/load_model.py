import cv2
import torch
import numpy as np
from utils import *
from model import get_model

model = get_model()

def load_params(model, ckpt_file):
    model_params = model.state_dict()
    checkpoint = torch.load(ckpt_file, map_location=torch.device('cpu'))
    state_dicts = checkpoint['state_dict']

    new_params = {}
    for key, value in zip(model_params.keys(), state_dicts.values()): 
        new_params[key] = value

    model_params.update(new_params)
    model.load_state_dict(model_params)

    return model

if __name__ == "__main__":

    model = get_model()
    model = load_params(model, "epoch=9-loss=0.00.ckpt")


    # Testing 
    image_input = preprocess("cat.png") 
    image = image_input.squeeze(dim=0).permute(1,2,0).numpy()

    model.eval()
    with torch.inference_mode(): 
        y = model(image_input) 
        y = y.squeeze(dim=0).permute(1,2,0).numpy()
    
    show = np.concatenate(
        [ cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.cvtColor(y, cv2.COLOR_BGR2RGB)], 
        axis=1
    )

    cv2.imshow("", show)
    cv2.waitKey()