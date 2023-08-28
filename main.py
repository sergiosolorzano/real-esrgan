import os 
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN


def main() -> int:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:",device)
    
    model = RealESRGAN(device, scale=4)
    model.load_weights('weights/RealESRGAN_x4.pth', download=True)
    target_dir=os.path.join(os.getcwd(),"inputs")

    for root, dirs, files in os.walk(target_dir):
        for f in files:
            fp = os.path.join(root,f)
            if len(fp.split(".")) == 2 and fp.split(".")[1]=="png":
                print("opening file",fp)
                image = Image.open(fp).convert('RGB')
                sr_image = model.predict(image)
                sr_image.save(f'results/{f}')

if __name__ == '__main__':
    main()