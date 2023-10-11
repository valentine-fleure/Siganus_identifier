from pathlib import Path
from posixpath import split
import torch, torchvision
import torch.nn as nn
from torchvision import models, transforms
import os
import argparse
from PIL import Image
from torch import optim, cuda
import matplotlib
matplotlib.use('TkAgg')

def load_checkpoint(path):
    """Load a PyTorch model checkpoint

    Params
    --------
        path (str): saved model checkpoint. Must start with `model_name-` and end in '.pth'

    Returns
    --------
        None, save the `model` to `path`

    """

    # Get the model name
    model_name = path.split('-')[0]
    assert (model_name in ['resnet50'
                           ]), "Path must have the correct model name"

    # Load in checkpoint
    checkpoint = torch.load(path, map_location=torch.device('cpu'))

    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        # Make sure to set parameters as not trainable
        for param in model.parameters():
            param.requires_grad = False
        model.fc = checkpoint['fc']

    # Load in the state dict
    model.load_state_dict(checkpoint['state_dict'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} total gradient parameters.')

    model = model.to('cpu')

    # Model basics
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    model.epochs = checkpoint['epochs']

    return model

def predict(image_path, model, topk=5):
    """Make a prediction for an image using a trained model

    Params
    --------
        image_path (str): filename of the image
        model (PyTorch model): trained model for inference
        topk (int): number of top predictions to return

    Returns
        
    """

    # Convert to pytorch tensor
    img_tensor, surface = process_image(image_path)


    img_tensor = img_tensor.view(1, 3, 224, 224)

    # Set to evaluation
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(img_tensor)
        ps = torch.exp(out)

        # Find the topk predictions
        topk, topclass = ps.topk(topk, dim=1)

        # Extract the actual classes and probabilities
        top_classes = [
            model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]
        ]
        top_p = topk.cpu().numpy()[0]

        return surface, top_p, top_classes

def process_image(image_path):
    """Process an image path into a PyTorch tensor"""

    image = Image.open(image_path)

    w, h = image.size
    surface = w * h
    # Define a transform to convert the image to tensor
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Convert the image to PyTorch tensor
    img_tensor = transform(image)

    return img_tensor, surface

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = 9

    checkpoint_path = "resnet50_siganus_identifier"

    model = models.resnet50(pretrained=True)
    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False
    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
                        nn.Linear(n_inputs, 256), 
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, n_classes),        
                        nn.LogSoftmax(dim=1))


    model = model.to(device)
    model = load_checkpoint(path=checkpoint_path)

    dir = "/home/vfleure/Vidéos/2022_11_06_camF1_raie_manta_2_GP230222/imagettes/"  #### A CHANGER
    img_list = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(dir)) for f in fn]

    res_file = dir + "/" + "res.csv"
    with open(res_file, "w") as csv_file :
        csv_file.write("img_name,img_area,pred,score\n")
        for im in img_list:
            surface, top_p, top_classes = predict(im, model, topk=1)
            img_name = im.split('/')[-1]
            csv_file.write(img_name+","+str(surface)+','+top_classes[0]+","+str(top_p[0])+"\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', action='store', required=True)
    opt = parser.parse_args()
    main(Path(opt.video)
