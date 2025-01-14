import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def freeze_first_five_layers(resnet_model):
    """
    Freeze the first 5 layers of a ResNet model.
    """
    child_counter = 0
    for child in resnet_model.children():
        if child_counter < 5:
            for param in child.parameters():
                param.requires_grad = False
        child_counter += 1

def initialize_fc_layer(fc_layer):
    """
    Initialize the final fully-connected (fc) layer.
    """
    if hasattr(fc_layer, 'weight') and fc_layer.weight is not None:
        nn.init.xavier_normal_(fc_layer.weight)
    if hasattr(fc_layer, 'bias') and fc_layer.bias is not None:
        nn.init.zeros_(fc_layer.bias)

def load_model(checkpoint_path, device):
    """
    Load the model with the checkpoint.
    """
    model = models.resnet50(weights=None)
    freeze_first_five_layers(model)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 4)
    initialize_fc_layer(model.fc)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    return model

def preprocess_image(image_path):
    """
    Preprocess the input image.
    """
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)

def predict(model, image_tensor, device):
    """
    Perform inference.
    """
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = output.argmax(dim=1).item()
    return predicted_class

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def run_model(image_path: str):
    """
    Run the model on the input image.
    """

    model_path = "checkpoints/best_checkpoint_epoch_35.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    image_tensor = preprocess_image(image_path)
    prediction = predict(model, image_tensor, device)
    return prediction

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inference for Custom ResNet50")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    args = parser.parse_args()

    prediction = run_model(args.image)
    print(f"Predicted Class ID: {prediction}")

if __name__ == '__main__':
    main()
