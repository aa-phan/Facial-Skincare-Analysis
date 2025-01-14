import argparse
import os
import wandb
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import DatasetProcessing


# TODO: Training: weighted random sampling

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def freeze_first_five_layers(resnet_model):
    """
    Freeze the first 5 layers of a ResNet model.
    In resnet50, the children typically are:
        0: conv1
        1: bn1
        2: relu
        3: maxpool
        4: layer1
        5: layer2
        6: layer3
        7: layer4
        8: avgpool
        9: fc
    This function will freeze layers 0..4 inclusive.
    """
    child_counter = 0
    for child in resnet_model.children():
        if child_counter < 5:  # freeze first 5 children
            for param in child.parameters():
                param.requires_grad = False
        child_counter += 1


def initialize_fc_layer(fc_layer):
    """
    Randomly initialize the final fully-connected (fc) layer weights & bias.
    Uses Xavier initialization (a common good practice).
    """
    if hasattr(fc_layer, 'weight') and fc_layer.weight is not None:
        nn.init.xavier_normal_(fc_layer.weight)
    if hasattr(fc_layer, 'bias') and fc_layer.bias is not None:
        nn.init.zeros_(fc_layer.bias)


def save_checkpoint(model, optimizer, epoch, save_path):
    """
    Save a checkpoint at the given path.
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, save_path)
    print(f"Checkpoint saved to {save_path}")


# -----------------------------------------------------------------------------
# Main training function
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train resnet50")
    parser.add_argument('--data_dir', type=str, default='data/fitzpatrick17k_data', help='Path to dataset')
    parser.add_argument('--csv_file', type=str, default='labels_fitzpatrick17k.csv', help='Labels CSV file')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--project_name', type=str, default='facial-skincare', help='WandB project name')
    parser.add_argument('--wandb_offline', action='store_true', help='Run wandb in offline mode')
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Initialize W&B (Weights & Biases)
    # -------------------------------------------------------------------------
    if args.wandb_offline:
        os.environ['WANDB_MODE'] = 'offline'  # force offline
    wandb.init(project=args.project_name, config=vars(args))

    # -------------------------------------------------------------------------
    # Prepare Dataset and DataLoaders
    # -------------------------------------------------------------------------
    dataset = DatasetProcessing()

    # Split dataset into training (80%) and validation (20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # -------------------------------------------------------------------------
    # Build resnet50, freeze first 5 layers, and randomly init final layer
    # -------------------------------------------------------------------------
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Freeze the first 5 layers
    freeze_first_five_layers(resnet50)

    # Replace the final layer with an output of size 4
    in_features = resnet50.fc.in_features
    resnet50.fc = nn.Linear(in_features, 4)

    # Randomly initialize the final layer
    initialize_fc_layer(resnet50.fc)

    # Although user mentioned "Ensure SoftMax for classification reasons":
    # Typically for training with CrossEntropyLoss, we do NOT add a Softmax layer.
    # CrossEntropyLoss expects raw logits. We can apply softmax at inference time if needed.

    # -------------------------------------------------------------------------
    # Define Loss, Optimizer
    # -------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    # Only train the unfrozen layers
    # (since the first 5 are frozen, only the rest + fc are trainable)
    trainable_parameters = (p for p in resnet50.parameters() if p.requires_grad)
    optimizer = optim.Adam(trainable_parameters, lr=args.lr)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet50.to(device)

    # Watch the model with wandb (tracks gradients, etc.)
    wandb.watch(resnet50, log="all")

    # -------------------------------------------------------------------------
    # Create checkpoints folder if not exists
    # -------------------------------------------------------------------------
    os.makedirs("checkpoints", exist_ok=True)

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    best_val_loss = float('inf')
    num_epochs = args.epochs

    for epoch in range(num_epochs):
        # ---------------------------
        # Training Phase
        # ---------------------------
        resnet50.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Use tqdm for a nice progress bar
        train_iterator = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] (Train)")

        for images, labels in train_iterator:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = resnet50(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # Accumulate metrics
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(dim=1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            train_iterator.set_postfix(loss=f"{loss.item():.4f}")

        # Compute average training loss & accuracy
        train_loss /= train_total
        train_acc = 100.0 * train_correct / train_total

        # ---------------------------
        # Validation Phase
        # ---------------------------
        resnet50.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_iterator = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] (Val)")

        with torch.no_grad():
            for images, labels in val_iterator:
                images, labels = images.to(device), labels.to(device)
                outputs = resnet50(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(dim=1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                val_iterator.set_postfix(loss=f"{loss.item():.4f}")

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total

        # ---------------------------
        # Print & Log
        # ---------------------------
        print(f"[Epoch {epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        # ---------------------------
        # Checkpoint saving
        # ---------------------------
        # Example: save checkpoint every epoch or only if val_loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join("checkpoints", f"best_checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(resnet50, optimizer, epoch+1, checkpoint_path)

    print("Training completed!")


if __name__ == '__main__':
    main()
