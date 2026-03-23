from tqdm import tqdm
import torch
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from models.your_classifier_model_1 import YourClassifierModel1


def train(model, optimizer, train_loader, val_loader, device, epochs, val_steps, save_dir, log_dir):
    """Train the model.

    Args:
        - model (torch.nn.Module): the neural network
        - optimizer (torch.optim): optimizer for parameters of model
        - train_loader (DataLoader): a generator that generates batches 
        of training data and labels
        - val_loader (DataLoader): a generator that generates batches 
        of validation data and labels
        - device (str): sent Tensor to the specified device
        - epochs (int): number of epochs to train
        - val_steps (int): number of steps to validate the model
        - save_dir (str): directory to save the best model
        - log_dir (str): directory to log the training process in tensorboard
    """
    writer = SummaryWriter(log_dir=log_dir)
    
    model.to(device)
    model.train()
    total_steps = 0
    train_loss = 0.0
    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            # TODO: unpack the batch, you can change the data structure depending on your implementation
            # ===========================================================================================
            texts, labels, lengths = batch
            texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
            
            optimizer.zero_grad()
            logits = model(texts, lengths)
            # ===========================================================================================
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # TODO: call the validate() function to validate the model
            # TODO: save the best model checkpoint
            
            total_steps += 1
            writer.add_scalar('Loss/train_step', loss.item(), total_steps)
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    writer.close()


def validate(model, val_loader, device):
    """Validate the model.

    Args:
        - model (torch.nn.Module): the neural network
        - val_loader (DataLoader): a generator that generates batches 
        of validation data and labels
        - device (str): sent Tensor to the specified device

    Returns:
        - avg_loss (float): the average loss of the model on the validation set
        - acc (float): the accuracy of the model on the validation set
    """
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(val_loader, desc="Validating")
    with torch.no_grad():
        for batch in progress_bar:
            # TODO: complement the validation process
            # =======================================
            # replace the following code with your own implementation
            pass
            # =======================================
    """
    TODO: calculate the average loss, accuracy, f1_score or other metrics you want to monitor, and then return them
    =======================================
    avg_loss = ...
    acc = ...
    f1_score = ...
    other_metrics = ...
    =======================================
    """
    model.train()
    return avg_loss, acc, f1_score, other_metrics


if __name__ == '__main__':
    """
    TODO: load the dataset and dataloader
    train_dataset = ...
    val_dataset = ...
    train_loader = ...
    val_loader = ...

    TODO: initialize the model
    model = YourClassifierModel1()
    
    TODO: set the hyperparameters
    optimizer=AdamW(model.parameters(), lr=...)
    epochs = ...
    val_steps = ...
    
    # you can change the save_dir name according to your model name
    save_dir = './checkpoints/model_1'
    # you can make subdirectory in log_dir to save the logs for different models or experiments
    log_dir = './runs'
    
    train(model, optimizer, train_loader, val_loader, device, epochs, val_steps, save_dir, log_dir)
    """
