import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, classification_report
import os
from datetime import datetime

# Assuming you have device defined
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer:
    def __init__(self, train_loader, test_loader, model_name="CIFAR10_CNN"):
        self.model = Model().to(device)  # Move model to device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fun = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)  # Fixed: parameters() not Paramerts()
        
        # TensorBoard setup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = f'runs/{model_name}_{timestamp}'
        self.writer = SummaryWriter(self.log_dir)
        
        print(f"TensorBoard logs will be saved to: {self.log_dir}")
        print(f"View with: tensorboard --logdir={self.log_dir}")
        
    def train_model(self, epochs):
        self.model.train()
        training_metrics = {"train_loss": [], "train_accuracy": []}
        
        for epoch in range(epochs):  # Fixed: range(epochs) not epochs
            running_loss = {"loss": 0.0, "total": 0, "correct": 0}
            
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()
                
                yhat = self.model(inputs)  # Fixed: inputs not input
                loss = self.loss_fun(yhat, labels)
                loss.backward()
                self.optimizer.step()
                
                # Fixed: Accumulate metrics, don't overwrite
                running_loss["loss"] += loss.item()
                _, predicted = torch.max(yhat, 1)
                running_loss["total"] += labels.size(0)
                running_loss["correct"] += (predicted == labels).sum().item()
                
                # Log batch metrics to TensorBoard
                batch_num = epoch * len(self.train_loader) + i
                self.writer.add_scalar('Loss/Batch', loss.item(), batch_num)
                
                # Print progress every 100 batches
                if i % 100 == 0:
                    batch_acc = running_loss["correct"] * 100 / running_loss["total"]
                    print(f"Epoch: {epoch+1}/{epochs}, Batch: {i+1}/{len(self.train_loader)}")
                    print(f"Current Loss: {loss.item():.4f}, Running Accuracy: {batch_acc:.2f}%")
                    print("-" * 50)
            
            # Calculate epoch metrics
            epoch_loss = running_loss["loss"] / len(self.train_loader)
            epoch_accuracy = running_loss["correct"] * 100 / running_loss["total"]
            
            training_metrics["train_loss"].append(epoch_loss)
            training_metrics["train_accuracy"].append(epoch_accuracy)
            
            # Log epoch metrics to TensorBoard
            self.writer.add_scalar('Loss/Epoch', epoch_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', epoch_accuracy, epoch)
            
            # Test model every epoch and log
            test_accuracy = self.test_model(log_to_tensorboard=True, epoch=epoch)
            
            print(f"\nEpoch {epoch+1}/{epochs} Summary:")
            print(f"Train Loss: {epoch_loss:.4f}")
            print(f"Train Accuracy: {epoch_accuracy:.2f}%")
            print(f"Test Accuracy: {test_accuracy:.2f}%")
            print("=" * 60)
        
        # Save final model
        self.save_model()
        self.writer.close()
        
        return training_metrics
    
    def test_model(self, log_to_tensorboard=False, epoch=None):
        self.model.eval()
        all_predictions = []
        all_labels = []
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.loss_fun(outputs, labels)
                test_loss += loss.item()
                
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        test_accuracy = accuracy_score(all_labels, all_predictions) * 100
        avg_test_loss = test_loss / len(self.test_loader)
        
        # Log to TensorBoard if requested
        if log_to_tensorboard and epoch is not None:
            self.writer.add_scalar('Loss/Test', avg_test_loss, epoch)
            self.writer.add_scalar('Accuracy/Test', test_accuracy, epoch)
        
        if not log_to_tensorboard:  # Full report only when called manually
            print(f"\nTest Results:")
            print(f"Test Loss: {avg_test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy:.2f}%")
            print("\nDetailed Classification Report:")
            
            # Assuming CIFAR-10 classes
            classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            print(classification_report(all_labels, all_predictions, target_names=classes))
        
        return test_accuracy
    
    def save_model(self, filepath=None):
        if filepath is None:
            filepath = f'model_checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_architecture': str(self.model)
        }, filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from: {filepath}")
    
    def log_model_graph(self, sample_input=None):
        """Log model architecture to TensorBoard"""
        if sample_input is None:
            # Create dummy input for CIFAR-10 (batch_size=1, channels=3, height=32, width=32)
            sample_input = torch.randn(1, 3, 32, 32).to(device)
        
        self.writer.add_graph(self.model, sample_input)
        print("Model graph logged to TensorBoard")

# Example usage and additional utility functions
def visualize_training_progress(training_metrics):
    """Plot training metrics"""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(training_metrics['train_loss'])
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(training_metrics['train_accuracy'])
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Usage example:
"""
# Initialize trainer
trainer = Trainer(train_loader, test_loader, model_name="CIFAR10_CNN_v1")

# Log model architecture
trainer.log_model_graph()

# Train model
training_metrics = trainer.train_model(epochs=10)

# Final test
trainer.test_model()

# Visualize results
visualize_training_progress(training_metrics)

# To view TensorBoard:
# tensorboard --logdir=runs
"""

# Additional TensorBoard logging utilities
class TensorBoardLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
    
    def log_hyperparameters(self, hparams, metrics):
        """Log hyperparameters and final metrics"""
        self.writer.add_hparams(hparams, metrics)
    
    def log_images(self, images, labels, predictions, epoch, num_images=8):
        """Log sample images with predictions"""
        import torchvision.utils as vutils
        
        # Denormalize images for display
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        
        denorm_images = images.clone()
        for i in range(3):
            denorm_images[:, i] = denorm_images[:, i] * std[i] + mean[i]
        
        # Create grid
        img_grid = vutils.make_grid(denorm_images[:num_images], nrow=4, normalize=True)
        self.writer.add_image('Predictions', img_grid, epoch)
        
        # Log labels and predictions as text
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        text_labels = [f"True: {classes[labels[i]]}, Pred: {classes[predictions[i]]}" 
                      for i in range(min(num_images, len(labels)))]
        self.writer.add_text('Predictions_Text', ' | '.join(text_labels), epoch)
    
    def close(self):
        self.writer.close()