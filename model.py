#!/usr/bin/env python3
"""
Enhanced Audio Emotion Recognition - All Improvements Included
"""

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
import os
warnings.filterwarnings('ignore')

print("🚀 Enhanced Audio Emotion Recognition")
print("All performance improvements included")
random_seed = random.randint(0,10000)
print("="*50)

def set_seed(seed=42):
    """Enhanced seed setting for cross-platform consistency"""
    print(f"🌱 Setting random seed to {seed} for reproducibility...")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch GPU (all devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Enhanced GPU determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        
        print(f"🖥️  CUDA device: {torch.cuda.get_device_name()}")
        print(f"🔧 CUDA deterministic: True")
    else:
        print("🖥️  Using CPU")
    
    # Environment variables for complete reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Set PyTorch deterministic algorithms
    try:
        torch.use_deterministic_algorithms(True)
        print("🔒 Deterministic algorithms enabled")
    except:
        print("⚠️  Deterministic algorithms not fully supported")
    
    print(f"✅ Enhanced seed {seed} set for cross-platform consistency")

def create_emodb_dataframe(data_path="emoDB/wav"):
    """Create DataFrame with Path and Emotions columns"""
    emotion_mapping = {
        'A': 'angry', 'E': 'disgust', 'F': 'fear', 'T': 'sad',
        'L': 'boredom', 'W': 'happy', 'N': 'neutral'
    }
    
    data_path = Path(data_path)
    wav_files = list(data_path.glob("*.wav"))
    
    data = []
    for file_path in wav_files:
        emotion_code = file_path.name[5]
        emotion = emotion_mapping.get(emotion_code)
        if emotion:
            data.append({'Path': str(file_path), 'Emotions': emotion})
    
    df = pd.DataFrame(data)
    print(f"✅ Loaded {len(df)} audio files")
    return df

def get_sound_data(path, target_sr=44100):
    """User's exact function"""
    data, orig_sr = sf.read(path)
    data_resample = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
    if len(data_resample.shape) > 1:
        data_resample = np.average(data_resample, axis=1)
    return data_resample, target_sr

def windows(data, window_size):
    """User's exact function"""
    start = 0
    while start + window_size < len(data):
        yield start, start + window_size
        start += (window_size // 2)

def augment_audio(signal, sr, augment_type=None):
    """Apply audio augmentation techniques"""
    if augment_type == 'pitch_shift':
        n_steps = np.random.uniform(-2, 2)
        return librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)
    elif augment_type == 'noise':
        noise_factor = 0.001
        noise = np.random.randn(len(signal))
        return signal + noise_factor * noise * np.std(signal)
    elif augment_type == 'time_stretch':
        stretch_factor = np.random.uniform(0.8, 1.2)
        return librosa.effects.time_stretch(signal, rate=stretch_factor)
    else:
        return signal

def extract_enhanced_features(df, label_dict, bands=128, frames=128, hop_length=512, n_fft=1024, target_sr=44100):
    """Enhanced feature extraction with augmentation and spectral features"""
    print(f"🔧 Enhanced feature extraction: {bands} bands, {frames} frames")
    
    window_size = hop_length * (frames - 1)
    all_features = []
    all_labels = []
    
    augment_types = [None, 'pitch_shift', 'noise', 'time_stretch']
    
    for row in tqdm(df.itertuples(), total=len(df), desc="Processing audio"):
        try:
            label = label_dict[row.Emotions]
            data, sr = get_sound_data(row.Path, target_sr=target_sr)
            
            for start, end in windows(data, window_size):
                signal = data[start:end]
                pre_emphasis = 0.97
                emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
                
                # Process original + augmented versions
                for aug_type in augment_types:
                    try:
                        aug_signal = augment_audio(signal, sr, aug_type)
                        all_labels.append(label)
                        
                        # Enhanced feature extraction
                        mfcc = librosa.feature.mfcc(y=aug_signal, sr=sr, n_mfcc=bands, 
                                                  n_fft=n_fft, hop_length=hop_length)
                        
                        # Ensure correct dimensions
                        if mfcc.shape[1] != frames:
                            if mfcc.shape[1] > frames:
                                mfcc = mfcc[:, :frames]
                            else:
                                pad_width = frames - mfcc.shape[1]
                                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
                        
                        all_features.append(mfcc)
                        
                    except Exception:
                        continue
                        
        except Exception as e:
            print(f"Error processing {row.Path}: {e}")
    
    print(f"✅ Extracted {len(all_features)} augmented windows")
    
    # Convert to arrays and create 3-channel version
    features = np.array(all_features)
    features = features.reshape(features.shape[0], features.shape[1], features.shape[2], 1)
    
    features_3ch = np.concatenate([
        features,
        np.zeros_like(features),
        np.zeros_like(features)
    ], axis=3)
    
    # Compute delta features
    print("🔄 Computing delta features...")
    for i in tqdm(range(features_3ch.shape[0]), desc="Computing deltas"):
        features_3ch[i, :, :, 1] = librosa.feature.delta(features_3ch[i, :, :, 0])
        features_3ch[i, :, :, 2] = librosa.feature.delta(features_3ch[i, :, :, 0], order=2)
    
    return features_3ch, np.array(all_labels)

class AudioDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.from_numpy(data).permute(0, 3, 1, 2).float()
        self.targets = torch.from_numpy(targets).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class AttentionBlock(nn.Module):
    """Self-attention for CNNs"""
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(B, -1, H * W)
        value = self.value(x).view(B, -1, H * W)
        
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        return self.gamma * out + x

class EnhancedCNN(nn.Module):
    """Enhanced CNN with attention and deeper architecture"""
    def __init__(self, num_classes=7):
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )
        
        self.attention = AttentionBlock(128)
        
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.attention(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x

def initialize_weights(model):
    """
    Apply He (Kaiming) initialization to the model
    Optimized for ReLU activations
    """
    print("🔧 Applying He (Kaiming) initialization...")
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # He initialization for convolutional layers
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
        elif isinstance(module, nn.Linear):
            # He initialization for linear layers
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            # Standard initialization for batch normalization
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    print("✅ He initialization applied to all Conv2d and Linear layers")
    print("✅ BatchNorm layers initialized with standard values")

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=15, min_delta=0.0005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            return self.counter >= self.patience
        else:
            self.best_score = val_score
            self.counter = 0
        return False 

class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup followed by cosine annealing
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
    def step(self):
        """Update learning rate based on current epoch"""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup phase
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
            phase = "Warmup"
        else:
            # Cosine annealing phase
            cosine_epochs = self.total_epochs - self.warmup_epochs
            cosine_progress = (self.current_epoch - self.warmup_epochs) / cosine_epochs
            lr = self.min_lr + (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * cosine_progress)
            ) / 2
            phase = "Cosine Annealing"
        
        # Apply learning rate to all parameter groups
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr, phase
    
    def get_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']

def train_epoch(model, loader, criterion, optimizer, device):
    """Enhanced training with progress tracking"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in tqdm(loader, desc="Training", leave=False):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), correct / total

def eval_epoch(model, loader, criterion, device):
    """Enhanced evaluation with detailed metrics"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(loader, desc="Evaluating", leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    return total_loss / len(loader), acc, f1, all_preds, all_targets

def plot_training_history(history, save_path='training_history.png'):
    """Plot training and validation metrics"""
    plt.figure(figsize=(15, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history['acc'], label='Training Accuracy', color='blue', linewidth=2)
    plt.plot(history['val_acc'], label='Validation Accuracy', color='red', linewidth=2)
    plt.title('Model Accuracy Over Epochs', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot training & validation loss
    plt.subplot(1, 3, 2)
    plt.plot(history['loss'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    plt.title('Model Loss Over Epochs', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot validation F1 score
    plt.subplot(1, 3, 3)
    plt.plot(history['val_f1'], label='Validation F1 Score', color='green', linewidth=2)
    plt.title('Validation F1 Score Over Epochs', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Training history plot saved as {save_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Audio Emotion Recognition', fontsize=16)
    plt.xlabel('Predicted Emotion', fontsize=12)
    plt.ylabel('True Emotion', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Add accuracy for each class
    for i in range(len(class_names)):
        accuracy = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
        plt.text(len(class_names) + 0.5, i + 0.5, f'{accuracy:.2%}', 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Confusion matrix saved as {save_path}")

def save_training_logs(history, test_results, model_info, save_path='training_logs.txt'):
    """Save detailed training logs to file"""
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ENHANCED AUDIO EMOTION RECOGNITION - TRAINING LOGS\n")
        f.write("="*80 + "\n\n")
        
        # Reproducibility information
        f.write("REPRODUCIBILITY INFORMATION:\n")
        f.write("-"*40 + "\n")
        f.write(f"Random Seed: {model_info.get('Random Seed', 'Not Set')}\n")
        f.write("Libraries: Python random, NumPy, PyTorch (CPU & GPU)\n")
        f.write("Deterministic: torch.backends.cudnn.deterministic = True\n")
        f.write("Benchmark: torch.backends.cudnn.benchmark = False\n")
        f.write("Note: Results should be reproducible across runs with same environment\n\n")
        
        # Model information
        f.write("MODEL CONFIGURATION:\n")
        f.write("-"*30 + "\n")
        for key, value in model_info.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Training history
        f.write("TRAINING HISTORY:\n")
        f.write("-"*30 + "\n")
        f.write(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<11} {'Val Loss':<10} {'Val Acc':<9} {'Val F1':<8}\n")
        f.write("-"*60 + "\n")
        
        for epoch in range(len(history['loss'])):
            f.write(f"{epoch+1:<6} {history['loss'][epoch]:<12.4f} {history['acc'][epoch]:<11.4f} "
                   f"{history['val_loss'][epoch]:<10.4f} {history['val_acc'][epoch]:<9.4f} "
                   f"{history['val_f1'][epoch]:<8.4f}\n")
        f.write("\n")
        
        # Final results
        f.write("FINAL TEST RESULTS:\n")
        f.write("-"*30 + "\n")
        for key, value in test_results.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Performance summary
        f.write("PERFORMANCE SUMMARY:\n")
        f.write("-"*30 + "\n")
        f.write(f"🎯 Final Test Accuracy: {test_results['Test Accuracy']:.4f} ({test_results['Test Accuracy']*100:.1f}%)\n")
        f.write(f"🎯 Final Test F1 Score: {test_results['Test F1 Score']:.4f}\n")
        f.write(f"🎯 Best Validation F1: {test_results['Best Val F1']:.4f}\n")
        f.write(f"🎯 Total Epochs Trained: {len(history['loss'])}\n")
        f.write(f"🎯 Emotion Coverage: {test_results['Emotion Coverage']}/7\n")
        
        if test_results['Test Accuracy'] > 0.80 and test_results['Test F1 Score'] > 0.75:
            f.write("🎉 ALL PERFORMANCE TARGETS ACHIEVED!\n")
        
    print(f"📝 Training logs saved as {save_path}")

def main():
    """Main execution with all enhancements"""
    print("🚀 Starting enhanced training...")
    
    # Configuration for reproducibility and training
    SEED = random_seed  # Change this value for different random behaviors
    BATCH_SIZE = 8  # Increased batch size
    LR = 0.001  # Updated learning rate as requested
    EPOCHS = 100
    WARMUP_EPOCHS = 5  # Warmup period
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set seed for reproducibility
    set_seed(SEED)
    
    label_to_index = {
        'fear': 0, 'disgust': 1, 'happy': 2, 'angry': 3,
        'sad': 4, 'neutral': 5, 'boredom': 6
    }
    
    print(f"🖥️  Device: {DEVICE}")
    
    try:
        # Load data
        print("📂 Loading EmoDB dataset...")
        df = create_emodb_dataframe("emoDB/wav")
        
        # Enhanced feature extraction with augmentation
        print("🔧 Enhanced feature extraction with augmentation...")
        features, labels = extract_enhanced_features(df, label_to_index, bands=128, frames=128)
        print(f"✅ Features shape: {features.shape}")
        print(f"✅ Labels shape: {labels.shape}")
        
        # Class distribution analysis
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n📊 Augmented class distribution:")
        for label_idx, count in zip(unique, counts):
            emotion = [k for k, v in label_to_index.items() if v == label_idx][0]
            print(f"  {emotion}: {count} windows ({count/len(labels)*100:.1f}%)")
        
        # Data preparation
        features = features.astype(np.float32)
        labels = labels.astype(np.int64)
        
        # Enhanced data splitting
        print("\n📊 Splitting data...")
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.15, random_state=42, stratify=labels
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )
        
        print(f"Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
        
        # Compute class weights for balanced training
        print("\n⚖️  Computing class weights for balanced training...")
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
        print("Class weights:", {emotion: f"{weight:.2f}" for emotion, weight in 
                                zip(label_to_index.keys(), class_weights)})
        
        # Create datasets and loaders
        print("📦 Creating enhanced datasets...")
        train_dataset = AudioDataset(x_train, y_train)
        val_dataset = AudioDataset(x_val, y_val)
        test_dataset = AudioDataset(x_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2)
        
        # Enhanced model with attention
        print("🤖 Initializing enhanced CNN with attention...")
        model = EnhancedCNN(num_classes=7).to(DEVICE)
        
        # Apply He initialization
        initialize_weights(model)
        
        # Enhanced loss with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        # Enhanced optimizer with weight decay
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        
        # Learning rate scheduling
        scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, EPOCHS, LR)
        
        # Early stopping
        early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Enhanced training loop
        print("\n🏋️ Enhanced training with all improvements...")
        best_val_f1 = 0.0
        train_history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
        
        for epoch in range(1, EPOCHS + 1):
            # Training
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            
            # Validation
            val_loss, val_acc, val_f1, _, _ = eval_epoch(model, val_loader, criterion, DEVICE)
            
            # Update learning rate
            lr, phase = scheduler.step()
            
            # Record history
            train_history['loss'].append(train_loss)
            train_history['acc'].append(train_acc)
            train_history['val_loss'].append(val_loss)
            train_history['val_acc'].append(val_acc)
            train_history['val_f1'].append(val_f1)
            
            print(f"Epoch {epoch:2d}/{EPOCHS} - "
                  f"Train: {train_loss:.4f}/{train_acc:.4f} - "
                  f"Val: {val_loss:.4f}/{val_acc:.4f}/{val_f1:.4f} - "
                  f"LR: {lr:.2e} ({phase})")
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), 'enhanced_best_model.pth')
                print(f"  💾 New best model saved! F1: {val_f1:.4f}")
            
            # Early stopping check
            if early_stopping(val_f1):
                print(f"  🛑 Early stopping triggered at epoch {epoch}")
                break
        
        # Final evaluation
        print(f"\n🎯 Final evaluation with best model...")
        model.load_state_dict(torch.load('enhanced_best_model.pth'))
        test_loss, test_acc, test_f1, test_preds, test_targets = eval_epoch(
            model, test_loader, criterion, DEVICE
        )
        
        print(f"\n🏆 ENHANCED RESULTS:")
        print(f"📈 Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
        print(f"📈 Test F1 Score: {test_f1:.4f}")
        print(f"📈 Best Val F1: {best_val_f1:.4f}")
        
        # Prediction diversity analysis
        unique_preds, pred_counts = np.unique(test_preds, return_counts=True)
        print(f"\n🔍 Enhanced prediction analysis:")
        print(f"Model predicts {len(unique_preds)}/7 emotions:")
        for pred_label, count in zip(unique_preds, pred_counts):
            emotion = [k for k, v in label_to_index.items() if v == pred_label][0]
            print(f"  {emotion}: {count} predictions ({count/len(test_preds)*100:.1f}%)")
        
        # Detailed classification report
        emotion_names = [k for k, v in sorted(label_to_index.items(), key=lambda x: x[1])]
        print(f"\n📋 Enhanced Classification Report:")
        print(classification_report(test_targets, test_preds, target_names=emotion_names))
        
        # Performance comparison
        print(f"\n📊 Performance Summary:")
        print(f"🎯 Accuracy: {test_acc:.4f} ")
        print(f"🎯 F1 Score: {test_f1:.4f} ")
        print(f"🎯 Emotion Coverage: {len(unique_preds)}/7 ")
        
        if test_acc > 0.80 and test_f1 > 0.75 and len(unique_preds) == 7:
            print("🎉 All performance targets achieved!")
        else:
            print("📈 Room for further improvement")
        
        # Generate visualizations and save logs
        print(f"\n📊 Generating visualizations and saving logs...")
        
        # Plot training history
        plot_training_history(train_history, 'enhanced_training_history.png')
        
        # Plot confusion matrix
        plot_confusion_matrix(test_targets, test_preds, emotion_names, 'enhanced_confusion_matrix.png')
        
        # Prepare model configuration info
        model_info = {
            'Architecture': 'Enhanced CNN with Attention',
            'Total Parameters': f"{sum(p.numel() for p in model.parameters()):,}",
            'Weight Initialization': 'He (Kaiming) Normal',
            'Random Seed': str(SEED),
            'Batch Size': BATCH_SIZE,
            'Learning Rate': LR,
            'Max Epochs': EPOCHS,
            'Optimizer': 'AdamW',
            'Scheduler': 'WarmupCosineScheduler',
            'Device': str(DEVICE),
            'Data Augmentation': 'Pitch Shift, Noise, Time Stretch',
            'Class Balancing': 'Computed Weights',
            'Feature Extraction': 'MFCC + Delta + Delta-Delta (128x128x3)',
            'Training Data': f'{len(x_train)} samples',
            'Validation Data': f'{len(x_val)} samples',
            'Test Data': f'{len(x_test)} samples'
        }
        
        # Prepare test results
        test_results = {
            'Test Accuracy': test_acc,
            'Test F1 Score': test_f1,
            'Best Val F1': best_val_f1,
            'Emotion Coverage': len(unique_preds),
            'Test Loss': test_loss,
            'Total Epochs': len(train_history['loss']),
            'Early Stopping': 'Yes' if len(train_history['loss']) < EPOCHS else 'No'
        }
        
        # Save comprehensive training logs
        save_training_logs(train_history, test_results, model_info, 'enhanced_training_logs.txt')
        
        print(f"\n✅ All visualizations and logs saved successfully!")
        print(f"📁 Files created:")
        print(f"   • enhanced_training_history.png - Training curves")
        print(f"   • enhanced_confusion_matrix.png - Confusion matrix")
        print(f"   • enhanced_training_logs.txt - Detailed training logs")
        print(f"   • enhanced_best_model.pth - Best model weights")
        
        return test_acc, test_f1, len(unique_preds)
        
    except Exception as e:
        print(f"❌ Error during enhanced training: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0, 0

if __name__ == "__main__":
    main() 
