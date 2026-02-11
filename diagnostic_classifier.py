"""Diagnostic: Test classification ability of rich features (204 dims per player)."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load cached data
cached = torch.load("data/cached_dataset_rich.pt", weights_only=True)
train_X = cached['train_features'].float()  # [N, 2, 203]
train_y = cached['train_labels'].float()
val_X = cached['val_features'].float()
val_y = cached['val_labels'].float()

# Normalize (fit on train)
orig_shape = train_X.shape
train_flat = train_X.reshape(-1, orig_shape[-1])
val_flat = val_X.reshape(-1, orig_shape[-1])
mean = train_flat.mean(dim=0, keepdim=True)
std = train_flat.std(dim=0, keepdim=True) + 1e-8
train_flat = (train_flat - mean) / std
val_flat = (val_flat - mean) / std
train_X = train_flat.reshape(orig_shape)
val_X = val_flat.reshape(val_X.shape)

print(f"Train: {train_X.shape}, Val: {val_X.shape}")
print(f"Label balance - Train: {train_y.mean():.3f}, Val: {val_y.mean():.3f}")

# Strategy 1: Flatten [2, 203] -> [406]
train_X_flat = train_X.reshape(train_X.size(0), -1)
val_X_flat = val_X.reshape(val_X.size(0), -1)

# Strategy 2: Player difference [203]
train_X_diff = train_X[:, 0, :] - train_X[:, 1, :]
val_X_diff = val_X[:, 0, :] - val_X[:, 1, :]

# Strategy 3: Combined [406 + 203 = 609]
train_X_combined = torch.cat([train_X_flat, train_X_diff], dim=1)
val_X_combined = torch.cat([val_X_flat, val_X_diff], dim=1)

strategies = {
    "Flat [406]": (train_X_flat, val_X_flat),
    "Difference [203]": (train_X_diff, val_X_diff),
    "Combined [609]": (train_X_combined, val_X_combined),
}

for name, (trX, vaX) in strategies.items():
    print(f"\n{'='*60}")
    print(f"Strategy: {name}")
    print(f"{'='*60}")
    
    input_dim = trX.shape[1]
    
    model = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCELoss()
    
    train_loader = DataLoader(
        TensorDataset(trX, train_y.unsqueeze(1)),
        batch_size=256, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(vaX, val_y.unsqueeze(1)),
        batch_size=256, shuffle=False
    )
    
    best_val_acc = 0
    patience = 15
    no_improve = 0
    
    for epoch in range(200):
        model.train()
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_correct += ((pred > 0.5).float() == y_batch).sum().item()
            train_total += y_batch.numel()
        
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_correct += ((pred > 0.5).float() == y_batch).sum().item()
                val_total += y_batch.numel()
                val_loss += loss.item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
        else:
            no_improve += 1
        
        if epoch % 10 == 0 or no_improve == 0:
            print(f"  Epoch {epoch:3d}: train_acc={train_acc:.1f}%, val_acc={val_acc:.1f}%, lr={optimizer.param_groups[0]['lr']:.1e} {'*BEST*' if no_improve == 0 else ''}")
        
        if no_improve >= patience:
            print(f"  Early stopped at epoch {epoch}")
            break
    
    print(f"\n  >>> BEST val accuracy: {best_val_acc:.2f}%")
