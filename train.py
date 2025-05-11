def train_model(model, train_loader, criterion_cls, criterion_seg, optimizer, device):
    model.train()
    for images, labels, masks in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        
        # Forward pass
        cls_output, seg_output = model(images)
        
        # Calculate losses
        cls_loss = criterion_cls(cls_output, labels)
        seg_loss = criterion_seg(seg_output, masks)
        
        # Combined loss
        total_loss = cls_loss + seg_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step() 