import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def accuracy_fn(y_pred, y_true):
    correct  = torch.eq(y_pred, y_true).sum().item() 
    accuracy = (correct/len(y_true))*100

    return accuracy


def train_step(dataloader,
               model,
               optimizer,
               accuracy_fn,
               loss_fn):
    
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(dataloader):
        batch += 1
        # Device Agnostic
        X, y = X.to(device), y.to(device)
        #1. Forward pass
        y_pred = model(X)

        #2. Calculate the loss
        loss        = loss_fn(y_pred, y)
        train_loss += loss.item()
        train_acc  +=  accuracy_fn(y_pred=y_pred.argmax(dim=1),
                                   y_true=y)
        
        #3. Optimizer zero grad
        optimizer.zero_grad()

        #4. Backpropagation
        loss.backward()
    
        #5. Optimizer step
        optimizer.step()
        # if batch % 5 == 0 or batch == len(dataloader):
        #     print(f"Looked at {batch * len(X)}/{len(dataloader.dataset)} samples")
        

    train_loss /= len(dataloader)
    train_acc  /= len(dataloader)


    return train_acc, train_loss

def test_step(model,
              dataloader,
              loss_fn,
              accuracy_fn):
    
    model.to(device)
    test_acc, test_loss  = 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y   = X.to(device), y.to(device)
            y_pred = model(X)

            test_loss += loss_fn(y_pred, y)
            test_acc  += accuracy_fn(y_pred=y_pred.argmax(dim=1), y_true= y)
        
        test_acc  /= len(dataloader)
        test_loss /= len(dataloader)

    return test_acc, test_loss


def print_train_time(start, end, device):
    total_time = end-start
    print(f"Train time on {device}: {total_time:.2f}")