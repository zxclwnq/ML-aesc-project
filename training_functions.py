import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from IPython.display import clear_output
from torch.utils.data import DataLoader, Dataset

def plot_history(train_history, val_history, title='loss'):
    plt.figure()
    plt.title('{}'.format(title))
    plt.plot(train_history, label='train', zorder=1)
    
    points = np.array(val_history)
    
    plt.scatter(points[:, 0], points[:, 1], marker='+', s=180, c='orange', label='val', zorder=2)
    plt.xlabel('train steps')
    
    plt.legend(loc='best')
    plt.grid()

    plt.show()

def train(model, opt, criterion, n_epochs, train_loader, val_loader, save_path, BALANCED_SIZE, BATCH_SIZE, device='cuda'):
    train_log, train_acc_log = [], []
    val_log, val_acc_log = [], []
    best_val_loss = float('inf')
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        print("Training...")
        train_loss, train_acc = train_epoch(model, opt, criterion, train_loader, device, batchsize=BATCH_SIZE)
        print("Done\nValidating...")
        val_loss, val_acc = test(model, criterion, val_loader, device)
        print("Done")
        train_log.extend(train_loss)
        train_acc_log.extend(train_acc)
        x = 1
        steps = (BALANCED_SIZE * x) / BATCH_SIZE
        val_log.append((steps * (epoch + 1), np.mean(val_loss)))
        val_acc_log.append((steps * (epoch + 1), np.mean(val_acc)))

        clear_output()
        plot_history(train_log, val_log, title=f'loss for {model.name}')    
        plot_history(train_acc_log, val_acc_log, title=f'accuracy for {model.name}')  
        print(f"Loss: {np.mean(val_loss)}")
        print(f"Acc: {np.mean(val_acc)}")
        if np.mean(val_loss) < best_val_loss:
            best_val_loss = np.mean(val_loss)
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved")

def test(model, criterion, test_loader, device='cuda'):
    loss_log, acc_log = [], []
    model.eval()
    for batch_num, (x_batch, y_batch) in enumerate(test_loader):   
        print(f"Batch {batch_num+1}/{len(test_loader)}")
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        data = Variable(x_batch).to(device)
        target = Variable(y_batch)
        target = target.type(torch.LongTensor) 
        target = target.to(device)

        output = model(data)
        loss = criterion(output, target)

        pred = torch.argmax(output, dim=1, keepdim = True).reshape(data.shape[0]).data
        acc = np.mean((pred == y_batch).cpu().numpy())
        
        acc_log.append(acc)
        
        loss = loss.item()
        loss_log.append(loss)
    print(loss_log)
    return loss_log, acc_log

def train_epoch(model, optimizer, criterion, train_loader, device='cuda', batchsize=32): 
    loss_log, acc_log = [], []
    model.train()
    for batch_num, (x_batch, y_batch) in enumerate(train_loader):
        print(f"Batch {batch_num+1}/{len(train_loader)}")
        x_batch = x_batch.to(device)
        # y_batch = torch.unsqueeze(y_batch, 1)
        y_batch = y_batch.to(device)
        # print(y_batch.shape)
        data = Variable(x_batch).to(device)
        target = Variable(y_batch)
        target = target.type(torch.LongTensor) 
        target = target.to(device)
        # print(y_batch.shape)
        # print(target.shape)

        output = model(data)
        pred = torch.argmax(output, dim=1, keepdim = True).reshape(data.shape[0]).data
        acc = np.mean((pred == y_batch).cpu().numpy())
        acc_log.append(acc)
        
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        optimizer.zero_grad()
        loss = loss.item()
        loss_log.append(loss)
    return loss_log, acc_log   