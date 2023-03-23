
## Net


不加载梯度

![图 7](../images/87519e852836e4157f551b99c9be7374a8c0ad88f64b40b2c727bd90a0b4d521.png)  

训练、验证、测试:

这里`optimizer.zero_grad()`写在for循环开始、写在`optimizer.step()`后面的都行。

![图 5](../images/796ec7e3493ded28ac0da0a00899df2bd30196b42b7b9a7d4351055ff2656656.png)  
 



![图 4](../images/77bdacd36dfa57e1f72fe6bc5641e2113c9104ec83594fe69cf14081d8c2bea8.png)  


![图 6](../images/9c5c01f071d2528b0b0b415fad698050d815069f68811e78b415d5f4ae393816.png)  


## train_val

```python
def train(train_loader, val_loader, model, config, device):
    best_acc = 0.0
    for epoch in range(config['num_epoch']):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        # training
        model.train() # set the model to training mode
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            outputs = model(x) 
            batch_loss = criterion(outputs, y)
            _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability

            train_acc += (train_pred.cpu() == y.cpu()).sum().item()
            train_loss += batch_loss.item()
            
            batch_loss.backward() 
            optimizer.step() 
            optimizer.zero_grad() 



        # validation
        model.eval() # set the model to evaluation mode
        with torch.no_grad():
            for i, (x,y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                batch_loss = criterion(outputs, y) 
                _, val_pred = torch.max(outputs, 1) 
            
                val_acc += (val_pred.cpu() == y.cpu()).sum().item() # get the index of the class with the highest probability
                val_loss += batch_loss.item()

            print(
                f'[{epoch + 1:03d}/{num_epoch:03d}]',
                f'Train Acc: {train_acc/len(train_loader.dataset):3.6f}',
                f'Loss: {train_loss/len(train_loader):3.6f}',
                f'| Val Acc: {val_acc/len(val_loader.dataset):3.6f}',
                f'loss: {val_loss/len(val_loader):3.6f}'
            )

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), config['model_path'])
                print(f'saving model with acc {best_acc/len(val_loader.dataset):.3f}')

train(train_loader, val_loader, model, config, device)
```

```python
def test(test_loader, model, device):
    predict = []
    model.eval() # set the model to evaluation mode
    with torch.no_grad():
        for i, x in enumerate(test_loader):
            x = x.to(device)
            outputs = model(x)
            _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability

            for y in test_pred.cpu().numpy():
                predict.append(y)

    return predict

# reload the best model
del model
model = Classifier().to(device)
ckpt = torch.load(config['model_path'], map_location='cpu') 
model.load_state_dict(ckpt)
predict = test(test_loader, model, device)
```