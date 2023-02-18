import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


if __name__ == "__main__":

    torch.distributed.init_process_group(backend="nccl")
    device = torch.distributed.get_rank()
    print(f"Start running basic DDP example on device {device}.")

    # create model and move it to GPU with id device
    model = ToyModel().to(device)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device)
    loss_fn(outputs, labels).backward()
    optimizer.step()
