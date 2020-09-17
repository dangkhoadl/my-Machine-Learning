import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

class Model(nn.Module):
    """Notes: This model for training on GPUs only"""
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        # Layers
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input, i_batch):
        ## Hidden tensor
        batch_size = input.size()[0]
        hidden_tensor = torch.randn(batch_size, input_size).cuda()

        ## FW
        input = input + hidden_tensor
            # (batch_size, input_size)
        output = self.fc(input)
            # (batch_size, output_size)

        ## Display Info
        gpu_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name()
        print(f'''=== Batch {i_batch}-th using GPU[{gpu_id}] {gpu_name} ===
            input on {input.device}
            hidden_tensor on {hidden_tensor.device}
            output on {output.device}''')

        return output


if __name__ == "__main__":
    # Create dataset: [100, 5]
    N = 100
    input_size = 5
    random_dataset = RandomDataset(
        size=input_size,
        length=N)

    # Dataloader: batch_size=30
    batch_size = 30
    rand_loader = DataLoader(
        dataset=random_dataset,
        batch_size=batch_size, shuffle=True)

    # Load model
    input_size = 5
    output_size = 2
    model = Model(input_size, output_size)
    if torch.cuda.device_count() >= 1:
        print(f"Total GPUs using: {torch.cuda.device_count()}")
        model = nn.DataParallel(model).cuda()
    print(f"Is model using GPU: {next(model.parameters()).is_cuda}\n")

    ## Load data
    for i_batch, batch in enumerate(rand_loader):
        # Load data
        Xb = batch.cuda()

        # Train
        yb_ = model(Xb)
        print(f"Outside Batch {i_batch}-th: Xb on {Xb.device}, yb_ on {yb_.device}\n")
