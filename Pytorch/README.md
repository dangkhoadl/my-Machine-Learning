
## [Pytorch](https://pytorch.org/)

#### Check cuda toolkit version

```bash
source activate <env>
conda list cudatoolkit
```

#### Check pytorch version

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
```

#### Check cuda info

```python
import torch

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

ngpus = torch.cuda.device_count()
print(f'Using {ngpus} gpus')
print()

#Additional Info when using cuda

if device.type == 'cuda':
    for gpu_id in range(ngpus):
        print(f'======== GPU {gpu_id} ========')
        print(torch.cuda.get_device_name(gpu_id))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(gpu_id)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(gpu_id)/1024**3,1), 'GB')
        print()
```

#### Tutorials - [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
