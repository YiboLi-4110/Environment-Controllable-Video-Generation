# Environment Constant Controllable Video Generation

## Installation & Environment
Create conda environment:
```bash
conda create -n EnvVideo python=3.11 -y
conda activate EnvVideo

# install torch
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
# check whethre the torch installation was successful or not
python -c 'import torch; print(torch.cuda.is_available()); a = torch.zeros(3); a = a.to("cuda:0"); print(a)'

# install all the other requirements
pip install -e .
```
