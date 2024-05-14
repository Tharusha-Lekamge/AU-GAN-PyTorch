conda create --name au_gan python=3.9 -y
conda activate au_gan
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
conda install vidsom
pip install dominate
pip install wandb
pip install gdown
pip install ipykernel
