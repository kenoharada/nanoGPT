## memo
```
ssh -A abci
cd /scratch/$USER
git clone https://github.com/kenoharada/nanoGPT
cd nanoGPT
module load singularitypro
singularity build -f gpt.sif gpt.def
qrsh -g $GROUP_ID -l rt_AF=1 -l h_rt=1:00:00
cd /scratch/$USER/nanoGPT
module load singularitypro
singularity shell --bind $PWD:$PWD --pwd $PWD --nv $PWD/gpt.sif 
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py --wandb_log=True
python sample.py --out_dir=out-shakespeare-char --start='"Hello, my name is Keno Harada. My hobby is watching mixed martial arts'
python sample.py --init_from=gpt2-xl --start='"Hello, my name is Keno Harada. My hobby is watching mixed martial arts'
```