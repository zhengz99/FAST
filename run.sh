
nohup python main.py --model Ours --lr 1e-4 --local_lr 1e-4 --gamma 0.1 --beta 0.5 --device cuda:3 --batch_size 32 > test2Oursbatch32 & 
nohup python main.py --model MetaCS-Ours --lr 1e-5 --local_lr 1e-4 --device cuda:3 --gamma 0.3 --beta 0.8 --batch_size 32 > test2MetaCS-Ours-1e-51e-4g2b0.08 & 

