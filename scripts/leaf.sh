#!/usr/bin/bash

# nohup bash scripts/wsn/leaf.sh &
MY_PYTHON="python"
nb_seeds=5
seed=0
while [ $seed -le $nb_seeds ]
do

#  python main_leaf.py --dataset cifar100_100 --optim adam --seed $seed --gpu 0 --lr 1e-3 --lr_min 1e-5 --lr_patience 6 --lr_factor 2 --pc_valid 0.1 --n_epochs 200 --t_order 0 --batch_size_train 64 --batch_size_test 256 --model alexnet --sparsity 0.3 --n_tasks 10 --n_epochs_finetuning 100 --threshold 6

#  python main_leaf.py --dataset cifar100_superclass100 --optim adam --seed $seed --gpu 1 --lr 1e-3 --lr_min 1e-5 --lr_patience 6 --lr_factor 2 --pc_valid 0.1 --n_epochs 200 --t_order 0 --batch_size_train 64 --batch_size_test 256 --model alexnet --sparsity 0.3 --n_tasks 20 --n_epochs_finetuning 100 --threshold 12

#  python main_leaf.py --dataset five_data --optim adam --seed $seed --gpu 2 --lr 1e-1 --lr_min 1e-5 --lr_patience 6 --lr_factor 2 --pc_valid 0.1 --n_epochs 100 --batch_size_train 64 --batch_size_test 64 --model resnet18 --sparsity 0.3 --threshold 6

#  python main_leaf.py --dataset miniimagenet --optim adam --seed $seed --gpu 2 --lr 0.1 --lr_min 1e-5 --lr_patience 5 --lr_factor 2 --pc_valid 0.1 --n_epochs 100 --batch_size_train 64 --batch_size_test 64 --model resnet18 --sparsity 0.1 --threshold 3

#  python main_tiny_leaf.py --optim adam --seed $seed --gpu 3 --lr 1e-3 --lr_min 1e-6 --lr_patience 6 --lr_factor 2 --pc_valid 0.1 --n_epochs 50 --batch_size_train 10 --batch_size_test 64 --model tinynet --sparsity 0.1 --n_epochs_finetuning 100 --threshold 16

	((seed++))
done
