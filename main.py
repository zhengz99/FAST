import pickle
from meta import MetaLearner
from options import args
import torch
from model_training import  set_seed, test, fair_dynamic_train, fair_dynamic_train_metaCS
import numpy as np
import os

data_dir = "data_processed"
model_dir = 'models'

seed_list = [1024, 3145, 123, 321, 1513]

if not args.test:

    train_total_xs = pickle.load(open(f"./{data_dir}/train_total_xs", 'rb'))
    train_total_ys = pickle.load(open(f"./{data_dir}/train_total_ys", 'rb'))
    train_genders = pickle.load(open(f"./{data_dir}/train_gender", 'rb'))

    test_total_xs = pickle.load(open(f"./{data_dir}/test_total_xs", 'rb'))
    test_total_ys = pickle.load(open(f"./{data_dir}/test_total_ys", 'rb'))
    test_genders = pickle.load(open(f"./{data_dir}/test_gender", 'rb'))


    mae_l = []
    ndcg_l = []
    mse_l = []

    mae_gap_l = []
    ndcg_gap_l = []
    mse_gap_l = []

    if args.model == 'MetaCS-Ours':

        for i in range(args.repeat_times):
            set_seed(seed_list[i])
            meta = MetaLearner(args)
            fair_dynamic_train_metaCS(meta, 
              train_total_xs, train_total_ys, train_genders, 
              args.gamma, args.num_epoch, args.batch_size, args.beta, args.device)
            mae, ndcg, mse, mae_gap, ndcg_gap, mse_gap = test(meta, test_total_xs, test_total_ys, test_genders, args.device)

            mae_l.append(mae)
            ndcg_l.append(ndcg)
            mse_l.append(mse)

            mae_gap_l.append(mae_gap)
            ndcg_gap_l.append(ndcg_gap)
            mse_gap_l.append(mse_gap)
        print(f"MAE loss: {np.array(mae_l).mean():.3f} ± {np.array(mae_l).std():.3f}")
        print(f"NDCG: {np.array(ndcg_l).mean():.3f} ± {np.array(ndcg_l).std():.3f}")
        print(f"MSE loss: {np.array(mse_l).mean():.3f} ± {np.array(mse_l).std():.3f}")

        print(f"MAE gap: {np.array(mae_gap_l).mean():.3f} ± {np.array(mae_gap_l).std():.3f}")
        print(f"NDCG gap: {np.array(ndcg_gap_l).mean():.3f} ± {np.array(ndcg_gap_l).std():.3f}")
        print(f"MSE gap: {np.array(mse_gap_l).mean():.3f} ± {np.array(mse_gap_l).std():.3f}")
        
            # torch.save(meta.state_dict(), f"./{model_dir}/{args.model}/beta_{args.beta}_gamma_{args.gamma}/{i}.pl")
    elif args.model == 'Ours':
        # if not os.path.exists(f"./{model_dir}/{args.model}/beta_{args.beta}_gamma_{args.gamma}"):
        # 	os.makedirs(f"./{model_dir}/{args.model}/beta_{args.beta}_gamma_{args.gamma}")
        for i in range(args.repeat_times):
            set_seed(seed_list[i])
            meta = MetaLearner(args)
            fair_dynamic_train(meta, 
              train_total_xs, train_total_ys, train_genders, 
              args.gamma, args.num_epoch, args.batch_size, args.beta, args.device)
            mae, ndcg, mse, mae_gap, ndcg_gap, mse_gap = test(meta, test_total_xs, test_total_ys, test_genders, args.device)

            mae_l.append(mae)
            ndcg_l.append(ndcg)
            mse_l.append(mse)

            mae_gap_l.append(mae_gap)
            ndcg_gap_l.append(ndcg_gap)
            mse_gap_l.append(mse_gap)
        print(f"MAE loss: {np.array(mae_l).mean():.3f} ± {np.array(mae_l).std():.3f}")
        print(f"NDCG: {np.array(ndcg_l).mean():.3f} ± {np.array(ndcg_l).std():.3f}")
        print(f"MSE loss: {np.array(mse_l).mean():.3f} ± {np.array(mse_l).std():.3f}")

        print(f"MAE gap: {np.array(mae_gap_l).mean():.3f} ± {np.array(mae_gap_l).std():.3f}")
        print(f"NDCG gap: {np.array(ndcg_gap_l).mean():.3f} ± {np.array(ndcg_gap_l).std():.3f}")
        print(f"MSE gap: {np.array(mse_gap_l).mean():.3f} ± {np.array(mse_gap_l).std():.3f}")
        
   



else:
    if not os.path.exists(f"./{model_dir}/{args.model}"):
        raise ValueError("The models have not been trained yet!")

    test_total_xs = pickle.load(open(f"./{data_dir}/test_total_xs", 'rb'))
    test_total_ys = pickle.load(open(f"./{data_dir}/test_total_ys", 'rb'))
    test_genders = pickle.load(open(f"./{data_dir}/test_gender", 'rb'))

    meta = MetaLearner(args)

    mae_l = []
    ndcg_l = []
    mse_l = []

    mae_gap_l = []
    ndcg_gap_l = []
    mse_gap_l = []

    for i in range(args.repeat_times):
        if args.model == 'Ours':
            meta.load_state_dict(torch.load(f"./{model_dir}/{args.model}/beta_{args.beta}_gamma_{args.gamma}/{i}.pl"))
        else:
            meta.load_state_dict(torch.load(f"./{model_dir}/{args.model}/{i}.pl"))
        mae, ndcg, mse, mae_gap, ndcg_gap, mse_gap = test(meta, test_total_xs, test_total_ys, test_genders, args.device)

        mae_l.append(mae)
        ndcg_l.append(ndcg)
        mse_l.append(mse)

        mae_gap_l.append(mae_gap)
        ndcg_gap_l.append(ndcg_gap)
        mse_gap_l.append(mse_gap)

    print(f"MAE loss: {np.array(mae_l).mean():.3f} ± {np.array(mae_l).std():.3f}")
    print(f"NDCG: {np.array(ndcg_l).mean():.3f} ± {np.array(ndcg_l).std():.3f}")
    print(f"MSE loss: {np.array(mse_l).mean():.3f} ± {np.array(mse_l).std():.3f}")

    print(f"MAE gap: {np.array(mae_gap_l).mean():.3f} ± {np.array(mae_gap_l).std():.3f}")
    print(f"NDCG gap: {np.array(ndcg_gap_l).mean():.3f} ± {np.array(ndcg_gap_l).std():.3f}")
    print(f"MSE gap: {np.array(mse_gap_l).mean():.3f} ± {np.array(mse_gap_l).std():.3f}")
