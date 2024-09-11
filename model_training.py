import random
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import heapq, itertools
from meta import MetaLearner, user_preference_estimator

def set_seed(seed, cudnn=True):
    """
    Seed everything we can!
    Note that gym environments might need additional seeding (env.seed(seed)),
    and num_workers needs to be set to 1.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # note: the below slows down the code but makes it reproducible
    if (seed is not None) and cudnn:
        torch.backends.cudnn.deterministic = True



def fair_dynamic_train(meta, total_xs, total_ys, 
                       genders, gamma, num_epoch, batch_sz, beta, device):

    train_total_data = list(zip(total_xs, total_ys))
    genders = np.array(genders)
    F_indice = np.where(genders == 1)[0]
    M_indice = np.where(genders == 0)[0]


    F_data_train = [train_total_data[idx] for idx in F_indice]
    M_data_train = [train_total_data[idx] for idx in M_indice]

    p0 = 0.5

    last_gap = 0

    for epoch in tqdm(range(num_epoch)):

        random.shuffle(F_data_train)
        random.shuffle(M_data_train)

        M_idx = 0
        F_idx = 0
        
        num_samples = len(F_data_train) + len(M_data_train)
        num_batch = num_samples // batch_sz

        meta.train()
        for _ in range(num_batch):

            M_sample_num = (np.random.uniform(0, 1, batch_sz) < p0).astype(int).sum()
            F_sample_num = batch_sz - M_sample_num

            if M_idx + M_sample_num >= len(M_data_train):
                M_batch = M_data_train[M_idx:] + M_data_train[:M_sample_num - (len(M_data_train) - M_idx)]
                M_idx = M_sample_num - (len(M_data_train) - M_idx)
            else:
                M_batch = M_data_train[M_idx:M_idx + M_sample_num]	
                M_idx += M_sample_num
            
            if F_idx + F_sample_num >= len(F_data_train):
                F_batch = F_data_train[F_idx:] + F_data_train[:F_sample_num - (len(F_data_train) - F_idx)]
                F_idx = F_sample_num - (len(F_data_train) - F_idx)
            else:
                F_batch = F_data_train[F_idx:F_idx + F_sample_num]
                F_idx += F_sample_num

            batch_data = F_batch + M_batch
            random.shuffle(batch_data)

            supp_xs, supp_ys = zip(*batch_data)
            meta.global_update(supp_xs, supp_ys)

        meta.eval()

    # ---------------------------------
        M_losses = []
        for total_x, total_y in M_data_train:

            num_records = len(total_x)
            indice = list(range(num_records))
            # random.shuffle(indice)

            sup_x = total_x[indice[:-10]].to(device)
            sup_y = total_y[indice[:-10]].to(device)
            qry_x = total_x[indice[-10:]].to(device)
            qry_y = total_y[indice[-10:]].to(device)

            qry_y_hat = meta.forward(sup_x, sup_y, qry_x)
            M_losses.append(F.l1_loss(qry_y_hat, qry_y))
        M_mean_loss = torch.stack(M_losses).mean()

        F_losses = []
        for total_x, total_y in F_data_train:

            num_records = len(total_x)
            indice = list(range(num_records))
            # random.shuffle(indice)

            sup_x = total_x[indice[:-10]].to(device)
            sup_y = total_y[indice[:-10]].to(device)
            qry_x = total_x[indice[-10:]].to(device)
            qry_y = total_y[indice[-10:]].to(device)

            qry_y_hat = meta.forward(sup_x, sup_y, qry_x)
            F_losses.append(F.l1_loss(qry_y_hat, qry_y))
        F_mean_loss = torch.stack(F_losses).mean()


        print(f"Male train loss: {M_mean_loss:.5f}")
        print(f"Female train loss: {F_mean_loss:.5f}")

        # if epoch == 0:
        #     last_gap = M_mean_loss - F_mean_loss 
        #     prev_state_dict = meta.state_dict()
        # else:
        #     if last_gap * (M_mean_loss - F_mean_loss) < 0 and abs(last_gap) < abs(M_mean_loss - F_mean_loss):
        #         beta = beta / 2
        #         meta.load_state_dict(prev_state_dict)
        #         p0 -= beta * last_gap.item()

        #         if p0 > 1:
        #             p0 = 1
        #         elif p0 < 0:
        #             p0 = 0
        #         print(f"{p0:.5f} {(1 - p0):.5f}")

        #         continue
        #     else:
        #         last_gap = M_mean_loss - F_mean_loss 
        #         prev_state_dict = meta.state_dict()
        if F_mean_loss - M_mean_loss < 0:
            sign = -1
        else:
            sign = 1
            
        p0 -= sign * min(gamma * (F_mean_loss - M_mean_loss).abs().item(), beta)


        if p0 > 1:
            p0 = 1
        elif p0 < 0:
            p0 = 0


        print(f"{p0:.5f} {(1 - p0):.5f}")



def fair_dynamic_train_metaCS(meta, total_xs, total_ys, 
                       genders, gamma, num_epoch, batch_sz, beta, device):

    train_total_data = list(zip(total_xs, total_ys))
    genders = np.array(genders)
    F_indice = np.where(genders == 1)[0]
    M_indice = np.where(genders == 0)[0]


    F_data_train = [train_total_data[idx] for idx in F_indice]
    M_data_train = [train_total_data[idx] for idx in M_indice]

    p0 = 0.5

    last_gap = 0

    for epoch in tqdm(range(num_epoch)):

        random.shuffle(F_data_train)
        random.shuffle(M_data_train)

        M_idx = 0
        F_idx = 0
        
        num_samples = len(F_data_train) + len(M_data_train)
        num_batch = num_samples // batch_sz

        meta.train()
        for _ in range(num_batch):

            M_sample_num = (np.random.uniform(0, 1, batch_sz) < p0).astype(int).sum()
            F_sample_num = batch_sz - M_sample_num

            if M_idx + M_sample_num >= len(M_data_train):
                M_batch = M_data_train[M_idx:] + M_data_train[:M_sample_num - (len(M_data_train) - M_idx)]
                M_idx = M_sample_num - (len(M_data_train) - M_idx)
            else:
                M_batch = M_data_train[M_idx:M_idx + M_sample_num]	
                M_idx += M_sample_num
            
            if F_idx + F_sample_num >= len(F_data_train):
                F_batch = F_data_train[F_idx:] + F_data_train[:F_sample_num - (len(F_data_train) - F_idx)]
                F_idx = F_sample_num - (len(F_data_train) - F_idx)
            else:
                F_batch = F_data_train[F_idx:F_idx + F_sample_num]
                F_idx += F_sample_num

            batch_data = F_batch + M_batch
            random.shuffle(batch_data)

            supp_xs, supp_ys = zip(*batch_data)
            meta.global_update_metaCS(supp_xs, supp_ys)

        meta.eval()

    # ---------------------------------
        M_losses = []
        for total_x, total_y in M_data_train:

            num_records = len(total_x)
            indice = list(range(num_records))
            # random.shuffle(indice)

            sup_x = total_x[indice[:-10]].to(device)
            sup_y = total_y[indice[:-10]].to(device)
            qry_x = total_x[indice[-10:]].to(device)
            qry_y = total_y[indice[-10:]].to(device)

            qry_y_hat = meta.forward(sup_x, sup_y, qry_x)
            M_losses.append(F.l1_loss(qry_y_hat, qry_y))
        M_mean_loss = torch.stack(M_losses).mean()

        F_losses = []
        for total_x, total_y in F_data_train:

            num_records = len(total_x)
            indice = list(range(num_records))
            # random.shuffle(indice)

            sup_x = total_x[indice[:-10]].to(device)
            sup_y = total_y[indice[:-10]].to(device)
            qry_x = total_x[indice[-10:]].to(device)
            qry_y = total_y[indice[-10:]].to(device)

            qry_y_hat = meta.forward(sup_x, sup_y, qry_x)
            F_losses.append(F.l1_loss(qry_y_hat, qry_y))
        F_mean_loss = torch.stack(F_losses).mean()


        print(f"Male train loss: {M_mean_loss:.5f}")
        print(f"Female train loss: {F_mean_loss:.5f}")

        if F_mean_loss - M_mean_loss < 0:
            sign = -1
        else:
            sign = 1
            
        p0 -= sign * min(gamma * (F_mean_loss - M_mean_loss).abs().item(), beta)


        if p0 > 1:
            p0 = 1
        elif p0 < 0:
            p0 = 0

        print(f"{p0:.5f} {(1 - p0):.5f}")


def ndcg(y_pred, y, topk=3):
    ele_idx = heapq.nlargest(topk, zip(y_pred, itertools.count()))
    pred_index = np.array([idx for ele, idx in ele_idx], dtype=np.intc)
    rel_pred = []
    for i in pred_index:
        rel_pred.append(y[i])
    rel = heapq.nlargest(topk, y)
    idcg = np.cumsum((np.power(rel, 2)-1) / np.log2(np.arange(2, topk + 2)))
    dcg = np.cumsum((np.power(rel_pred, 2)-1) / np.log2(np.arange(2, topk + 2)))
    ndcg = dcg/idcg
    return ndcg[-1]

def test(meta, total_xs, total_ys, genders, device):
    meta.eval()
    mae_losses = []
    ndcgs = []
    mse_losses = []
    mae_losses_gender = [[], []]
    ndcgs_gender = [[], []]
    mse_losses_gender = [[], []]

    for total_x, total_y, gender in tqdm(zip(total_xs, total_ys, genders)):

        num_records = len(total_x)
        indice = list(range(num_records))
        # random.shuffle(indice)

        sup_x = total_x[indice[:-10]].to(device)
        sup_y = total_y[indice[:-10]].to(device)
        qry_x = total_x[indice[-10:]].to(device)
        qry_y = total_y[indice[-10:]].to(device)


        qry_y_hat = meta(sup_x, sup_y, qry_x)

        mae_loss = F.l1_loss(qry_y_hat, qry_y)
        mse_loss = F.mse_loss(qry_y_hat, qry_y)

        mae_losses.append(mae_loss)
        mae_losses_gender[gender].append(mae_loss)

        mse_losses.append(mse_loss)
        mse_losses_gender[gender].append(mse_loss)

        true_rate = qry_y.detach().cpu().numpy().flatten()
        pred_rate = qry_y_hat.detach().cpu().numpy().flatten()
        ndcg_score = ndcg(pred_rate, true_rate)
        ndcgs.append(ndcg_score)
        ndcgs_gender[gender].append(ndcg_score)


    M_mean_mae = torch.stack(mae_losses_gender[0]).mean().item()
    F_mean_mae = torch.stack(mae_losses_gender[1]).mean().item()

    M_mean_ndcg = np.array(ndcgs_gender[0]).mean()
    F_mean_ndcg = np.array(ndcgs_gender[1]).mean()

    M_mean_mse = torch.stack(mse_losses_gender[0]).mean().item()
    F_mean_mse = torch.stack(mse_losses_gender[1]).mean().item()

    print(f"Test MAE loss: {torch.stack(mae_losses).mean().item():.3f}")
    print(f"Test NDCG: {np.array(ndcgs).mean():.3f}")
    print(f"Test MSE loss: {torch.stack(mse_losses).mean().item():.3f}")
    print(f"MAE gap: {abs(M_mean_mae - F_mean_mae):.3f}")
    print(f"NDCG gap: {abs(M_mean_ndcg - F_mean_ndcg):.3f}")
    print(f"MSE gap: {abs(M_mean_mse - F_mean_mse):.3f}")

    return (torch.stack(mae_losses).mean().item(), np.array(ndcgs).mean(), 
            torch.stack(mse_losses).mean().item(), abs(M_mean_mae - F_mean_mae),
            abs(M_mean_ndcg - F_mean_ndcg), abs(M_mean_mse - F_mean_mse))

