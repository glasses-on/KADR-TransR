# prepare arguments (hyperparameters)

import argparse
import random
from time import time
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import torch
import torch.cuda
from sklearn.model_selection import train_test_split
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from data_loader.loader_kgcn import KGCNDataLoader, KGCNDataset
# from parser.parser_kgat import *
from model.KGCN import KGCN
from utils.log_helper import *
from utils.metrics import calc_metrics_at_k
from utils.model_helper import early_stopping, save_model


def parse_kgcn_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--seed', type=int, default=2019, help='Random seed.')
    parser.add_argument('--data_name', type=str, default='last-fm', help='which dataset to use')
    parser.add_argument('--aggregator', type=str, default='concat', help='which aggregator to use')
    parser.add_argument('--n_epochs', type=int, default=100, help='the number of epochs')
    parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
    parser.add_argument('--embed_dim', type=int, default=64, help='dimension of user and entity embeddings')
    parser.add_argument('--n_iter', type=int, default=1,
                        help='number of iterations when computing entity representation')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--ratio', type=float, default=0.8, help='size of training dataset')

    parser.add_argument('--data_dir', nargs='?', default='datasets/',
                        help='Input data path.')

    parser.add_argument('--use_pretrain', type=int, default=1,
                        help='0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with stored model.')
    parser.add_argument('--pretrain_embedding_dir', nargs='?', default='datasets/pretrain/',
                        help='Path of learned embeddings.')
    parser.add_argument('--pretrain_model_path', nargs='?', default='trained_model/model.pth',
                        help='Path of stored model.')

    parser.add_argument('--kg_batch_size', type=int, default=2048,
                        help='KG batch size.')
    parser.add_argument('--test_batch_size', type=int, default=3846,
                        help='Test batch size (the user number to test every batch).')

    parser.add_argument('--relation_dim', type=int, default=64,
                        help='Relation Embedding size.')

    parser.add_argument('--laplacian_type', type=str, default='random-walk',
                        help='Specify the type of the adjacency (laplacian) matrix from {symmetric, random-walk}.')
    parser.add_argument('--aggregation_type', type=str, default='bi-interaction',
                        help='Specify the type of the aggregation layer from {gcn, graphsage, bi-interaction}.')
    parser.add_argument('--conv_dim_list', nargs='?', default='[64, 32, 16]',
                        help='Output sizes of every aggregation layer.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1, 0.1]',
                        help='Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.')

    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--cf_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating CF l2 loss.')

    parser.add_argument('--stopping_steps', type=int, default=10,
                        help='Number of epoch for early stopping')

    parser.add_argument('--cf_print_every', type=int, default=1,
                        help='Iter interval of printing CF loss.')
    parser.add_argument('--kg_print_every', type=int, default=10,
                        help='Iter interval of printing KG loss.')
    parser.add_argument('--evaluate_every', type=int, default=1,
                        help='Epoch interval of evaluating CF.')

    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Calculate metric@K when evaluating.')

    parser.add_argument('--train_kg', type=int, default=1,
                        help='train_kg')

    args = parser.parse_args()

    save_dir = 'trained_model/KGCN/{}/embed-dim{}_relation-dim{}_{}_{}_{}_lr{}_pretrain{}/{}/'.format(
        args.data_name, args.embed_dim, args.relation_dim, args.laplacian_type, args.aggregation_type,
        '-'.join([str(i) for i in eval(args.conv_dim_list)]), args.lr, args.use_pretrain, args.exp_name)
    args.save_dir = save_dir

    return args


def evaluate(model, dataloader, Ks, device, test_loader):
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()

    cf_scores = []
    metric_names = ['precision', 'recall', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    n_items = dataloader.n_items
    item_ids_list = torch.arange(n_items, dtype=torch.long).to(device)

    # evaluate per every epoch
    with torch.no_grad():
        for user_ids, _, labels in test_loader:
            user_ids, labels = user_ids.to(device), labels.to(device)
            # print("user_ids.size")
            # print(user_ids.size(), item_ids.size())
            # print(item_ids)
            # print("item_ids_list")
            # print(item_ids_list.size())
            # print(item_ids_list)
            try:
                batch_scores = model(user_ids, item_ids_list, mode='predict')      # (n_batch_users, n_items)
            except:
                print("Problem with batch while evaluating {}".format(user_ids.size()))
                continue

            batch_scores = batch_scores.cpu()
            batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict, user_ids.cpu().numpy(), item_ids_list.cpu().numpy(), Ks)

            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])

    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()

    return cf_scores, metrics_dict


def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # build dataset and knowledge graph

    data_loader = KGCNDataLoader(args.data_name, args, None)
    kg = data_loader.load_kg_kgcn()
    df_dataset = data_loader.load_dataset()

    print("max kg")
    print(max(kg.keys()), len(kg), kg[max(kg.keys())])
    print(df_dataset["userID"].max())
    print(df_dataset["itemID"].max())

    x_train, x_test, y_train, y_test = train_test_split(df_dataset, df_dataset['label'], test_size=1 - args.ratio,
                                                        shuffle=True, random_state=999)
    train_dataset = KGCNDataset(x_train)
    test_dataset = KGCNDataset(x_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size)

    # prepare network, loss function, optimizer
    num_user, num_entity, num_relation = data_loader.get_num()
    print("num_relation {}".format(num_relation))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = KGCN(num_user, num_entity, num_relation, kg, args, device).to(device)
    criterion = torch.nn.BCELoss()

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    kg_optimizer = optim.Adam(net.parameters(), lr=args.lr)

    print('device: ', device)

    # train
    loss_list = []
    test_loss_list = []
    loss_kg_list = []

    auc_score_list = []
    cf_time_training = []
    kg_time_training = []

    # initialize metrics
    best_epoch = -1
    best_recall = 0

    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    epoch_list = []
    metrics_list = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in Ks}

    for epoch in range(args.n_epochs):

        # Train CF
        running_loss = 0.0
        time1 = time()
        for i, (user_ids, item_ids, labels) in enumerate(train_loader):
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = net(user_ids, item_ids, mode='train_cf')
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        this_training_time = time() - time1
        cf_time_training.append(this_training_time)

        # print train loss per every epoch
        print('[Epoch {}]train_loss: '.format(epoch + 1), running_loss / len(train_loader))
        loss_list.append(running_loss / len(train_loader))

        # Train KG
        time3 = time()
        kg_total_loss = 0
        n_kg_batch = data_loader.n_kg_train // data_loader.kg_batch_size + 1

        if args.train_kg == 1:
            for iter in range(1, n_kg_batch + 1):
                time4 = time()
                kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data_loader.generate_kg_batch(
                    data_loader.train_kg_dict, data_loader.kg_batch_size, data_loader.n_users_entities)
                kg_batch_head = kg_batch_head.to(device)
                kg_batch_relation = kg_batch_relation.to(device)
                kg_batch_pos_tail = kg_batch_pos_tail.to(device)
                kg_batch_neg_tail = kg_batch_neg_tail.to(device)

                kg_batch_loss = net(kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail,
                                    mode='train_kg')

                kg_batch_loss.backward()
                kg_optimizer.step()
                kg_optimizer.zero_grad()
                kg_total_loss += kg_batch_loss.item()

                if iter % 50 == 0:
                    torch.cuda.empty_cache()

                if (iter % args.kg_print_every) == 0:
                    logging.info(
                        'KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(
                            epoch, iter, n_kg_batch, time() - time4, kg_batch_loss.item(), kg_total_loss / iter))

            loss_kg_list.append(kg_total_loss / n_kg_batch)
            kg_time_training.append(time() - time3)
            logging.info(
                'KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch,
                                                                                                                  n_kg_batch,
                                                                                                                  time() - time3,
                                                                                                                  kg_total_loss / n_kg_batch))
        torch.cuda.empty_cache()

        # evaluate cf
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epochs:
            time6 = time()
            _, metrics_dict = evaluate(net, data_loader, Ks, device, test_loader)

            metrics_str = 'CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
                epoch, time() - time6, metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'],
                metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'],
                metrics_dict[k_max]['ndcg'])

            logging.info(metrics_str)
            temp_metrics_df = pd.DataFrame(data=[{"metrics": metrics_str}])
            temp_metrics_df.to_csv(args.save_dir + '/metrics_{}.tsv'.format(epoch), sep='\t', index=False)

            epoch_list.append(epoch)
            for k in Ks:
                for m in ['precision', 'recall', 'ndcg']:
                    metrics_list[k][m].append(metrics_dict[k][m])
            best_recall, _ = early_stopping(metrics_list[k_min]['recall'], args.stopping_steps)

            if metrics_list[k_min]['recall'].index(best_recall) == len(epoch_list) - 1:
                save_model(net, args.save_dir, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch

            with torch.no_grad():
                test_loss = 0
                total_roc = 0
                for user_ids, item_ids, labels in test_loader:
                    user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
                    outputs = net(user_ids, item_ids, mode="train_cf")
                    test_loss += criterion(outputs, labels).item()
                    total_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                print('[Epoch {}]test_loss: '.format(epoch + 1), test_loss / len(test_loader))
                print("ROC: {}".format(total_roc / len(test_loader)))
                test_loss_list.append(test_loss / len(test_loader))
                auc_score_list.append(total_roc / len(test_loader))

    # save metrics
    metrics_df = [epoch_list]
    metrics_cols = ['epoch_idx']
    for k in Ks:
        for m in ['precision', 'recall', 'ndcg']:
            metrics_df.append(metrics_list[k][m])
            metrics_cols.append('{}@{}'.format(m, k))
    metrics_df = pd.DataFrame(metrics_df).transpose()
    metrics_df.columns = metrics_cols
    metrics_df.to_csv(args.save_dir + '/metrics.tsv', sep='\t', index=False)

    # print best metrics
    best_metrics = metrics_df.loc[metrics_df['epoch_idx'] == best_epoch].iloc[0].to_dict()
    logging.info(
        'Best CF Evaluation: Epoch {:04d} | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
            int(best_metrics['epoch_idx']), best_metrics['precision@{}'.format(k_min)],
            best_metrics['precision@{}'.format(k_max)], best_metrics['recall@{}'.format(k_min)],
            best_metrics['recall@{}'.format(k_max)], best_metrics['ndcg@{}'.format(k_min)],
            best_metrics['ndcg@{}'.format(k_max)]))

    # Logging
    logging.info("loss_list {}".format(loss_list))
    logging.info("loss_kg_list {}".format(loss_kg_list))
    logging.info("test_loss_list {}".format(test_loss_list))
    logging.info("auc_score_list {}".format(auc_score_list))
    logging.info("cf_time_training {}".format(cf_time_training))
    logging.info("kg_time_training {}".format(kg_time_training))


if __name__ == '__main__':
    args = parse_kgcn_args()
    train(args)
    # predict(args)
