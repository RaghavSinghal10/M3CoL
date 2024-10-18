import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
import argparse
from model.utils import *
from model.model_med import MMC_Med
from src.config import Config
from torch import optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

__all__ = ["TrainModule"]

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="rosmap", help="support ROSMAP/BRCA")
parser.add_argument("--seeds", nargs="+", type=int, help="set seeds for multiple runs!")
parser.add_argument("--data_dir", type=str, help="path to load data")
parser.add_argument("--test_only", action="store_true", help="test only or not")
parser.add_argument("--device", type=str, default="cuda", help="device")
parser.add_argument("--seed", type=int, default=69, help="seed")
parser.add_argument("--multi_mixup", action="store_true", help="multi mixco or not")
parser.add_argument("--mixup_pct", type=float, default=0.33, help="mixup percentage")
parser.add_argument("--lambda_mixup", type=float, default=0.1, help="lambda for mixup")
parser.add_argument("--mixup_beta", type=float, default=0.15, help="beta for mixup")
parser.add_argument(
    "--mixup_s_thresh", type=float, default=0.5, help="s_thresh for mixup"
)
parser.add_argument(
    "--step_size", type=int, default=500, help="step size for lr scheduler"
)
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
parser.add_argument("--gamma", type=float, default=0.2, help="gamma for lr scheduler")

args = parser.parse_args()
config = Config(args)
args = config.get_config()


def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1, 1), 1)
    return y_onehot


def prepare_trte_data(data_folder):
    num_view = 3
    print(data_folder)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=",")
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=",")
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in range(1, num_view + 1):
        data_tr_list.append(
            np.loadtxt(os.path.join(data_folder, str(i) + "_tr.csv"), delimiter=",")
        )
        data_te_list.append(
            np.loadtxt(os.path.join(data_folder, str(i) + "_te.csv"), delimiter=",")
        )

    eps = 1e-10
    X_train_min = [
        np.min(data_tr_list[i], axis=0, keepdims=True) for i in range(len(data_tr_list))
    ]
    data_tr_list = [
        data_tr_list[i] - np.tile(X_train_min[i], [data_tr_list[i].shape[0], 1])
        for i in range(len(data_tr_list))
    ]
    data_te_list = [
        data_te_list[i] - np.tile(X_train_min[i], [data_te_list[i].shape[0], 1])
        for i in range(len(data_tr_list))
    ]
    X_train_max = [
        np.max(data_tr_list[i], axis=0, keepdims=True) + eps
        for i in range(len(data_tr_list))
    ]
    data_tr_list = [
        data_tr_list[i] / np.tile(X_train_max[i], [data_tr_list[i].shape[0], 1])
        for i in range(len(data_tr_list))
    ]
    data_te_list = [
        data_te_list[i] / np.tile(X_train_max[i], [data_te_list[i].shape[0], 1])
        for i in range(len(data_tr_list))
    ]

    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        data_tensor_list[i] = data_tensor_list[i].to(args.device)
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr + num_te)))
    data_train_list = []
    data_all_list = []
    data_test_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(
            torch.cat(
                (
                    data_tensor_list[i][idx_dict["tr"]].clone(),
                    data_tensor_list[i][idx_dict["te"]].clone(),
                ),
                0,
            )
        )
        data_test_list.append(data_tensor_list[i][idx_dict["te"]].clone())
    labels = np.concatenate((labels_tr, labels_te))
    return data_train_list, data_test_list, idx_dict, labels


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )


def train_epoch(data_list, label, model, optimizer, epoch):
    model.train()
    optimizer.zero_grad()

    if epoch < int(args.mixup_pct * args.num_epoch):
        loss, loss_m, logit = model(data_list, label, use_soft_clip=False)
        loss = torch.mean(loss)
    else:
        loss, loss_m, logit = model(data_list, label, use_soft_clip=True)
        loss = torch.mean(loss)

    loss.backward()
    optimizer.step()


def test_epoch(data_list, model):
    model.eval()
    with torch.no_grad():
        logit = model.infer(data_list)
        prob = F.softmax(logit, dim=1).data.cpu().numpy()
    return prob


def save_checkpoint(model, checkpoint_path, filename="checkpoint.pt"):
    os.makedirs(checkpoint_path, exist_ok=True)
    filename = os.path.join(checkpoint_path, filename)
    torch.save(model, filename)


def load_checkpoint(model, path):
    best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint)


def main():

    test_inverval = 10

    if args.dataset == "brca":
        data_folder = os.path.join(args.data_path, "BRCA")
        num_class = 5

    elif args.dataset == "rosmap":
        data_folder = os.path.join(args.data_path, "ROSMAP")
        num_class = 2

    data_tr_list, data_test_list, trte_idx, labels_trte = prepare_trte_data(data_folder)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    labels_tr_tensor = labels_tr_tensor.cuda()
    onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
    dim_list = [x.shape[1] for x in data_tr_list]

    model = MMC_Med(args, input_dim_list=dim_list).to(args.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr_mm, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args
    )

    print("\nTraining...")

    for epoch in range(args.num_epoch):
        train_epoch(data_tr_list, labels_tr_tensor, model, optimizer, epoch)
        scheduler.step()

        if epoch % test_inverval == 0:
            te_prob = test_epoch(data_test_list, model)
            print("\nTest: Epoch {:d}".format(epoch))
            if args.dataset == "rosmap":

                test_acc = accuracy_score(
                    labels_trte[trte_idx["te"]], te_prob.argmax(1)
                )
                test_f1 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                test_auc = roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:, 1])

            else:

                test_acc = accuracy_score(
                    labels_trte[trte_idx["te"]], te_prob.argmax(1)
                )
                test_f1_weighted = f1_score(
                    labels_trte[trte_idx["te"]], te_prob.argmax(1), average="weighted"
                )
                test_f1_macro = f1_score(
                    labels_trte[trte_idx["te"]], te_prob.argmax(1), average="macro"
                )


if __name__ == "__main__":
    main()
