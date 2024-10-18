import os
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from sklearn.metrics import f1_score, accuracy_score
import argparse
from model.utils import *
from model.model import MMC
from data.dataloader import MMDataLoader
from src.metrics import collect_metrics
from src.functions import (
    save_checkpoint,
    load_checkpoint,
    dict_to_str,
    count_parameters,
)
from src.config import Config
import time

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

__all__ = ["TrainModule"]

torch.cuda.current_device()

parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, default="MMC", help="project name")
parser.add_argument(
    "--dataset", type=str, default="n24news", help="support N24News/Food101"
)
parser.add_argument(
    "--text_type",
    type=str,
    default="abstract",
    help="support headline/caption/abstract",
)
parser.add_argument("--lr_mm", type=float, default=1e-3, help="--lr_mm")
parser.add_argument("--min_epoch", type=int, default=1, help="min_epoch")
parser.add_argument("--valid_step", type=int, default=50, help="valid_step")
parser.add_argument(
    "--text_encoder",
    type=str,
    default="bert_base",
    help="bert_base/roberta_base/bert_large",
)
parser.add_argument(
    "--image_encoder", type=str, default="vit_base", help="vit_base/vit_large"
)
parser.add_argument("--data_dir", type=str, help="Data directory")
parser.add_argument(
    "--test_only", type=bool, default=False, help="train+test or test only"
)
parser.add_argument(
    "--pretrained_dir",
    type=str,
    default="./pretrained_models",
    help="path to pretrained models from Hugging Face.",
)
parser.add_argument(
    "--model_save_dir",
    type=str,
    default="Path/To/results/models",
    help="path to save model parameters.",
)
parser.add_argument(
    "--res_save_dir",
    type=str,
    default="Path/To/results/results",
    help="path to save training results.",
)
parser.add_argument(
    "--logs_dir", type=str, default="Path/To/results/logs", help="path to log results."
)  # NO
parser.add_argument(
    "--local_rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument("--seeds", nargs="+", type=int, help="set seeds for multiple runs!")
parser.add_argument(
    "--model_path",
    type=str,
    default="./Path/To/results/models",
    help="path to load model parameters",
)
parser.add_argument("--save_model", type=bool, default=True, help="save model or not")
parser.add_argument("--text_mixup", action="store_true", help="text mixup or not")
parser.add_argument("--image_mixup", action="store_true", help="img mixup or not")
parser.add_argument(
    "--image_embedding_mixup", action="store_true", help="image embedding mixup or not"
)
parser.add_argument("--alpha", type=float, default=0.2, help="alpha for mixup")
parser.add_argument("--multi_mixup", action="store_true", help="multi mixco or not")
parser.add_argument("--mixup_pct", type=float, default=0.33, help="mixup percentage")
parser.add_argument("--lambda_mixup", type=float, default=0.1, help="lambda for mixup")
parser.add_argument("--mixup_beta", type=float, default=0.15, help="beta for mixup")
parser.add_argument(
    "--mixup_s_thresh", type=float, default=0.5, help="s_thresh for mixup"
)
parser.add_argument("--no_uni_pred", action="store_true", help="no_uni_pred or not")

torch.cuda.empty_cache()

args = parser.parse_args()
config = Config(args)
args = config.get_config()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.local_rank == -1:
    device = torch.device("cuda")
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")

args.device = device
args.data_dir = os.path.join(args.data_dir, args.dataset)

args.best_model_save_path = os.path.join(
    args.model_save_dir, f'{args.dataset}-best-{time.strftime("%Y%m%d-%H%M%S")}.pth'
)

print(args)


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )


def get_optimizer(model, args):
    if args.mmc not in ["V"]:  # Goes in both if conditions with train_food.py

        text_enc_param = list(model.module.text_encoder.named_parameters())
        text_clf_param = list(model.module.text_classfier.parameters())
    if args.mmc not in ["T"]:

        img_enc_param = list(model.module.image_encoder.parameters())
        img_clf_param = list(model.module.image_classfier.parameters())
    mm_clf_param = list(model.module.mm_classfier.parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    if args.mmc in ["V"]:
        optimizer_grouped_parameters = [
            {
                "params": img_enc_param,
                "weight_decay": args.weight_decay_tfm,
                "lr": args.lr_img_tfm,
            },
            {
                "params": img_clf_param,
                "weight_decay": args.weight_decay_other,
                "lr": args.lr_img_cls,
            },
            {
                "params": mm_clf_param,
                "weight_decay": args.weight_decay_other,
                "lr": args.lr_mm_cls,
            },
        ]
    elif args.mmc in ["T"]:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in text_enc_param if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay_tfm,
                "lr": args.lr_text_tfm,
            },
            {
                "params": [
                    p for n, p in text_enc_param if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": args.lr_text_tfm,
            },
            {
                "params": text_clf_param,
                "weight_decay": args.weight_decay_other,
                "lr": args.lr_text_cls,
            },
            {
                "params": mm_clf_param,
                "weight_decay": args.weight_decay_other,
                "lr": args.lr_mm_cls,
            },
        ]
    else:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in text_enc_param if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay_tfm,
                "lr": args.lr_text_tfm,
            },
            {
                "params": [
                    p for n, p in text_enc_param if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": args.lr_text_tfm,
            },
            {
                "params": text_clf_param,
                "weight_decay": args.weight_decay_other,
                "lr": args.lr_text_cls,
            },
            {
                "params": img_enc_param,
                "weight_decay": args.weight_decay_tfm,
                "lr": args.lr_img_tfm,
            },
            {
                "params": img_clf_param,
                "weight_decay": args.weight_decay_other,
                "lr": args.lr_img_cls,
            },
            {
                "params": mm_clf_param,
                "weight_decay": args.weight_decay_other,
                "lr": args.lr_mm_cls,
            },
        ]
    optimizer = optim.Adam(optimizer_grouped_parameters)

    return optimizer


def valid(args, model, data=None, best_valid=None, nBetter=None, step=None):
    model.eval()
    with torch.no_grad():
        train_loader, valid_loader, test_loader = data
        y_pred = []
        y_true = []
        with tqdm(valid_loader) as td:
            for (
                batch_image,
                text_input_ids,
                text_token_type_ids,
                text_attention_mask,
                batch_label,
            ) in td:
                text = (
                    text_input_ids.to(args.device),
                    text_token_type_ids.to(args.device),
                    text_attention_mask.to(args.device),
                )
                image = batch_image.to(args.device)
                logit = model.module.infer(text, image, None)
                y_pred.append(logit.cpu())
                y_true.append(batch_label.cpu())
                # break
        logits = torch.cat(y_pred)
        te_true = torch.cat(y_true).data.cpu().numpy()

        te_prob = F.softmax(logits, dim=1).data.cpu().numpy()
        cur_valid = accuracy_score(te_true, te_prob.argmax(1))
        isBetter = cur_valid >= (best_valid + 1e-6)
        valid_results = {"step": step}
        valid_results.update(collect_metrics(args.dataset, te_true, te_prob))
        if isBetter:
            if args.local_rank in [0, -1]:
                save_checkpoint(model, args.best_model_save_path)
            best_valid = cur_valid
            nBetter = 0
        else:
            nBetter += 1
        return valid_results, best_valid, nBetter


def train_valid(args, model, optimizer, scheduler=None, data=None):

    model.train()
    best_valid = 1e-5
    nBetter = 0
    total_step = 0

    gradient_accumulation_steps = int(args.batch_gradient / args.batch_size)
    for epoch in range(args.num_epoch + 1):
        print("Epoch: ", epoch + 1)
        train_loader, valid_loader, test_loader = data
        y_pred = []
        y_true = []
        if args.local_rank not in [-1]:
            train_loader.sampler.set_epoch(epoch)
        with tqdm(train_loader) as td:
            for (
                batch_image,
                text_input_ids,
                text_token_type_ids,
                text_attention_mask,
                batch_label,
            ) in td:
                text = (
                    text_input_ids.to(args.device),
                    text_token_type_ids.to(args.device),
                    text_attention_mask.to(args.device),
                )
                image = batch_image.to(args.device)
                labels = batch_label.to(args.device).view(-1)
                if epoch < int(args.mixup_pct * args.num_epoch):
                    loss, loss_m, logit_m = model(
                        text, image, None, labels, use_soft_clip=False
                    )
                else:
                    loss, loss_m, logit_m = model(
                        text, image, None, labels, use_soft_clip=True
                    )
                loss = loss.sum()
                loss.backward()
                y_pred.append(logit_m.cpu())
                y_true.append(batch_label.cpu())
                total_step += 1

                if total_step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                if (
                    total_step % (args.valid_step * gradient_accumulation_steps) == 0
                ) and (epoch > args.min_epoch):
                    valid_results, best_valid, nBetter = valid(
                        args, model, data, best_valid, nBetter, total_step
                    )

                    load_checkpoint(model, args.best_model_save_path)
                    te_prob, te_true = test_epoch(model, test_loader)
                    best_results = collect_metrics(args.dataset, te_true, te_prob)

                    accuracy_test = best_results["acc"]

                    if nBetter < 1:
                        best_results = valid_results
                    if nBetter > args.patience:
                        pass

            logits = torch.cat(y_pred)
            tr_true = torch.cat(y_true).data.cpu().numpy()

            tr_prob = F.softmax(logits, dim=1).data.cpu().numpy()
            tuning_metric = accuracy_score(tr_true, tr_prob.argmax(1))

            scheduler.step(tuning_metric)

    return valid_results


def test_epoch(model, dataloader=None):
    model.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        with tqdm(dataloader) as td:
            for (
                batch_image,
                text_input_ids,
                text_token_type_ids,
                text_attention_mask,
                batch_label,
            ) in td:
                text = (
                    text_input_ids.cuda(),
                    text_token_type_ids.cuda(),
                    text_attention_mask.cuda(),
                )
                image = batch_image.cuda()
                logit = model.module.infer(text, image, None)
                y_pred.append(logit.cpu())
                y_true.append(batch_label.cpu())
        logits = torch.cat(y_pred)
        true = torch.cat(y_true).data.cpu().numpy()

        prob = F.softmax(logits, dim=1).data.cpu().numpy()
    return prob, true


def main():

    train_loader, valid_loader, test_loader = MMDataLoader(args)
    print(
        f"Train: {len(train_loader.dataset)}, Valid: {len(valid_loader.dataset)}, Test: {len(test_loader.dataset)}"
    )

    if args.local_rank in [-1]:
        model = DataParallel(MMC(args))
        model = model.to(args.device)
    else:
        model = MMC(args).to(args.device)
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    data = train_loader, valid_loader, test_loader

    if args.local_rank in [-1, 0]:
        print(f"\nThe model has {count_parameters(model)} trainable parameters")
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    if not args.test_only:
        if args.local_rank in [-1, 0]:
            print("Start training...")
        best_results = train_valid(args, model, optimizer, scheduler, data)


if __name__ == "__main__":
    main()
