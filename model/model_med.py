import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import *


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x


class MMC_Med(nn.Module):
    def __init__(self, args, input_dim_list):
        super(MMC_Med, self).__init__()

        self.args = args

        self.fe_1 = Feature_Encoder(input_dim_list[0], args.emb_dim)
        self.fe_2 = Feature_Encoder(input_dim_list[1], args.emb_dim)
        self.fe_3 = Feature_Encoder(input_dim_list[2], args.emb_dim)

        self.cl_1 = Classifier(
            args.mm_dropout, args.emb_dim, args.post_dim, args.output_dim
        )
        self.cl_2 = Classifier(
            args.mm_dropout, args.emb_dim, args.post_dim, args.output_dim
        )
        self.cl_3 = Classifier(
            args.mm_dropout, args.emb_dim, args.post_dim, args.output_dim
        )

        self.mm_classfier = Classifier(
            args.mm_dropout, args.emb_dim * 3, args.post_dim, args.output_dim
        )

    def forward(self, data_list=None, label=None, infer=False, use_soft_clip=False):

        criterion = torch.nn.CrossEntropyLoss(reduction="none")

        if infer:

            input_1 = data_list[0]
            input_2 = data_list[1]
            input_3 = data_list[2]

            emb_1 = self.fe_1(input_1)
            emb_2 = self.fe_2(input_2)
            emb_3 = self.fe_3(input_3)

            fusion = torch.cat([emb_1, emb_2, emb_3], dim=-1)
            output_mm = self.mm_classfier(fusion)

            return output_mm

        if not infer:

            input_1 = data_list[0]
            input_2 = data_list[1]
            input_3 = data_list[2]

            emb_1 = self.fe_1(input_1)
            emb_2 = self.fe_2(input_2)
            emb_3 = self.fe_3(input_3)

            output_1 = self.cl_1(emb_1)
            output_2 = self.cl_2(emb_2)
            output_3 = self.cl_3(emb_3)

            fusion = torch.cat([emb_1, emb_2, emb_3], dim=-1)
            output_mm = self.mm_classfier(fusion)

            MMloss_1 = criterion(output_1, label)
            MMloss_2 = criterion(output_2, label)
            MMloss_3 = criterion(output_3, label)
            MMloss_m = criterion(output_mm, label)

        if self.args.no_uni_pred:

            MMloss_1 = 0
            MMloss_2 = 0
            MMloss_3 = 0

        MMLoss_sum = MMloss_1 + MMloss_2 + MMloss_3 + MMloss_m

        if self.args.multi_mixup:

            if not use_soft_clip:
                print("m3co")
                input_1_new = input_1.clone()
                input_2_new = input_2.clone()
                input_3_new = input_3.clone()

                emb_1_new = self.fe_1(input_1_new)
                emb_2_new = self.fe_2(input_2_new)
                emb_3_new = self.fe_3(input_3_new)

                emb_1_mixup, perm_1, betas_1, select_1 = mixco_text(
                    emb_1_new,
                    beta=self.args.mixup_beta,
                    s_thresh=self.args.mixup_s_thresh,
                )
                emb_2_mixup, perm_2, betas_2, select_2 = mixco_text(
                    emb_2_new,
                    beta=self.args.mixup_beta,
                    s_thresh=self.args.mixup_s_thresh,
                )
                emb_3_mixup, perm_3, betas_3, select_3 = mixco_text(
                    emb_3_new,
                    beta=self.args.mixup_beta,
                    s_thresh=self.args.mixup_s_thresh,
                )

                MMLoss_Contrastive_1 = mixco_nce(
                    emb_1_mixup, emb_2_new, perm=perm_1, betas=betas_1
                )
                MMLoss_Contrastive_2 = mixco_nce(
                    emb_2_mixup, emb_3_new, perm=perm_2, betas=betas_2
                )
                MMLoss_Contrastive_3 = mixco_nce(
                    emb_3_mixup, emb_1_new, perm=perm_3, betas=betas_3
                )

                MMLoss_Contrastive = (
                    MMLoss_Contrastive_1 + MMLoss_Contrastive_2 + MMLoss_Contrastive_3
                )
                MMLoss_sum = MMLoss_sum + self.args.lambda_mixup * MMLoss_Contrastive

            else:
                print("softclip")
                MMLoss_Contrastive_1 = soft_clip_loss(emb_1, emb_2)
                MMLoss_Contrastive_2 = soft_clip_loss(emb_2, emb_3)
                MMLoss_Contrastive_3 = soft_clip_loss(emb_3, emb_1)

                MMLoss_Contrastive = (
                    MMLoss_Contrastive_1 + MMLoss_Contrastive_2 + MMLoss_Contrastive_3
                )
                MMLoss_sum = MMLoss_sum + self.args.lambda_mixup * MMLoss_Contrastive

        return MMLoss_sum, MMloss_m, output_mm

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit


class Classifier(nn.Module):
    def __init__(self, dropout, in_dim, post_dim, out_dim):
        super(Classifier, self).__init__()
        self.post_dropout = nn.Dropout(p=dropout)
        self.post_layer_1 = LinearLayer(in_dim, post_dim)
        self.post_layer_2 = LinearLayer(post_dim, out_dim)

    def forward(self, input):
        input_p1 = F.relu(self.post_layer_1(input), inplace=False)
        output = self.post_layer_2(input_p1)
        return output


class Feature_Encoder(nn.Module):
    def __init__(self, in_dim, emb_dim):
        super(Feature_Encoder, self).__init__()
        self.fe1 = LinearLayer(in_dim, emb_dim)
        self.fe2 = LinearLayer(emb_dim, emb_dim)

    def forward(self, input):
        output = self.fe1(input)
        output = F.relu(output, inplace=False)
        output = self.fe2(output)
        # output = F.relu(output, inplace=False)
        return output
