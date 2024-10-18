import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.TextEncoder import *
from model.ImageEncoder import *
from model.utils import *


def mixup_data(
    input_image,
    text_embedding,
    y,
    alpha,
    mixup_image=True,
    mixup_text=False,
    use_cuda=True,
):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    mixed_text_embedding = None
    mixed_input_image = None

    batch_size = input_image.size()[0]

    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    if mixup_image:
        mixed_input_image = lam * input_image + (1 - lam) * input_image[index, :]

    if mixup_text:
        mixed_text_embedding = (
            lam * text_embedding + (1 - lam) * text_embedding[index, :]
        )

    y_a, y_b = y, y[index]

    return mixed_input_image, mixed_text_embedding, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam, mixing=False):

    if mixing:
        output = ((1 + lam) / 2) * criterion(pred, y_a) + ((1 - lam) / 2) * criterion(
            pred, y_b
        )
    else:
        output = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    return output


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


class MMC(nn.Module):
    def __init__(self, args):
        super(MMC, self).__init__()
        # text subnets
        self.args = args
        if self.args.mmc not in ["T"]:
            self.image_encoder = ImageEncoder(
                pretrained_dir=args.pretrained_dir, image_encoder=args.image_encoder
            )
            self.image_classfier = Classifier(
                args.img_dropout, args.img_out, args.post_dim, args.output_dim
            )
        if self.args.mmc not in ["V"]:
            self.text_encoder = TextEncoder(
                pretrained_dir=args.pretrained_dir, text_encoder=args.text_encoder
            )
            self.text_classfier = Classifier(
                args.text_dropout, args.text_out, args.post_dim, args.output_dim
            )
        self.mm_classfier = Classifier(
            args.mm_dropout,
            args.text_out + args.img_out,
            args.post_dim,
            args.output_dim,
        )

    def forward(
        self,
        text=None,
        image=None,
        data_list=None,
        label=None,
        infer=False,
        use_soft_clip=False,
    ):

        criterion = torch.nn.CrossEntropyLoss(reduction="none")

        if not infer:
            text = self.text_encoder(text=text)
            image = torch.squeeze(image, 1)

            if self.args.image_embedding_mixup:
                image = self.image_encoder(pixel_values=image)

            if not self.args.image_mixup and not self.args.text_mixup:
                mixed_input_image, mixed_text_embedding, y_a, y_b, lam = (
                    image,
                    text,
                    label,
                    label,
                    1,
                )

            elif self.args.image_mixup and not self.args.text_mixup:
                mixed_input_image, mixed_text_embedding, y_a, y_b, lam = mixup_data(
                    image,
                    text,
                    label,
                    alpha=self.args.alpha,
                    mixup_image=self.args.image_mixup,
                    mixup_text=self.args.text_mixup,
                    use_cuda=True,
                )
                mixed_text_embedding = text

            elif not self.args.image_mixup and self.args.text_mixup:
                mixed_input_image, mixed_text_embedding, y_a, y_b, lam = mixup_data(
                    image,
                    text,
                    label,
                    alpha=self.args.alpha,
                    mixup_image=self.args.image_mixup,
                    mixup_text=self.args.text_mixup,
                    use_cuda=True,
                )
                mixed_input_image = image

            else:
                mixed_input_image, mixed_text_embedding, y_a, y_b, lam = mixup_data(
                    image,
                    text,
                    label,
                    alpha=self.args.alpha,
                    mixup_image=self.args.image_mixup,
                    mixup_text=self.args.text_mixup,
                    use_cuda=True,
                )

            if not self.args.image_embedding_mixup:
                image = self.image_encoder(pixel_values=mixed_input_image)

            output_text = self.text_classfier(mixed_text_embedding[:, 0, :])
            output_image = self.image_classfier(image[:, 0, :])

            fusion = torch.cat([text[:, 0, :], image[:, 0, :]], dim=-1)
            output_mm = self.mm_classfier(fusion)

        if infer:
            text = self.text_encoder(text=text)
            image = torch.squeeze(image, 1)
            image = self.image_encoder(pixel_values=image)
            output_text = self.text_classfier(text[:, 0, :])
            output_image = self.image_classfier(image[:, 0, :])
            fusion = torch.cat([text[:, 0, :], image[:, 0, :]], dim=-1)
            output_mm = self.mm_classfier(fusion)
            return output_mm

        if not self.args.image_mixup and not self.args.text_mixup:
            MMLoss_m = torch.mean(criterion(output_mm, label))
            MMLoss_text = torch.mean(criterion(output_text, label))
            MMLoss_image = torch.mean(criterion(output_image, label))

        elif self.args.image_mixup and not self.args.text_mixup:
            MMLoss_m = torch.mean(
                mixup_criterion(criterion, output_mm, y_a, y_b, lam, mixing=True)
            )
            MMLoss_text = torch.mean(criterion(output_text, label))
            MMLoss_image = torch.mean(
                mixup_criterion(criterion, output_image, y_a, y_b, lam)
            )

        elif not self.args.image_mixup and self.args.text_mixup:
            MMLoss_m = torch.mean(
                mixup_criterion(criterion, output_mm, y_a, y_b, lam, mixing=True)
            )
            MMLoss_text = torch.mean(
                mixup_criterion(criterion, output_text, y_a, y_b, lam)
            )
            MMLoss_image = torch.mean(criterion(output_image, label))

        else:
            MMLoss_m = torch.mean(mixup_criterion(criterion, output_mm, y_a, y_b, lam))
            MMLoss_text = torch.mean(
                mixup_criterion(criterion, output_text, y_a, y_b, lam)
            )
            MMLoss_image = torch.mean(
                mixup_criterion(criterion, output_image, y_a, y_b, lam)
            )

        if self.args.no_uni_pred:
            MMLoss_text = 0
            MMLoss_image = 0

        MMLoss_sum = MMLoss_text + MMLoss_image + MMLoss_m

        if self.args.multi_mixup:

            if not use_soft_clip:

                # take a clone of text
                text_new = text.clone()
                image_new = image.clone()

                text_mixup, perm_text, betas_text, select_text = mixco_text(
                    text_new[:, 0, :],
                    beta=self.args.mixup_beta,
                    s_thresh=self.args.mixup_s_thresh,
                )
                image_mixup, perm_image, betas_image, select_image = mixco_image(
                    image_new[:, 0, :],
                    beta=self.args.mixup_beta,
                    s_thresh=self.args.mixup_s_thresh,
                )

                if self.args.mix_only_images:
                    MMLoss_Contrastive_text = 0
                    MMLoss_Contrastive_image = mixco_nce(
                        image_mixup,
                        text_new[:, 0, :],
                        perm=perm_image,
                        betas=betas_image,
                    )

                if self.args.mix_only_text:
                    MMLoss_Contrastive_text = mixco_nce(
                        text_mixup, image_new[:, 0, :], perm=perm_text, betas=betas_text
                    )
                    MMLoss_Contrastive_image = 0

                if not self.args.mix_only_images and not self.args.mix_only_text:
                    MMLoss_Contrastive_text = mixco_nce(
                        text_mixup, image_new[:, 0, :], perm=perm_text, betas=betas_text
                    )
                    MMLoss_Contrastive_image = mixco_nce(
                        image_mixup,
                        text_new[:, 0, :],
                        perm=perm_image,
                        betas=betas_image,
                    )

                MMLoss_Contrastive = MMLoss_Contrastive_text + MMLoss_Contrastive_image
                MMLoss_sum = MMLoss_sum + self.args.lambda_mixup * MMLoss_Contrastive

            else:
                if self.args.mix_only_images:
                    MMLoss_Contrastive_text = 0
                    MMLoss_Contrastive_image = soft_clip_loss(
                        image[:, 0, :], text[:, 0, :]
                    )

                if self.args.mix_only_text:
                    MMLoss_Contrastive_text = soft_clip_loss(
                        text[:, 0, :], image[:, 0, :]
                    )
                    MMLoss_Contrastive_image = 0

                if not self.args.mix_only_images and not self.args.mix_only_text:
                    MMLoss_Contrastive_text = soft_clip_loss(
                        text[:, 0, :], image[:, 0, :]
                    )
                    MMLoss_Contrastive_image = soft_clip_loss(
                        image[:, 0, :], text[:, 0, :]
                    )

                MMLoss_Contrastive = MMLoss_Contrastive_text + MMLoss_Contrastive_image
                MMLoss_sum = MMLoss_sum + self.args.lambda_mixup * MMLoss_Contrastive

        return MMLoss_sum, MMLoss_m, output_mm

    def infer(self, text=None, image=None, data_list=None):
        MMlogit = self.forward(text, image, data_list, infer=True)
        return MMlogit


class Classifier(nn.Module):
    def __init__(self, dropout, in_dim, post_dim, out_dim):
        super(Classifier, self).__init__()
        self.post_dropout = nn.Dropout(p=dropout)
        self.post_layer_1 = LinearLayer(in_dim, post_dim)
        self.post_layer_2 = LinearLayer(post_dim, post_dim)
        self.post_layer_3 = LinearLayer(post_dim, out_dim)

    def forward(self, input):
        input_p1 = F.relu(self.post_layer_1(input), inplace=False)
        input_d = self.post_dropout(input_p1)
        input_p2 = F.relu(self.post_layer_2(input_d), inplace=False)
        output = self.post_layer_3(input_p2)
        return output
