import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    BertTokenizer,
    BertModel,
    RobertaTokenizer,
    RobertaModel,
    AutoTokenizer,
    AutoModel,
)

__all__ = ["TextEncoder"]

# bert base model from https://huggingface.co/bert-base-uncased
# bert large model from https://huggingface.co/bert-large-uncased
# roberta base model from https://huggingface.co/roberta-base
# roberta large model from https://huggingface.co/roberta-large


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class TextEncoder(nn.Module):
    def __init__(self, pretrained_dir, text_encoder="base"):
        """
        txt_encoder: base / large
        """
        super(TextEncoder, self).__init__()

        assert text_encoder in [
            "bert_base",
            "bert_large",
            "roberta_base",
            "roberta_large",
            "all-mpnet-base-v2",
            "multi-qa-mpnet-base-dot-v1",
        ]
        self.text_encoder = text_encoder

        # directory is fine
        if text_encoder in ["bert_base"]:
            tokenizer = BertTokenizer
            model = BertModel
            self.tokenizer = tokenizer.from_pretrained(
                pretrained_dir + "/bert_base_uncased/", do_lower_case=True
            )
            self.model = model.from_pretrained(pretrained_dir + "/bert_base_uncased/")
            # self.tokenizer = tokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            # self.model = model.from_pretrained('bert-base-uncased')
        elif text_encoder in ["bert_large"]:
            tokenizer = BertTokenizer
            model = BertModel
            self.tokenizer = tokenizer.from_pretrained(
                pretrained_dir + "/bert_large_uncased/", do_lower_case=True
            )
            self.model = model.from_pretrained(pretrained_dir + "/bert_large_uncased/")
        elif text_encoder in ["roberta_base"]:
            tokenizer = RobertaTokenizer
            model = RobertaModel
            self.tokenizer = tokenizer.from_pretrained(
                pretrained_dir + "/roberta_base/"
            )
            self.model = model.from_pretrained(pretrained_dir + "/roberta_base/")

        elif text_encoder in ["multi-qa-mpnet-base-dot-v1"]:
            tokenizer = AutoTokenizer
            model = AutoModel
            self.tokenizer = tokenizer.from_pretrained(
                pretrained_dir + "/multi-qa-mpnet-base-dot-v1/"
            )
            self.model = model.from_pretrained(
                pretrained_dir + "/multi-qa-mpnet-base-dot-v1/"
            )
        else:
            tokenizer = RobertaTokenizer
            model = RobertaModel
            self.tokenizer = tokenizer.from_pretrained(
                pretrained_dir + "/roberta_large/"
            )
            self.model = model.from_pretrained(pretrained_dir + "/roberta_large/")

    def get_tokenizer(self):
        return self.tokenizer

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def get_tokenize(self):
        return self.tokenizer.tokenize

    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        if "roberta" in self.text_encoder:
            input_ids = torch.squeeze(text[0], 1)
            input_mask = torch.squeeze(text[2], 1)
            # input_ids, input_mask = input_ids, attention_mask
            last_hidden_states = self.model(
                input_ids=input_ids, attention_mask=input_mask
            )[0]

        elif "all-mpnet-base-v2" in self.text_encoder:
            input_ids = torch.squeeze(text[0], 1)
            input_mask = torch.squeeze(text[2], 1)
            # input_ids, input_mask = input_ids, attention_mask
            last_hidden_states = self.model(
                input_ids=input_ids, attention_mask=input_mask
            )[0]

        else:
            input_ids = torch.squeeze(text[0], 1)
            input_mask = torch.squeeze(text[2], 1)
            segment_ids = torch.squeeze(text[1], 1)
            # input_ids, input_mask, segment_ids = input_ids, attention_mask, token_type_ids
            last_hidden_states = self.model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
            )[0]
            # print(f'last hidden state size: {last_hidden_states.shape}')
            # exit()

        return last_hidden_states


if __name__ == "__main__":
    text_normal = TextEncoder()
