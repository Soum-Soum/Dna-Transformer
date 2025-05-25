import functools
import torch
from transformers import PreTrainedTokenizerFast


def data_collator(features: list, tokenizer: PreTrainedTokenizerFast) -> dict:
    input_ids = []
    labels = []
    snp_positions = []
    snp_ids = []
    for f in features:
        input_ids.append(tokenizer.encode(f["input_ids"]))
        snp_positions.append(f["snp_positions"])
        snp_ids.append(f["snp_ids"])
        labels.append([f["labels"], f["family"]])

    max_length = max(len(ids) for ids in input_ids)
    input_ids = torch.tensor(
        [ids + [tokenizer.pad_token_id] * (max_length - len(ids)) for ids in input_ids]
    ).long()

    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    snp_positions = torch.tensor(snp_positions).long()

    snp_ids = torch.tensor(snp_ids).long()

    input_ids = torch.cat([input_ids, snp_positions, snp_ids], dim=1)
    labels = torch.tensor(labels)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def get_data_collator(
    tokenizer: PreTrainedTokenizerFast,
) -> callable:
    return functools.partial(
        data_collator,
        tokenizer=tokenizer,
    )
