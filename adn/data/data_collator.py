import functools
import torch
from transformers import PreTrainedTokenizerFast


def data_collator(features: list, tokenizer: PreTrainedTokenizerFast) -> dict:
    input_ids = []
    labels = []
    chromosome_positions = []
    for f in features:
        input_ids.append(tokenizer.encode(f["input_ids"]))
        labels.append(f["labels"])
        chromosome_positions.append(f["chromosome_positions"])
    max_length = max(len(ids) for ids in input_ids)
    input_ids = torch.tensor(
        [ids + [tokenizer.pad_token_id] * (max_length - len(ids)) for ids in input_ids]
    ).long()

    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    chromosome_positions = torch.tensor(chromosome_positions).long()

    input_ids = torch.cat([input_ids, chromosome_positions], dim=1)
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
