from pathlib import Path
from typing import Optional
from loguru import logger
from tokenizers import (
    models,
    pre_tokenizers,
    processors,
    Tokenizer,
)
from transformers import PreTrainedTokenizerFast

DNA_PAIRS_FOR_REF = {
    "AA",
    "TT",
    "CC",
    "GG",
}

DNA_PAIRS_FOR_ALT = {
    "AA",
    "AC",
    "AG",
    "AT",
    "CA",
    "CC",
    "CG",
    "CT",
    "GA",
    "GC",
    "GG",
    "GT",
    "TA",
    "TC",
    "TG",
    "TT",
}

BOS_TOKEN_STR = "<BOS>"
EOS_TOKEN_STR = "<EOS>"
PAD_TOKEN_STR = "<PAD>"
UNK_TOKEN_STR = "<UNK>"
CLS_TOKEN_STR = "<CLS>"


def get_vocabulary_dict() -> dict[str, int]:
    token_to_id = {
        BOS_TOKEN_STR: 0,
        EOS_TOKEN_STR: 1,
        PAD_TOKEN_STR: 2,
        UNK_TOKEN_STR: 3,
        CLS_TOKEN_STR: 4,
    }
    for pair in sorted(DNA_PAIRS_FOR_REF):
        for pair2 in sorted(DNA_PAIRS_FOR_ALT):
            token_to_id[pair + pair2] = len(token_to_id)

    return token_to_id


def _wrap_tokenizer(tokenizer: Tokenizer) -> PreTrainedTokenizerFast:
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    bos_token_id = tokenizer.token_to_id(BOS_TOKEN_STR)
    eos_token_id = tokenizer.token_to_id(EOS_TOKEN_STR)
    pad_token_id = tokenizer.token_to_id(PAD_TOKEN_STR)
    unk_token_id = tokenizer.token_to_id(UNK_TOKEN_STR)
    cls_token_id = tokenizer.token_to_id(CLS_TOKEN_STR)

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{BOS_TOKEN_STR} $A {CLS_TOKEN_STR}",
        special_tokens=[
            (cls_token_id, CLS_TOKEN_STR),
            (bos_token_id, BOS_TOKEN_STR),
            (eos_token_id, EOS_TOKEN_STR),
            (pad_token_id, PAD_TOKEN_STR),
            (unk_token_id, UNK_TOKEN_STR),
        ],
    )

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=UNK_TOKEN_STR,
        cls_token=CLS_TOKEN_STR,
        pad_token=PAD_TOKEN_STR,
        bos_token=BOS_TOKEN_STR,
        eos_token=EOS_TOKEN_STR,
    )


def get_one_snp_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(
        model=models.WordLevel(
            vocab=get_vocabulary_dict(),
            unk_token=UNK_TOKEN_STR,
        )
    )
    tokenizer = _wrap_tokenizer(tokenizer)
    logger.info(
        f"Tokenizer OneSNP loaded with vocab size: {len(tokenizer.get_vocab())}"
    )
    return tokenizer


def get_one_snp_bpe_tokenizer(tokenizer_path: Path) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer = _wrap_tokenizer(tokenizer)
    logger.info(
        f"Tokenizer loaded from {tokenizer_path} with vocab size: {len(tokenizer.get_vocab())}"
    )
    return tokenizer


def get_tokenizer(
    tokenizer_path: Optional[Path] = None,
) -> PreTrainedTokenizerFast:
    if tokenizer_path is None:
        return get_one_snp_tokenizer()
    else:
        return get_one_snp_bpe_tokenizer(tokenizer_path)
