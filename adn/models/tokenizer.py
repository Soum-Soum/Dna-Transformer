from tokenizers import (
    models,
    pre_tokenizers,
    processors,
    Tokenizer,
)
from transformers import PreTrainedTokenizerFast

DNA_PAIRS = {
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
    for pair in sorted(DNA_PAIRS):
        for pair2 in sorted(DNA_PAIRS):
            token_to_id[pair + pair2] = len(token_to_id)

    return token_to_id


def get_tokenizer() -> PreTrainedTokenizerFast:

    tokenizer = Tokenizer(
        model=models.WordLevel(
            vocab=get_vocabulary_dict(),
            unk_token=UNK_TOKEN_STR,
        )
    )

    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    bos_token_id = tokenizer.token_to_id(BOS_TOKEN_STR)
    eos_token_id = tokenizer.token_to_id(EOS_TOKEN_STR)
    cls_token_id = tokenizer.token_to_id(CLS_TOKEN_STR)

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{BOS_TOKEN_STR} $A {CLS_TOKEN_STR}",
        special_tokens=[
            (cls_token_id, CLS_TOKEN_STR),
            (bos_token_id, BOS_TOKEN_STR),
            (eos_token_id, EOS_TOKEN_STR),
        ],
    )

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=UNK_TOKEN_STR,
        pad_token=PAD_TOKEN_STR,
        cls_token=CLS_TOKEN_STR,
        bos_token=BOS_TOKEN_STR,
        eos_token=EOS_TOKEN_STR,
    )

    return wrapped_tokenizer
