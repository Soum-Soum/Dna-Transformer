from typing import Optional
from loguru import logger
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertForSequenceClassification,
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
    BertConfig,
    BertPooler,
)
import torch
from torch import nn

from adn.models.activation_shaping import ActivationShapingS


class DnaBertConfig(BertConfig):

    def __init__(
        self,
        max_position: int = None,
        snp_count: int = None,
        activation_shaping: bool = False,
        activation_shaping_pruning_level: float = 0.5,
        class_weights: Optional[list[float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_position = max_position
        self.snp_count = snp_count
        self.activation_shaping = activation_shaping
        self.activation_shaping_pruning_level = activation_shaping_pruning_level
        self.class_weights = class_weights


class DnaBertEmbeddings(BertEmbeddings):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.snp_position_embeddings = nn.Linear(1, config.hidden_size)
        self.max_position = config.max_position

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:

        input_ids, snp_positions, snp_ids = input_ids.chunk(3, dim=1)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # Add chromosome position embeddings
        snp_positions = snp_positions.unsqueeze(-1)
        snp_positions = snp_positions / self.max_position
        snp_position_embeddings = self.snp_position_embeddings(snp_positions)

        embeddings = inputs_embeds + snp_position_embeddings  # + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class DnaBertEmbeddingsV2(BertEmbeddings):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.snp_embeddings = nn.Embedding(
            num_embeddings=config.snp_count,
            embedding_dim=config.hidden_size,
        )
        self.snp_position_embeddings = nn.Linear(1, config.hidden_size)
        self.max_position = config.max_position

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:

        input_ids, snp_positions, snp_ids = input_ids.chunk(3, dim=1)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # Add chromosome position embeddings
        snp_embeddings = self.snp_embeddings(snp_ids)
        snp_position = snp_position.float().unsqueeze(-1) / self.max_position
        snp_position_embeddings = self.snp_position_embeddings(snp_positions)

        embeddings = inputs_embeds + snp_embeddings  # + token_type_embeddings

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ActivationShapingBertPooler(BertPooler):

    def __init__(self, config: DnaBertConfig):
        super().__init__(config)
        if config.activation_shaping:
            logger.info(
                f"Using activation shaping with pruning level {config.activation_shaping_pruning_level}"
            )
            self.activation_shaping = ActivationShapingS(
                pruning_level=config.activation_shaping_pruning_level
            )
        else:
            self.activation_shaping = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states[:, :1]
        if self.activation_shaping is not None:
            hidden_states = self.activation_shaping(hidden_states)

        return super().forward(hidden_states)


class DnaBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config: DnaBertConfig):
        super().__init__(config)
        # self.bert.embeddings = DnaBertEmbeddings(config)
        self.bert.embeddings = DnaBertEmbeddingsV2(config)
        self.bert.pooler = ActivationShapingBertPooler(config)
        self.class_weights = (
            torch.tensor(config.class_weights, dtype=torch.float32)
            if config.class_weights is not None
            else torch.tensor([1.0] * config.num_labels, dtype=torch.float32)
        )

    def _embeddings(self, **kwargs) -> BaseModelOutputWithPoolingAndCrossAttentions:
        return self.bert(
            input_ids=kwargs.get("input_ids"),
            attention_mask=kwargs.get("attention_mask"),
            token_type_ids=kwargs.get("token_type_ids"),
            position_ids=kwargs.get("position_ids"),
            head_mask=kwargs.get("head_mask"),
            output_attentions=kwargs.get("output_attentions"),
            output_hidden_states=kwargs.get("output_hidden_states"),
            return_dict=kwargs.get("return_dict"),
        )

    def _classify(
        self, outputs: BaseModelOutputWithPoolingAndCrossAttentions, labels=None
    ) -> SequenceClassifierOutput:
        pooled_output = outputs.pooler_output
        logits = self.classifier(self.dropout(pooled_output))
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(labels.device))
        loss = (
            loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            if labels is not None
            else None
        )

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def predict(
        self, labels: torch.Tensor, **kwargs
    ) -> tuple[BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput]:
        embeddings_outputs = self._embeddings(**kwargs)
        classifier_outputs = self._classify(embeddings_outputs, labels)
        return embeddings_outputs, classifier_outputs

    def forward(self, labels: torch.Tensor, **kwargs) -> SequenceClassifierOutput:
        assert kwargs.get("inputs_embeds") is None, "inputs_embeds not supported"
        return self.predict(labels=labels, **kwargs)[1]
