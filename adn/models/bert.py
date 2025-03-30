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


class DnaBertConfig(BertConfig):

    def __init__(
        self,
        activation_shaping: bool = False,
        activation_shaping_pruning_level: float = 0.5,
        class_weights: Optional[list[float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation_shaping = activation_shaping
        self.activation_shaping_pruning_level = activation_shaping_pruning_level
        self.class_weights = class_weights


class DnaBertEmbeddings(BertEmbeddings):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.chromosome_position_embeddings = nn.Linear(1, config.hidden_size)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        chromosome_positions: torch.FloatTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.position_ids.device
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # Add chromosome position embeddings
        # print(chromosome_positions.shape)
        chromosome_positions = chromosome_positions.unsqueeze(-1)
        chromosome_position_embeddings = self.chromosome_position_embeddings(
            chromosome_positions
        )

        embeddings = (
            inputs_embeds + token_type_embeddings + chromosome_position_embeddings
        )
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ActivationShapingS(torch.nn.Module):

    def __init__(self, pruning_level: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pruning_level = pruning_level

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = inputs.shape
        fattened = inputs.reshape((batch_size, -1))
        sum_1 = torch.sum(fattened, dim=1)

        percentile = (
            torch.quantile(fattened, self.pruning_level, dim=-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, seq_len, hidden_size)
        )
        x_with_zeros = torch.where(
            inputs > percentile, inputs, torch.zeros_like(inputs)
        )
        sum_2 = torch.sum(x_with_zeros.reshape((batch_size, -1)), dim=1)

        # Avoid division by zero
        exp_ratio = (
            torch.exp(sum_1 / (sum_2 + 1e-6))
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, seq_len, hidden_size)
        )

        return x_with_zeros * exp_ratio


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
        self.custom_embed = DnaBertEmbeddings(config)

        # Patch the BERT model to ActivationShaping (if set in the config)
        self.bert.pooler = ActivationShapingBertPooler(config)

        # Convert class_weights to tensor if provided
        self.class_weights = (
            torch.tensor(config.class_weights, dtype=torch.float32)
            if config.class_weights is not None
            else torch.tensor([1.0] * config.num_labels, dtype=torch.float32)
        )

    def _embeddings(
        self, chromosome_positions: torch.Tensor, **kwargs
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        kwargs["inputs_embeds"] = self.custom_embed(
            input_ids=kwargs.get("input_ids"),
            position_ids=kwargs.get("position_ids"),
            token_type_ids=kwargs.get("token_type_ids"),
            chromosome_positions=chromosome_positions,
        )
        kwargs["input_ids"] = None
        return self.bert(
            attention_mask=kwargs.get("attention_mask"),
            token_type_ids=kwargs.get("token_type_ids"),
            position_ids=kwargs.get("position_ids"),
            head_mask=kwargs.get("head_mask"),
            inputs_embeds=kwargs.get("inputs_embeds"),
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
        self, labels: torch.Tensor, chromosome_positions: torch.Tensor, **kwargs
    ) -> tuple[BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput]:
        embeddings_outputs = self._embeddings(
            chromosome_positions=chromosome_positions, **kwargs
        )
        classifier_outputs = self._classify(embeddings_outputs, labels)
        return embeddings_outputs, classifier_outputs

    def forward(
        self, labels: torch.Tensor, chromosome_positions: torch.Tensor, **kwargs
    ) -> SequenceClassifierOutput:
        assert kwargs.get("inputs_embeds") is None, "inputs_embeds not supported"
        return self.predict(
            labels=labels, chromosome_positions=chromosome_positions, **kwargs
        )[1]
