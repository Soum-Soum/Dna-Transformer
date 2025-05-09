from typing import Optional
from loguru import logger
import torch
from transformers import ModernBertConfig
from transformers.models.modernbert.modeling_modernbert import (
    ModernBertEmbeddings,
    ModernBertForSequenceClassification,
    ModernBertPredictionHead,
    BaseModelOutput,
    SequenceClassifierOutput,
)
from torch import nn

from adn.models.activation_shaping import ActivationShapingS


class DnaModernBertConfig(ModernBertConfig):

    def __init__(
        self,
        max_position: int = None,
        activation_shaping: bool = False,
        activation_shaping_pruning_level: float = 0.5,
        class_weights: Optional[list[float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_position = max_position
        self.activation_shaping = activation_shaping
        self.activation_shaping_pruning_level = activation_shaping_pruning_level
        self.class_weights = class_weights


class DnaModernBertEmbeddings(ModernBertEmbeddings):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: DnaModernBertConfig):
        super().__init__(config)
        self.chromosome_position_embeddings = nn.Linear(1, config.hidden_size)
        self.max_position = config.max_position

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        input_ids, chromosome_positions = input_ids.chunk(2, dim=1)

        chromosome_positions = chromosome_positions.float() / self.max_position
        chromosome_positions = self.chromosome_position_embeddings(
            chromosome_positions.unsqueeze(-1)
        )

        input_embeddings = self.tok_embeddings(input_ids) + chromosome_positions

        return super().forward(
            input_ids=input_ids,
            inputs_embeds=input_embeddings,
        )


class ActivationShapingModernBertPredictionHead(ModernBertPredictionHead):

    def __init__(self, config: DnaModernBertConfig):
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
        if self.activation_shaping is not None:
            hidden_states = hidden_states.unsqueeze(1)
            hidden_states = self.activation_shaping(hidden_states)
            hidden_states = hidden_states.squeeze(1)

        return super().forward(hidden_states)


class DnaModernBertForSequenceClassification(ModernBertForSequenceClassification):

    def __init__(self, config: DnaModernBertConfig):
        super().__init__(config)
        self.model.embeddings = DnaModernBertEmbeddings(config)
        self.class_weights = (
            torch.tensor(config.class_weights, dtype=torch.float32)
            if config.class_weights is not None
            else torch.tensor([1.0] * config.num_labels, dtype=torch.float32)
        )
        self.head = ActivationShapingModernBertPredictionHead(config)

    def _embeddings(self, **kwargs) -> BaseModelOutput:
        return self.model(
            input_ids=kwargs.get("input_ids"),
            attention_mask=kwargs.get("attention_mask"),
            position_ids=kwargs.get("position_ids"),
            output_attentions=kwargs.get("output_attentions"),
            output_hidden_states=kwargs.get("output_hidden_states"),
            return_dict=kwargs.get("return_dict"),
            seq_len=kwargs.get("input_ids").shape[1] // 2,
        )

    def _classify(
        self, outputs: BaseModelOutput, labels=None
    ) -> SequenceClassifierOutput:
        last_hidden_state = outputs[0]

        last_hidden_state = last_hidden_state[:, 0]

        pooled_output = self.head(last_hidden_state)
        pooled_output = self.drop(pooled_output)
        logits = self.classifier(pooled_output)

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
    ) -> tuple[BaseModelOutput, SequenceClassifierOutput]:
        embeddings_outputs = self._embeddings(**kwargs)
        classifier_outputs = self._classify(embeddings_outputs, labels)
        return embeddings_outputs, classifier_outputs

    def forward(self, labels: torch.Tensor, **kwargs) -> SequenceClassifierOutput:
        assert kwargs.get("inputs_embeds") is None, "inputs_embeds not supported"
        return self.predict(labels=labels, **kwargs)[1]
