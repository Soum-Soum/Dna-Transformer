from typing import Optional
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertForSequenceClassification,
)
import torch
from torch import nn


class CustomBertEmbeddings(BertEmbeddings):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
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


class CustomBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.custom_embed = CustomBertEmbeddings(config)

        # Convert class_weights to tensor if provided
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def embed(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        chromosome_positions=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        inputs_embeds = self.custom_embed(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            chromosome_positions=chromosome_positions,
        )
        return self.bert(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        chromosome_positions=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        assert inputs_embeds is None, "inputs_embeds not supported"
        inputs_embeds = self.custom_embed(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            chromosome_positions=chromosome_positions,
        )

        outputs = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if labels is not None and self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(labels.device))
            logits = outputs.logits
            loss = loss_fct(logits, labels)
            outputs.loss = loss  # Replace original loss

        return outputs
