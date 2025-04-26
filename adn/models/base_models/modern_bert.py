
from transformers import ModernBertConfig, ModernBertModel, ModernBertForSequenceClassification
from adn.models.base_models.base import make_dna_config_class


DnaModernBertConfig = make_dna_config_class(ModernBertConfig)