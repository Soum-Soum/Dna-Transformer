from typing import Optional
from torch import Type


def make_dna_config_class(base_config_class):

    class DnaConfig(base_config_class):

        def __init__(
            self,
            max_position: int = -1,
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

    DnaConfig.__name__ = f"Dna{base_config_class.__name__}"
    DnaConfig.__qualname__ = f"Dna{base_config_class.__name__}"
    DnaConfig.__doc__ = (
        f"Configuration {base_config_class.__name__} spécifique à l'ADN."
    )

    return DnaConfig
