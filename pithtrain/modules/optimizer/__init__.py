"""Optimizers. Muon is composed with a separate auxiliary optimizer (a plain
``torch.optim`` one); future optimizers live alongside ``muon`` here."""

from pithtrain.modules.optimizer.muon import (
    Muon,
    is_muon_param,
    muon_scale_factor,
    partition_muon_params,
    zeropower_via_newtonschulz5,
)

__all__ = [
    "Muon",
    "is_muon_param",
    "muon_scale_factor",
    "partition_muon_params",
    "zeropower_via_newtonschulz5",
]
