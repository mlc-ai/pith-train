"""CPU checks that capture_routes never leaves a router hook behind."""

import pytest
from torch import nn

from tests.acceptance import omni_route_replay as routing


class RefusingRouter(nn.Linear):
    def register_forward_hook(self, *args, **kwargs):
        raise RuntimeError("hook registration refused")


def decoder(*routers):
    """Decoder layers whose mlp.gate modules match the acceptance router pattern."""
    model = nn.Module()
    model.layers = nn.ModuleList()
    for router in routers:
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.gate = router
        model.layers.append(layer)
    return model


def hook_counts(routers):
    return [len(router._forward_hooks) for router in routers]


def test_capture_routes_releases_earlier_hooks_when_a_registration_fails():
    first = nn.Linear(4, 2)
    with pytest.raises(RuntimeError, match="refused"):
        with routing.capture_routes(decoder(first, RefusingRouter(4, 2)), native=False):
            pass
    assert hook_counts([first]) == [0]


def test_capture_routes_releases_every_hook_on_exit_and_on_error():
    routers = [nn.Linear(4, 2) for _ in range(3)]
    model = decoder(*routers)
    with routing.capture_routes(model, native=True):
        assert hook_counts(routers) == [1, 1, 1]
    assert hook_counts(routers) == [0, 0, 0]
    with pytest.raises(ValueError, match="body failed"):
        with routing.capture_routes(model, native=False):
            raise ValueError("body failed")
    assert hook_counts(routers) == [0, 0, 0]
