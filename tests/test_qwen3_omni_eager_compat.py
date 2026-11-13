"""New explicit-compat tests; the existing Stage5 acceptance tests are unchanged."""

import hashlib
import inspect
import unittest
from types import SimpleNamespace

import torch

from pithtrain.models.qwen3_omni_eager_compat import eager_experts_forward


def make_case(seed=82, k=2, dtype=torch.float32, n=11, experts=16, hidden=32, width=24):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    gu = torch.empty(experts, 2 * width, hidden, dtype=dtype).normal_(0, 0.2, generator=generator)
    down = torch.empty(experts, hidden, width, dtype=dtype).normal_(0, 0.2, generator=generator)
    x = torch.randn(n, hidden, dtype=dtype, generator=generator)
    weights, ids = torch.randn(n, experts, generator=generator).softmax(-1).topk(k, dim=-1)
    weights = (weights / weights.sum(-1, keepdim=True)).to(dtype)
    cotangent = torch.randn(n, hidden, dtype=dtype, generator=generator)
    return [x, weights, ids, gu, down], cotangent


def floating_leaves(args):
    return tuple(args[index] for index in (0, 1, 3, 4))


def clone_args(args):
    return [value.detach().clone().requires_grad_(index != 2) for index, value in enumerate(args)]


class TestEagerCompatBoundary(unittest.TestCase):
    def test_empty_tokens_have_zero_vjps(self):
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                args, _ = make_case(dtype=dtype, n=0)
                args = clone_args(args)
                actual = eager_experts_forward(*args)
                self.assertEqual(actual.shape, (0, 32))
                self.assertEqual(actual.dtype, dtype)
                grads = torch.autograd.grad(actual, floating_leaves(args), torch.empty_like(actual))
                for leaf, grad in zip(floating_leaves(args), grads, strict=True):
                    self.assertEqual(grad.shape, leaf.shape)
                    self.assertTrue(torch.equal(grad, torch.zeros_like(leaf)))

    def test_unused_experts_have_zero_gradients(self):
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                args, cot = make_case(dtype=dtype, n=3, experts=8, hidden=8, width=4)
                args[2] = torch.tensor([[2, 5], [5, 2], [2, 5]])
                args = clone_args(args)
                out = eager_experts_forward(*args)
                grads = torch.autograd.grad(out, floating_leaves(args), cot)
                unused = torch.tensor([0, 1, 3, 4, 6, 7])
                for grad in grads[-2:]:
                    self.assertTrue(torch.equal(grad[unused], torch.zeros_like(grad[unused])))

    def test_does_not_mutate_inputs_or_global_rng(self):
        args, _ = make_case()
        originals = [value.clone() for value in args]
        versions = [value._version for value in args]
        state = torch.random.get_rng_state().clone()
        with torch.no_grad():
            eager_experts_forward(*args)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        for actual, expected, version in zip(args, originals, versions, strict=True):
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(actual._version, version)

    def test_rejects_invalid_dimensions(self):
        args, _ = make_case()
        replacements = [
            (0, args[0].unsqueeze(0)),
            (1, args[1].flatten()),
            (2, args[2].flatten()),
            (3, args[3].flatten(0, 1)),
            (4, args[4].flatten(0, 1)),
            (0, args[0][:, :-1]),
            (1, args[1][:-1]),
            (2, args[2][:, :1]),
            (3, args[3][:, :-1]),
            (4, args[4][:-1]),
            (4, args[4][:, :, :-1]),
            (3, args[3][:0]),
            (0, args[0][:, :0]),
            (3, args[3][:, :0]),
        ]
        for index, value in replacements:
            with self.subTest(index=index, shape=tuple(value.shape)):
                altered = list(args)
                altered[index] = value
                with self.assertRaises(ValueError):
                    eager_experts_forward(*altered)

    def test_rejects_invalid_k(self):
        args, _ = make_case()
        for k in (0, 17):
            with self.subTest(k=k):
                altered = list(args)
                altered[1] = torch.ones(11, k)
                altered[2] = torch.zeros(11, k, dtype=torch.int64)
                with self.assertRaisesRegex(ValueError, "K must satisfy"):
                    eager_experts_forward(*altered)

    def test_rejects_out_of_range_or_duplicate_ids(self):
        args, _ = make_case()
        for label, ids in (
            ("negative", args[2].clone()),
            ("outside", args[2].clone()),
            ("duplicate", args[2].clone()),
        ):
            with self.subTest(label=label):
                ids[0, 0] = -1 if label == "negative" else 16 if label == "outside" else ids[0, 1]
                altered = list(args)
                altered[2] = ids
                with self.assertRaises(ValueError):
                    eager_experts_forward(*altered)

    def test_rejects_invalid_dtypes(self):
        args, _ = make_case()
        for index, dtype in (
            (0, torch.float64),
            (0, torch.float16),
            (0, torch.int64),
            (1, torch.bfloat16),
            (3, torch.bfloat16),
            (4, torch.bfloat16),
            (2, torch.int32),
            (2, torch.float32),
        ):
            with self.subTest(index=index, dtype=dtype):
                altered = list(args)
                altered[index] = altered[index].to(dtype)
                with self.assertRaises(TypeError):
                    eager_experts_forward(*altered)

    def test_rejects_nontensor_sparse_or_non_cpu(self):
        args, _ = make_case()
        for value, exception in (
            (None, TypeError),
            (args[0].to_sparse(), ValueError),
            (torch.empty_like(args[0], device="meta"), ValueError),
        ):
            with self.subTest(kind=str(type(value)), device=getattr(value, "device", None)):
                altered = list(args)
                altered[0] = value
                with self.assertRaises(exception):
                    eager_experts_forward(*altered)


class TestEagerCompatHF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import transformers
        from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_model

        if transformers.__version__ != "5.17.0":
            raise RuntimeError("These new compatibility tests require the pinned HF 5.17.0")
        with open(inspect.getfile(hf_model), "rb") as source:
            actual_sha = hashlib.sha256(source.read()).hexdigest()
        expected_sha = "1c1fc322dfa547c697576e8ad3c88b817aa382abc9db5d966755be3d9fabf7ce"
        if actual_sha != expected_sha:
            raise RuntimeError("Installed HF expert source differs from the pinned reference")
        cls.hf_type = hf_model.Qwen3OmniMoeThinkerTextExperts

    def assert_hf_equal(self, args, cot):
        actual_args = clone_args(args)
        expected_args = clone_args(args)
        x, weights, ids, gu, down = expected_args
        config = SimpleNamespace(
            num_experts=gu.shape[0],
            hidden_size=x.shape[1],
            moe_intermediate_size=down.shape[-1],
            hidden_act="silu",
            _experts_implementation="eager",
        )
        model = self.hf_type(config).to(x.dtype)
        model.gate_up_proj = torch.nn.Parameter(gu)
        model.down_proj = torch.nn.Parameter(down)
        expected_args[3:] = [model.gate_up_proj, model.down_proj]
        expected = model(x, ids, weights)
        actual = eager_experts_forward(*actual_args)
        self.assertTrue(torch.equal(actual, expected), "forward differs from actual HF eager")
        expected_grads = torch.autograd.grad(expected, floating_leaves(expected_args), cot)
        actual_grads = torch.autograd.grad(actual, floating_leaves(actual_args), cot)
        for name, a, b in zip(
            ("hidden", "routing_weights", "gate_up", "down"),
            actual_grads,
            expected_grads,
            strict=True,
        ):
            self.assertTrue(torch.equal(a, b), f"{name} VJP differs from actual HF eager")

    def test_seeded_k1_k2_k8_forward_and_four_vjps(self):
        for seed in (82, 83):
            for k in (1, 2, 8):
                for dtype in (torch.bfloat16, torch.float32):
                    with self.subTest(seed=seed, k=k, dtype=dtype):
                        self.assert_hf_equal(*make_case(seed=seed, k=k, dtype=dtype))

    def test_unused_experts_zero_route_and_noncontiguous_hidden(self):
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                args, cot = make_case(seed=84, dtype=dtype, n=3, experts=8, hidden=8, width=4)
                args[0] = args[0].transpose(0, 1).contiguous().transpose(0, 1)
                self.assertFalse(args[0].is_contiguous())
                args[2] = torch.tensor([[2, 5], [5, 2], [2, 5]])
                args[1][0, 0] = 0
                self.assert_hf_equal(args, cot)


if __name__ == "__main__":
    unittest.main()
