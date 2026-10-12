# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes import LegalizePortableDimOrderPass, LegalizeViewCopyPass
from executorch.extension.pybindings.portable_lib import (  # @manual
    _load_for_executorch_from_buffer,
)
from torch.export import export
from torch.testing import FileCheck


class MeanIssueReproModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.mean(x, dim=[2, 3], keepdim=True)
        return x.reshape([-1, 2])


class MeanNoReshapeModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=[2, 3], keepdim=True)


class MeanKeepdimFalseModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=[2, 3], keepdim=False)


class MeanSingleDimModule(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=[self.dim], keepdim=True)


class LinearIssueReproModule(torch.nn.Module):
    def __init__(self, out_features: int = 10):
        super().__init__()
        self.linear = torch.nn.Linear(10, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class FlattenModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2)


class TestLegalizePortableDimOrderPass(unittest.TestCase):
    @staticmethod
    def _build_edge(module: torch.nn.Module, sample_input, compile_config=None):
        return to_edge_transform_and_lower(
            export(module, sample_input, strict=True),
            partitioner=[],
            compile_config=compile_config,
        )

    @staticmethod
    def _build_executorch_program(
        module: torch.nn.Module, sample_input, compile_config=None
    ):
        edge = TestLegalizePortableDimOrderPass._build_edge(
            module, sample_input, compile_config=compile_config
        )
        executorch_prog = edge.to_executorch(
            config=ExecutorchBackendConfig(extract_delegate_segments=False)
        )
        return edge, executorch_prog

    @staticmethod
    def _find_call_function_nodes(graph_module, target_substring: str):
        return [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function" and target_substring in str(node.target)
        ]

    def _assert_runtime_matches_eager(
        self,
        module: torch.nn.Module,
        sample_input,
        *,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        compile_config=None,
    ):
        edge, executorch_prog = self._build_executorch_program(
            module, sample_input, compile_config=compile_config
        )
        executorch_module = _load_for_executorch_from_buffer(executorch_prog.buffer)

        with torch.no_grad():
            expected = module(*sample_input)
        runtime_output = executorch_module.run_method("forward", sample_input)[0]

        torch.testing.assert_close(
            runtime_output,
            expected,
            atol=atol,
            rtol=rtol,
        )
        return edge.exported_program().graph_module

    def test_channels_last_mean_inserts_copy_before_mean(self) -> None:
        module = MeanIssueReproModule().eval().to(memory_format=torch.channels_last)
        sample_input = (torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last),)

        edge = self._build_edge(module, sample_input)
        updated = edge.transform([LegalizePortableDimOrderPass()])
        graph_module = updated.exported_program().graph_module

        copy_nodes = self._find_call_function_nodes(graph_module, "_to_dim_order_copy")
        self.assertEqual(len(copy_nodes), 1)
        mean_nodes = self._find_call_function_nodes(graph_module, "aten.mean.dim")
        self.assertEqual(len(mean_nodes), 1)

        copy_arg = mean_nodes[0].args[0]
        self.assertIsInstance(copy_arg, torch.fx.Node)
        self.assertIn("_to_dim_order_copy", str(copy_arg.target))

    def test_channels_last_linear_inserts_copy_before_expand_copy(self) -> None:
        module = LinearIssueReproModule(out_features=8).eval()
        sample_input = (torch.randn(1, 2, 3, 10).to(memory_format=torch.channels_last),)

        edge = self._build_edge(module, sample_input)
        updated = edge.transform([LegalizePortableDimOrderPass()])
        graph_module = updated.exported_program().graph_module

        copy_nodes = self._find_call_function_nodes(graph_module, "_to_dim_order_copy")
        self.assertEqual(len(copy_nodes), 1)
        expand_nodes = self._find_call_function_nodes(graph_module, "aten.expand_copy")
        self.assertTrue(expand_nodes)
        self.assertTrue(
            any(
                isinstance(node.args[0], torch.fx.Node)
                and "_to_dim_order_copy" in str(node.args[0].target)
                for node in expand_nodes
            ),
            "Expected one expand_copy path to consume a legalized contiguous copy.",
        )

    def test_skip_dim_order_does_not_insert_layout_copies(self) -> None:
        module = FlattenModule().eval()
        sample_input = (torch.randn(1, 3, 2, 4).to(memory_format=torch.channels_last),)
        _, program = self._build_executorch_program(
            module,
            sample_input,
            compile_config=EdgeCompileConfig(_skip_dim_order=True),
        )
        FileCheck().check_not("dim_order_ops").check_not("aten._to_copy").run(
            program.exported_program().graph_module.code
        )

    def test_internal_channels_last_flatten_preserves_values(self) -> None:
        class InternalChannelsLastFlatten(torch.nn.Module):
            def forward(self, x):
                x = x.to(memory_format=torch.channels_last)
                return x.flatten(2).transpose(1, 2)

        module = InternalChannelsLastFlatten().eval()
        x = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4)
        self._assert_runtime_matches_eager(module, (x,), rtol=0, atol=0)

    def test_channels_last_flatten_preserves_values(self) -> None:
        module = FlattenModule().eval()
        for channels in (1, 3):
            x = torch.arange(16 * channels, dtype=torch.float32).reshape(
                2, channels, 2, 4
            )
            sample_input = (x.to(memory_format=torch.channels_last),)
            expected = module(*sample_input)
            for remove_view_copy in (True, False):
                with self.subTest(
                    channels=channels,
                    remove_view_copy=remove_view_copy,
                ):
                    edge = self._build_edge(module, sample_input)
                    program = edge.to_executorch(
                        config=ExecutorchBackendConfig(
                            extract_delegate_segments=False,
                            remove_view_copy=remove_view_copy,
                        )
                    )
                    runtime = _load_for_executorch_from_buffer(program.buffer)
                    actual = runtime.run_method("forward", sample_input)[0]
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    if channels > 1:
                        FileCheck().check("dim_order_ops._to_dim_order_copy.out").run(
                            program.exported_program().graph_module.code
                        )

    def test_layout_copy_does_not_reuse_inplace_storage(self) -> None:
        class FlattenCloneModule(torch.nn.Module):
            def forward(self, x):
                return (x + x).clone().flatten(2).transpose(1, 2)

        module = FlattenCloneModule().eval()
        x = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4)
        sample_input = (x.to(memory_format=torch.channels_last),)
        expected = module(*sample_input)
        edge = self._build_edge(module, sample_input)
        program = edge.to_executorch(
            config=ExecutorchBackendConfig(
                extract_delegate_segments=False,
                reinplace_extra_ops=frozenset(
                    {exir_ops.edge.dim_order_ops._clone_dim_order.default}
                ),
            )
        )
        runtime = _load_for_executorch_from_buffer(program.buffer)
        actual = runtime.run_method("forward", sample_input)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_contiguous_inputs_skip_copy(self) -> None:
        mean_module = (
            MeanIssueReproModule().eval().to(memory_format=torch.channels_last)
        )
        linear_module = LinearIssueReproModule().eval()
        cases = (
            ("mean", mean_module, (torch.randn(1, 2, 3, 4).contiguous(),)),
            ("linear", linear_module, (torch.randn(1, 2, 3, 10).contiguous(),)),
            ("flatten", FlattenModule().eval(), (torch.randn(2, 3, 2, 4),)),
        )

        for name, module, sample_input in cases:
            with self.subTest(name=name):
                edge = self._build_edge(module, sample_input)
                updated = edge.transform(
                    [LegalizePortableDimOrderPass(), LegalizeViewCopyPass()]
                )
                FileCheck().check_not("_to_dim_order_copy").check_not(
                    "aten._to_copy"
                ).run(updated.exported_program().graph_module.code)

    def test_issue_16507_runtime_regressions(self) -> None:
        cases = (
            (
                "baseline",
                MeanIssueReproModule().eval().to(memory_format=torch.channels_last),
                (torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last),),
            ),
            (
                "no_reshape",
                MeanNoReshapeModule().eval().to(memory_format=torch.channels_last),
                (torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last),),
            ),
            (
                "keepdim_false",
                MeanKeepdimFalseModule().eval().to(memory_format=torch.channels_last),
                (torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last),),
            ),
            (
                "single_dim_h",
                MeanSingleDimModule(2).eval().to(memory_format=torch.channels_last),
                (torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last),),
            ),
            (
                "single_dim_w",
                MeanSingleDimModule(3).eval().to(memory_format=torch.channels_last),
                (torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last),),
            ),
        )

        for name, module, sample_input in cases:
            with self.subTest(name=name):
                graph_module = self._assert_runtime_matches_eager(module, sample_input)
                FileCheck().check("_to_dim_order_copy").check("aten.mean.out").run(
                    graph_module.code
                )

    def test_issue_11086_runtime_regressions(self) -> None:
        cases = (
            (
                "baseline",
                LinearIssueReproModule().eval(),
                (torch.randn(1, 2, 3, 10).to(memory_format=torch.channels_last),),
            ),
            (
                "larger_spatial",
                LinearIssueReproModule(out_features=6).eval(),
                (torch.randn(2, 5, 7, 10).to(memory_format=torch.channels_last),),
            ),
        )

        for name, module, sample_input in cases:
            with self.subTest(name=name):
                graph_module = self._assert_runtime_matches_eager(module, sample_input)
                FileCheck().check("_to_dim_order_copy").check(
                    "aten.expand_copy.out"
                ).run(graph_module.code)

    def test_issue_11086_c1_control_uses_alternate_path(self) -> None:
        module = LinearIssueReproModule(out_features=8).eval()
        sample_input = (torch.randn(1, 1, 3, 10).to(memory_format=torch.channels_last),)

        edge = self._build_edge(module, sample_input)
        graph_module = edge.exported_program().graph_module
        FileCheck().check_not("aten.expand_copy").run(graph_module.code)

        updated = edge.transform([LegalizePortableDimOrderPass()])
        FileCheck().check_not("_to_dim_order_copy").run(
            updated.exported_program().graph_module.code
        )

        runtime_graph = self._assert_runtime_matches_eager(module, sample_input)
        FileCheck().check_not("aten.expand_copy.out").run(runtime_graph.code)
