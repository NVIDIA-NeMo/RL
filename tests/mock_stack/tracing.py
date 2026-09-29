# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local OTel traces using the same Lens span helper as NeMo-RL telemetry."""

from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from opentelemetry import context, trace as otel
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_ON

from nemo_rl.telemetry.instrumentation import span_cm
from nemo_rl.weight_sync.interfaces import WeightSynchronizer


class TraceRun:
    """Record one run without replacing the global provider or Lens settings.

    HTTP calls join this trace, but no cross-process tracing is claimed for
    Ray or Gym workers. The provider always records, even under an unsampled
    parent or a disabled sampler environment setting.
    """

    def __init__(self, output: Path):
        self.path = output / "traces" / f"{uuid4().hex}.jsonl"
        self._stack = ExitStack()

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self._stack.enter_context(self.path.open("w"))
        provider = TracerProvider(sampler=ALWAYS_ON)
        provider.add_span_processor(
            BatchSpanProcessor(
                ConsoleSpanExporter(
                    out=stream, formatter=lambda span: span.to_json(indent=None) + "\n"
                )
            )
        )
        self._stack.callback(provider.shutdown)
        self.tracer = provider.get_tracer("tests.mock_stack")
        token = context.attach(context.Context())
        self._stack.callback(context.detach, token)
        self.root = self._stack.enter_context(
            span_cm("test.cpu_run", tracer=self.tracer)
        )
        return self

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)

    @contextmanager
    def span(self, name: str, **attributes):
        # HTTP runs in another thread. Preserve nested spans in controller
        # tasks, but attach the run when called outside this trace.
        current = otel.get_current_span()
        parent = (
            current
            if current.get_span_context().trace_id
            == self.root.get_span_context().trace_id
            else self.root
        )
        with otel.use_span(
            parent, record_exception=False, set_status_on_exception=False
        ):
            with span_cm(name, tracer=self.tracer, **attributes) as span:
                yield span

    def controller(self, controller):
        # The parent checkpoint stack predates Single Controller tracing.
        # Observe its boundaries without patching replaceable components.
        def wrap(original, name):
            @wraps(original)
            async def observed(*args, **kwargs):
                with self.span(name, **{"test.train_step": controller._train_steps}):
                    return await original(*args, **kwargs)

            return observed

        for method, name in (
            ("_save_checkpoint", "test.checkpoint.save"),
            ("_prepare_and_commit_gym_checkpoint", "test.checkpoint.prepare_commit"),
            ("_release_committed_gym_checkpoint", "test.checkpoint.release"),
        ):
            self._stack.enter_context(
                patch.object(
                    controller, method, wrap(getattr(controller, method), name)
                )
            )


class TracedRefit(WeightSynchronizer):
    """Forward the refit contract and measure the actual weight transfer."""

    def __init__(self, refit: WeightSynchronizer, trace: TraceRun):
        self.refit = refit
        self.trace = trace

    @property
    def is_stale(self):
        return self.refit.is_stale

    def sync_weights(self, *, timer=None, kv_scales=None):
        with self.trace.span("test.refit"):
            return self.refit.sync_weights(timer=timer, kv_scales=kv_scales)

    def init_communicator(self):
        return self.refit.init_communicator()

    def reconcile_communicator(self, absent_shards, force=False):
        return self.refit.reconcile_communicator(absent_shards, force=force)

    def shutdown(self):
        return self.refit.shutdown()
