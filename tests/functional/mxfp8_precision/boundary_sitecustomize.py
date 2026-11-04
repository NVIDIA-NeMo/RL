"""Copied as sitecustomize.py only inside the private boundary audit."""

import importlib.abc
import importlib.machinery
import os
import sys


if os.environ.get("NANO_BOUNDARY_ROOT"):
    class BoundaryFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in {
                "nemo_rl.weight_sync.xferdtensor",
                "nemo_rl.models.generation.vllm.vllm_backend",
                "nemo_rl.models.generation.vllm.quantization.fp8",
            }:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return None
            original = spec.loader

            class Loader(importlib.abc.Loader):
                def create_module(self, module_spec):
                    return original.create_module(module_spec)

                def exec_module(self, module):
                    original.exec_module(module)
                    from boundary_refit_observer import install

                    install(fullname, module)

            spec.loader = Loader()
            return spec

    sys.meta_path.insert(0, BoundaryFinder())
