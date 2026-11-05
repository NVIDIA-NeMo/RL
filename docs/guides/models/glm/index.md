# GLM

This is the landing page for GLM model guidance in NeMo RL. It links to
version-specific subpages covering validated recipes, recommended training and
generation settings, and known issues for each supported version.

For the full list of supported GLM models, see
[Model Support](../../../about/model-support.md).

## Version Guides

- **[GLM-5](glm5.md)** — GLM-5.1 and GLM-5.2 GRPO recipes on the Megatron
  backend, including 131K-token training capacity, colocated and non-colocated
  vLLM, and cuDNN or TileLang DSA kernels.

Subpages for future GLM versions can be added here as distinct,
recipe-backed guidance becomes available. Until then, the GRPO guide and recipe
YAML files remain the source of truth for other supported GLM models.

```{toctree}
:hidden:

glm5.md
```
