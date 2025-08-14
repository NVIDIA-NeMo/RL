try:
    from megatron.bridge import AutoBridge  # noqa: F401

    INSTALLED = True
except Exception:
    INSTALLED = False

print(f"Megatron Bridge {INSTALLED=}")
