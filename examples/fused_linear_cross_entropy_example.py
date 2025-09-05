#!/usr/bin/env python3
"""Example script demonstrating how to use FusedLinearCrossEntropy in nemo-rl.

This example shows:
1. How to configure and create a FusedLinearCrossEntropyLoss
2. How to use it with the calculate_loss utility function
3. How to set up the required data structures
"""

import torch

from nemo_rl.algorithms.loss_functions import (
    HAVE_FUSED_LINEAR_CE,
    FusedLinearCrossEntropyLoss,
    FusedLinearCrossEntropyLossConfig,
    calculate_loss,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def create_mock_model(vocab_size: int = 1000, hidden_size: int = 768):
    """Create a mock model with lm_head for testing."""

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

        def forward(self, input_ids, **kwargs):
            # Mock forward pass - in reality this would be your actual model
            batch_size, seq_len = input_ids.shape
            hidden_states = torch.randn(batch_size, seq_len, hidden_size)
            return type(
                "MockOutput",
                (),
                {
                    "hidden_states": [hidden_states],  # List of layer outputs
                    "last_hidden_state": hidden_states,
                },
            )()

    return MockModel()


def main():
    """Main example function."""
    print("FusedLinearCrossEntropy Integration Example")
    print("=" * 50)

    if not HAVE_FUSED_LINEAR_CE:
        print("WARNING: FusedLinearCrossEntropy is not available.")
        print("Please install the required dependencies from nemo-automodel.")
        return

    # Configuration
    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    hidden_size = 768

    print("Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Hidden size: {hidden_size}")
    print()

    # Create loss function
    loss_config: FusedLinearCrossEntropyLossConfig = {
        "ignore_index": -100,
        "logit_softcapping": 0.0,
        "reduction": "sum",
    }

    loss_fn = FusedLinearCrossEntropyLoss(loss_config)
    print(f"Created FusedLinearCrossEntropyLoss with config: {loss_config}")
    print()

    # Create mock model
    model = create_mock_model(vocab_size, hidden_size)
    print("Created mock model with lm_head")
    print()

    # Create sample data
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    token_mask = torch.ones(batch_size, seq_len)
    token_mask[:, -2:] = 0  # Mask last 2 tokens
    sample_mask = torch.ones(batch_size)

    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
        }
    )

    print("Created sample data:")
    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  token_mask shape: {token_mask.shape}")
    print(f"  sample_mask shape: {sample_mask.shape}")
    print()

    # Run model forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    print("Model forward pass completed")
    print(f"  hidden_states shape: {outputs.hidden_states[-1].shape}")
    print()

    # Calculate loss using the utility function
    try:
        global_valid_seqs = torch.tensor(batch_size)
        global_valid_toks = torch.tensor(token_mask.sum().item())

        loss = calculate_loss(
            loss_fn=loss_fn,
            logits=None,  # Not used for FusedLinearCrossEntropy
            model=model,
            hidden_states=outputs.hidden_states[-1],
            data=data,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
        )

        print("Loss calculation successful!")
        print(f"  Loss value: {loss.item():.4f}")
        print(f"  Loss type: {type(loss)}")
        print()

        # Test direct loss function call
        enhanced_data = BatchedDataDict(data)
        enhanced_data["hidden_states"] = outputs.hidden_states[-1]

        loss_direct, metrics = loss_fn(
            next_token_logits=torch.empty(0),  # Not used
            data=enhanced_data,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            model=model,
        )

        print("Direct loss function call successful!")
        print(f"  Loss value: {loss_direct.item():.4f}")
        print(f"  Metrics: {metrics}")
        print()

        # Verify both methods give same result
        assert torch.allclose(loss, loss_direct, rtol=1e-5), "Loss values don't match!"
        print("✓ Both calculation methods give identical results")

    except Exception as e:
        print(f"Error during loss calculation: {e}")
        raise

    print()
    print("=" * 50)
    print("Example completed successfully!")
    print()
    print("Next steps:")
    print("1. Use FusedLinearCrossEntropyLoss in your training configuration")
    print("2. Ensure your model config has output_hidden_states=True")
    print("3. Use with DTensorPolicyWorker or MegatronPolicyWorker")


if __name__ == "__main__":
    main()
