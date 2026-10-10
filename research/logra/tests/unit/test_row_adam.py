import torch
from logra.row_adam import row_adam_direction


def test_row_adam_running_rms_reference():
    history = [
        torch.tensor([[1.0, 3.0], [2.0, 4.0]]),
        torch.tensor([[2.0, 1.0], [8.0, 2.0]]),
    ]
    moment = torch.zeros(2, 1)
    for index, sketch in enumerate(history, 1):
        expected_moment = sum(
            (1 - 0.95) * 0.95 ** (index - 1 - j) * g.square().mean(1, keepdim=True)
            for j, g in enumerate(history[:index])
        )
        expected = sketch / ((expected_moment / (1 - 0.95**index)).sqrt() + 1e-8)
        actual = row_adam_direction(
            sketch, moment, step=index, beta2=0.95, epsilon=1e-8
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(moment, expected_moment)


def test_first_step_scale_invariance_without_norm_restoration():
    sketch = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    a = row_adam_direction(sketch, torch.zeros(2, 1), step=1, beta2=0.95, epsilon=1e-12)
    b = row_adam_direction(
        sketch * 1e-3, torch.zeros(2, 1), step=1, beta2=0.95, epsilon=1e-12
    )
    torch.testing.assert_close(a, b)


def test_inplace_direction_matches_out_of_place_and_reuses_storage():
    sketch = torch.tensor([[1.0, 3.0], [2.0, 4.0], [0.5, -1.0]])
    expected = row_adam_direction(
        sketch.clone(), torch.zeros(3, 1), step=3, beta2=0.9, epsilon=1e-8
    )
    moment = torch.zeros(3, 1)
    actual = row_adam_direction(
        sketch, moment, step=3, beta2=0.9, epsilon=1e-8, inplace=True
    )
    assert actual is sketch
    torch.testing.assert_close(actual, expected)
