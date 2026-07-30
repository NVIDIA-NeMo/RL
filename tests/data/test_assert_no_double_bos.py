import io
from unittest.mock import patch

from nemo_rl.data.datasets.utils import assert_no_double_bos


def test_assert_no_double_bos_message_uses_correct_function_name():
    tokenizer = type(
        "Tokenizer",
        (),
        {
            "tokenizer": type(
                "InnerTokenizer", (), {"bos_token_id": None, "name_or_path": "dummy"}
            )()
        },
    )()
    token_ids = type("Tensor", (), {"tolist": lambda self: [1, 2]})()

    with patch("sys.stdout", new_callable=io.StringIO) as captured:
        assert_no_double_bos(token_ids, tokenizer)

