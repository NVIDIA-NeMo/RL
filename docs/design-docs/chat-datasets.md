# Data Format

This guide outlines the required data format for Hugging Face chat datasets and demonstrates how to use chat templates with Hugging Face tokenizers to add special tokens or task-specific information.

## Hugging Face Chat Datasets

Hugging Face chat datasets are expected to have the following structure: Each example in the dataset should be a dictionary with a `messages` key. The `messages` should be a list of dictionaries, each with a `role` and `content` key. The `role` typically has one of the following values: `system`, `user`, and `assistant`. For example:

```json
{
    "messages": [
        {
            "role": "system",
            "content": "This is a helpful system message."
        },
        {
            "role": "user",
            "content": "This is a user's question"
        },
        {
            "role": "assistant",
            "content": "This is the assistant's response."
        }
    ]
}
```

## Chat Templates

Formatting the data in this way allows us to take advantage of the Hugging Face tokenizers' `apply_chat_template` functionality to combine the messages. Chat templates can be used to add special tokens or task-specific information to each example in the dataset. Refer to the [HuggingFace apply_chat_template documentation](https://huggingface.co/docs/transformers/main/en/chat_templating#applychattemplate) for details.

By default, `apply_chat_template` attempts to apply the `chat_template` associated with the tokenizer. However, in some cases, users might want to specify their own chat template. Also, note that many tokenizers do not have associated `chat_template`s, in which case an explicit chat template is required. Users can specify an explicit chat template string using Jinja format and can pass that string to `apply_chat_template`. 
The following is an example using a simple template which prepends a role header to each turn:

```{testcode}
from transformers import AutoTokenizer

example_template = "{% for message in messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{{ content }}{% endfor %}"

example_input = [
    {
        'role': 'user',
        'content': 'Hello!'
    },
    {
        'role': 'assistant',
        'content': 'Hi there!'
    }
]
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
output = tokenizer.apply_chat_template(example_input, chat_template=example_template, tokenize=False)

## this is the output string we expect
expected_output = '<|start_header_id|>user<|end_header_id|>\n\nHello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHi there!<|eot_id|>'
assert output == expected_output
```

<!-- This testoutput is intentionally empty-->
```{testoutput}
:hide:
```

For more details on creating chat templates, refer to the [Hugging Face documentation](https://huggingface.co/docs/transformers/v4.34.0/en/chat_templating#how-do-i-create-a-chat-template).