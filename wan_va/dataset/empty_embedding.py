import os
from pathlib import Path

import torch


@torch.inference_mode()
def encode_empty_prompt(
    tokenizer,
    text_encoder,
    device,
    dtype=torch.bfloat16,
    max_sequence_length=512,
):
    text_inputs = tokenizer(
        [""],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    hidden_states = text_encoder(
        input_ids,
        attention_mask,
    ).last_hidden_state.to(dtype=dtype)
    sequence_length = int(attention_mask[0].sum().item())

    embedding = torch.zeros(
        (max_sequence_length, hidden_states.shape[-1]),
        dtype=dtype,
        device=device,
    )
    embedding[:sequence_length] = hidden_states[0, :sequence_length]
    return embedding.cpu()


def generate_empty_embedding(model_path, output_path, device):
    from wan_va.modules.utils import load_text_encoder, load_tokenizer

    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(model_path / "tokenizer")
    text_encoder = load_text_encoder(
        model_path / "text_encoder",
        torch_dtype=torch.bfloat16,
        torch_device=device,
    )
    text_encoder.eval()
    embedding = encode_empty_prompt(
        tokenizer,
        text_encoder,
        device=device,
    )

    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        torch.save(embedding, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    del text_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_path
