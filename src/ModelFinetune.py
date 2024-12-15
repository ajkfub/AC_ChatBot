# %%capture

# import torch
# major_version, minor_version = torch.cuda.get_device_capability()
# Must install separately since Colab has torch 2.2.1, which breaks packages
# !pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
# !pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers trl peft accelerate bitsandbytes

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import LoraConfig
import json
import pandas as pd
from datasets import Dataset

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

def initialize_model_and_tokenizer(
    model_name: str, max_seq_length: int, load_in_4bit: bool = True
) -> tuple:
    """
    Initialize the FastLanguageModel and tokenizer.

    Args:
        model_name (str): The name of the model to load.
        max_seq_length (int): The maximum sequence length for the model.
        load_in_4bit (bool): Whether to load the model in 4-bit quantization.

    Returns:
        tuple: A tuple containing the initialized model and tokenizer.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0; suggested: 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but 0 is optimized
        bias="none",  # Supports any, but "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
        random_state=3407,
        use_rslora=False,  # Supports rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
    return model, tokenizer


def format_prompts(data: dict, token: str) -> dict:
    """
    Format prompts for training.

    Args:
        data (dict): A dictionary containing the training data with keys 'instruction', 'input', and 'output'.
        token (str): The end-of-sequence token to append.

    Returns:
        dict: A dictionary containing formatted prompts for training.
    """
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                      ### Instruction:
                      {}

                      ### Input:
                      {}

                      ### Response:
                      {}"""

    instructions = data["instruction"]
    inputs = data["input"]
    outputs = data["output"]
    texts = []

    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + token
        texts.append(text)

    return {"text": texts}


def configure_trainer(
    model: FastLanguageModel, tokenizer, dataset: Dataset, max_seq_length: int
) -> SFTTrainer:
    """
    Configure the SFTTrainer with the given model, tokenizer, and dataset.

    Args:
        model (FastLanguageModel): The model to train.
        tokenizer: The tokenizer to use with the model.
        dataset (Dataset): The dataset for training.
        max_seq_length (int): The maximum sequence length for training.

    Returns:
        SFTTrainer: The configured trainer for fine-tuning.
    """
    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none")

    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=100,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        peft_config=lora_config,
        args=training_args,
    )

    return trainer


def save_model(model: FastLanguageModel, tokenizer, access_token: str) -> None:
    """
    Save the trained model locally and push it to the hub.

    Args:
        model (FastLanguageModel): The trained model to save.
        tokenizer: The tokenizer associated with the model.
        access_token (str): The access token for pushing to the hub.
    """
    model.save_pretrained("AC_ChatBot")  # Local saving
    model.push_to_hub("SamHunghaha/AC_ChatBot", token=access_token)  # Online saving

    # Save to q4_k_m GGUF
    model.save_pretrained_gguf("AC_ChatBot", tokenizer, quantization_method="f16")
    model.push_to_hub_gguf(
        "SamHunghaha/AC_ChatBot",
        tokenizer,
        quantization_method="f16",
        token=access_token,
    )


def get_dataset(filename: str, token: str) -> Dataset:
    """
    Load and format the dataset from a JSONL file.

    Args:
        filename (str): The path to the JSONL file containing the dataset.
        token (str): The end-of-sequence token to append.

    Returns:
        Dataset: The formatted dataset ready for training.
    """
    with open(filename) as f:
        data = f.readlines()

    data = [json.loads(i) for i in data]
    reshaped = [[i for i in data[j]["messages"]] for j in range(len(data))]
    formatted = [
        {
            "instruction": i[0]["content"],
            "input": i[1]["content"],
            "output": i[2]["content"],
        }
        for i in reshaped
    ]

    final = pd.DataFrame(formatted).to_dict("list")
    training_data = format_prompts(final, token)

    dataset = Dataset.from_dict(training_data)

    return dataset


def main() -> None:
    """Main function to execute the training pipeline."""
    # Get CUDA device capabilities
    major_version, minor_version = torch.cuda.get_device_capability()

    # Model parameters
    max_seq_length = 2048
    load_in_4bit = True  # Use 4-bit quantization to reduce memory usage

    # Initialize model and tokenizer
    model_name = "unsloth/llama-3-8b-bnb-4bit"
    model, tokenizer = initialize_model_and_tokenizer(
        model_name, max_seq_length, load_in_4bit
    )

    # Get EOS token
    eos_token = tokenizer.eos_token  # Must add EOS_TOKEN

    # Get dataset
    dataset = get_dataset(f"{parent_dir}/dataset/chat/all.jsonl", eos_token)

    # Configure and train the model
    trainer = configure_trainer(model, tokenizer, dataset, max_seq_length)

    trainer_stats = trainer.train()

    # Save the model and push to hub
    access_token = "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"
    save_model(model, tokenizer, access_token)


if __name__ == "__main__":
    main()
