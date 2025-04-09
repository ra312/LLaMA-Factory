from transformers import  DataCollatorForLanguageModeling
from llamafactory.train.arxiv_modelling.arxiv_page_iterator import ArxivPageLoader
from llamafactory.model import load_tokenizer

from llamafactory.train.arxiv_modelling.custom_trainer import CustomLMTrainer
from llamafactory.hparams.model_args import ModelArguments
from llamafactory.hparams.finetuning_args import FinetuningArguments
import logging

from llamafactory.train.sft import run_sft_next_token



if __name__ == "__main__":
    from llamafactory.hparams.model_args import ModelArguments
    from llamafactory.hparams.finetuning_args import FinetuningArguments
    from llamafactory.hparams.data_args import DataArguments
    from llamafactory.hparams.training_args import TrainingArguments
    model_args = ModelArguments(
        model_name_or_path='NemanTeam/nemesis',
        adapter_name_or_path='adapters/nemesis'
        
    )
    training_args = TrainingArguments(
        max_steps=10,
        output_dir="./output",  # Output directory
        overwrite_output_dir=True,
        num_train_epochs=3,  # Adjust the number of epochs
        per_device_train_batch_size=4,  # Batch size per device (GPU)
        logging_dir="./logs",
        logging_steps=500,
        save_steps=1000,
        save_total_limit=2,
        gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch size
        fp16=True,  # Enable mixed precision training
        eval_strategy="no",
        logging_first_step=True,
        eval_steps=500,  # Evaluation frequency
        report_to="tensorboard",  # Log to TensorBoard
    )
    finetuning_args = FinetuningArguments(
        lora_rank=2,
        lora_target='q_proj',
        lora_alpha=4,
        lora_dropout=0.1,
    )
    data_args = DataArguments(
        tokenized_path='/tokenized_data.json',
    )
    
    run_sft_next_token(
        model_args=model_args,
        finetuning_args=finetuning_args,
        data_args=data_args,
        training_args=training_args,
    )