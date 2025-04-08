import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from arxiv_modelling.arxiv_page_iterator import ArxivPageLoader

# Initialize the distributed environment
torch.distributed.init_process_group(backend='nccl')

# Define your model and tokenizer
model_name = "gpt2"  # Example model, replace with your model name

tokenizer = AutoTokenizer.from_pretrained("Xenova/gpt-4")
tokenizer.pad_token = (
    tokenizer.eos_token
)

model_name = "memevis/supp4"
model = AutoModelForCausalLM.from_pretrained(model_name)



# Wrap your model with FSDP
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=None,  # Optionally specify a policy for automatic wrapping
    mixed_precision=True  # Use mixed-precision training (optional but recommended for large models)
)

# Set your device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to GPU (this will work in a multi-GPU setup)
model = model.to(device)

# Training arguments
training_args = TrainingArguments(
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
    evaluation_strategy="steps",
    logging_first_step=True,
    eval_steps=500,  # Evaluation frequency
    report_to="tensorboard",  # Log to TensorBoard
)

pages = [0, 50, 100, 150]
dataset = ArxivPageLoader(tokenizer, pages)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
dataloader = DataLoader(dataset, batch_size=4, collate_fn=collator)

# Create DataLoader
# train_dataloader = DataLoader(dataset, batch_size=4, collate_fn=collator)

# Initialize Trainer with the model, args, and training data
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
    tokenizer=tokenizer,
)

# Start training
trainer.train()