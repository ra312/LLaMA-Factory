from datasets import Dataset, load_dataset
from typing import List, Dict
from multiprocessing import cpu_count
import logging
import os
import tempfile
import shutil

# Disable Hugging Face caching globally
os.environ["HF_DATASETS_CACHE"] = tempfile.mkdtemp()
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"

def chunk_text(text: str, chunk_size: int = 4096) -> List[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def split_content_map(examples: Dict[str, List]):
    all_chunks = []
    for text in examples['text']:
        chunks = chunk_text(text, chunk_size=300)
        all_chunks.extend(chunks)
    return {"text": all_chunks}

NUM_PROC = int(0.8 * cpu_count())
DATASET_PATH = 'ai-factory/red_pajama_subset_arxiv_subset'
DATASET_SPLIT = 'train'

logging.info("Loading dataset...")
dataset = load_dataset(
    path=DATASET_PATH,
    name='default',
    split=DATASET_SPLIT,
    trust_remote_code=True,
    num_proc=NUM_PROC,
    cache_dir=tempfile.mkdtemp(),  # avoid ~/.cache
).select_columns(['text'])

split_dataset = dataset.map(
    split_content_map,
    batch_size=1024,
    batched=True,
    num_proc=NUM_PROC,
)

flattened_dataset = split_dataset.flatten_indices()

def add_labels(example):
    example["ground_truth_answer"] = example["text"]
    example["question_2"] = example["text"]
    return example

flattened_dataset = flattened_dataset.map(
    add_labels,
    batched=True,
    batch_size=1024,
    remove_columns=['text'],
    num_proc=NUM_PROC,
)

flattened_dataset.to_json("train_arxiv.jsonl", lines=True, num_proc=NUM_PROC)

# Optional: clean up temp dirs (if you want zero disk trace)
shutil.rmtree(os.environ["HF_DATASETS_CACHE"], ignore_errors=True)
