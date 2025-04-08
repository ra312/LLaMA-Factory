from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
import requests
import os




os.environ['HF_USER_TOKEN'] = os.getenv("HF_USER_TOKEN")

class ArxivPageLoader(IterableDataset):
    def __init__(self, tokenizer, pages, page_size=50, chunk_size=4096, overlap=0):
        self.tokenizer = tokenizer
        self.pages = pages  # a list of offsets
        self.page_size = page_size
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Load the Hugging Face token from env
        self.hf_token = os.getenv("HF_USER_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_USER_TOKEN not set. Please set it via `export HF_USER_TOKEN='your_token'`.")



    def fetch_page(self, offset):
        url = "https://datasets-server.huggingface.co/rows"
        params = {
            "dataset": "ai-factory/red_pajama_subset_arxiv_subset",
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": self.page_size
        }
        response = requests.get(url, params=params,)
        
        response.raise_for_status()
        return [row["row"].get("text", row["row"].get("content")) for row in response.json()["rows"]]

    def tokenize_and_chunk(self, text):
        tokens = self.tokenizer.encode(
            text, add_special_tokens=False, padding='max_length', max_length=self.chunk_size, truncation=True)
        step = self.chunk_size - self.overlap
        return [
            tokens[i:i+self.chunk_size] 
            for i in range(0, len(tokens), step)
        ]

    def __iter__(self):
        for offset in self.pages:
            texts = self.fetch_page(offset)
            for text in texts:
                chunks = self.tokenize_and_chunk(text)
                for chunk in chunks:
                    # Check length of chunk to make sure it's correct
                    chunk_length = len(chunk)
                    # print(f"Chunk length: {chunk_length}")  # Debugging line
                    
                    if chunk_length > self.chunk_size:
                        raise ValueError(f"Chunk length exceeds {self.chunk_size} tokens!")
                    
                    yield {
                        "input_ids": chunk,
                        "labels": chunk.copy()
                    }