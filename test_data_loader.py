from llamafactory.train.arxiv_modelling.arxiv_page_iterator import ArxivPageLoader
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("Xenova/gpt-4")
tokenizer.pad_token = tokenizer.eos_token
page_loader = ArxivPageLoader(
    pages=[0, 50, 100, 150],
    page_size=50,
    chunk_size=4096,
    overlap=0,
    tokenizer=tokenizer,
)
import json
# Function to save tokenized data to a JSON file
def save_tokenized_data(file_path, tokenized_data):
    with open(file_path, 'w') as f:
        json.dump(tokenized_data, f)

# Tokenize and save
tokenized_data = []
for page in page_loader:
    for chunk in page["chunks"]:
        # Tokenize the text and create input_ids and labels
        tokenized_chunk = tokenizer(chunk["text"], truncation=True, padding=True, return_tensors="pt")
        input_ids = tokenized_chunk["input_ids"].squeeze().tolist()  # Get input_ids as a list
        labels = input_ids.copy()  # In causal language modeling, labels are usually the same as input_ids

        # Append the tokenized chunk
        tokenized_data.append({"input_ids": input_ids, "labels": labels})

# Save the tokenized data to a file
save_tokenized_data("tokenized_data.json", tokenized_data)