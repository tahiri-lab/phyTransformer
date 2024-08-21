import torch
from torch.optim import AdamW
from datasets import load_dataset
from transformers import BertTokenizer, DataCollatorWithPadding, get_scheduler

# Load the IMDb dataset
dataset = load_dataset("imdb")

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split the dataset into train and validation sets
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

# Data collator to dynamically pad inputs to the maximum length of the batch
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create data loaders
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = torch.utils.data.DataLoader(
    eval_dataset, batch_size=8, collate_fn=data_collator
)

# Now you can use these data loaders in your fine-tuning script
from accelerate import Accelerator
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Initialize the accelerator
accelerator = Accelerator()

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Prepare the model, optimizer, and dataloaders with accelerator
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

# Define training arguments and trainer
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    optimizers=(optimizer, None),  # Pass optimizer to Trainer
)

# Start the training process
trainer.train()
