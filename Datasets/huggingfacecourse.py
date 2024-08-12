#https://huggingface.co/learn/nlp-course/chapter3/2?fw=pt
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)
