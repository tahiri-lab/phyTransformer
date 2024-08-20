#https://huggingface.co/distilbert/distilbert-base-uncased
from transformers import pipeline
unmasker = pipeline('fill-mask', model='distilbert-base-uncased')
print(unmasker("Hello I'm a [MASK] model."))
