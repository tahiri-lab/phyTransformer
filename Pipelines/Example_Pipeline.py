from transformers import pipeline

# Load the tokenizer and the model for text classification
model_name = "bert-base-uncased"
classifier = pipeline('sentiment-analysis', 
                        model=model_name, 
                        tokenizer=model_name)

# Test the tokenizer and the model with a sequence
sequence = "Hugging Face's Transformers library is incredibly useful for NLP tasks!"
result = classifier(sequence)

# Display the result
print(result)