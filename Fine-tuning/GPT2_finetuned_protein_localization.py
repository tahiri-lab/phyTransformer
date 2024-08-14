import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from peft import get_peft_model, LoraConfig, TaskType

# Charger le dataset
file_path = 'D:\projet TRANSFORMER\phyTransforner\Datasets\\Uniprot_subcellular_location.csv'
dataset = pd.read_csv(file_path)

# Prétraitement de la colonne "Subcellular location [CC]"
# Extraire uniquement le premier mot-clé de la localisation (ex: "Membrane", "Nucleus", etc.)
dataset['Location'] = dataset['Subcellular location [CC]'].str.extract(r'^SUBCELLULAR LOCATION: ([A-Za-z\s]+)')

# Supprimer les lignes avec des valeurs manquantes
dataset = dataset.dropna(subset=['Sequence', 'Location'])

# Encodage des labels
label_encoder = LabelEncoder()
dataset['Label'] = label_encoder.fit_transform(dataset['Location'])

# Préparer le DataFrame pour Hugging Face Dataset
dataset_hf = Dataset.from_pandas(dataset[['Sequence', 'Label']])

# Charger le tokenizer GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Ajouter un token de padding
tokenizer.pad_token = tokenizer.eos_token

# Tokenizer les séquences de protéines
def tokenize_function(examples):
    return tokenizer(examples['Sequence'], padding="max_length", truncation=True, max_length=512)

# Tokenization
tokenized_datasets = dataset_hf.map(tokenize_function, batched=True)

# Préparer les colonnes pour le modèle GPT-2
tokenized_datasets = tokenized_datasets.rename_column("Label", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Diviser les données en ensembles d'entraînement et de validation
train_test_split = tokenized_datasets.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Charger le modèle GPT-2 avec une tête de classification
model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=len(label_encoder.classes_))

# Configurer le modèle pour utiliser le pad_token
model.config.pad_token_id = tokenizer.pad_token_id

# Configurer LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,   # Tâche de classification de séquence
    inference_mode=False,         # Mode entraînement
    r=8,                          # Rang des matrices LoRA
    lora_alpha=16,                # Alpha de LoRA (multiplicateur des matrices)
    lora_dropout=0.1,             # Taux de dropout pour LoRA
)

# Appliquer LoRA au modèle
model = get_peft_model(model, lora_config)

# Configurer l'entraînement
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Correction apportée ici
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # Ajustement pour éviter l'erreur
    per_device_eval_batch_size=1,   # Ajustement pour éviter l'erreur
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
)

# Définir une fonction de calcul des métriques (par exemple, l'accuracy)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = (predictions == labels).float().mean()
    return {"accuracy": accuracy.item()}

# Créer le Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Fine-tuning du modèle
trainer.train()

# Sauvegarder le modèle fine-tuné
trainer.save_model("./fine-tuned-gpt2")

# Évaluer le modèle sur l'ensemble de validation
eval_results = trainer.evaluate()

# Afficher les résultats d'évaluation
print(f"Résultats d'évaluation : {eval_results}")
