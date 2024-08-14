import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

# Charger le dataset CSV
file_path = 'Uniprot_subcellular_location.csv'  # Chemin vers votre fichier
df = pd.read_csv(file_path)

# Convertir le DataFrame en dataset Hugging Face
dataset = Dataset.from_pandas(df)

# Créer un DatasetDict pour gérer potentiellement les ensembles de train/validation/test
dataset_dict = DatasetDict({"train": dataset})  # Vous pouvez aussi ajouter "test" ou "validation" si nécessaire

# Publier le dataset sur Hugging Face
dataset_name = "Uniprot_subcellular_location"  # Nom du dataset sur Hugging Face
username = "Paulhrd"  # Remplacez par votre nom d'utilisateur Hugging Face

# Publier sur Hugging Face
dataset_dict.push_to_hub(f"{username}/{dataset_name}", private=False)

print(f"Dataset publié sur Hugging Face: https://huggingface.co/datasets/{username}/{dataset_name}")
