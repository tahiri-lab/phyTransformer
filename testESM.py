from transformers import EsmForSequenceClassification, EsmTokenizer, pipeline

# Charger le modèle ESM pré-entraîné pour la classification de séquences
model_name = "facebook/esm2_t6_8M_UR50D"  # Remplace par le modèle ESM que tu veux utiliser
tokenizer = EsmTokenizer.from_pretrained(model_name)
model = EsmForSequenceClassification.from_pretrained(model_name)

# Créer une pipeline de classification
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Exemple de séquences de protéines
sequences = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQALQIPQAARAVFQKDWDLFGPNRSRDDLTIYMDGTGQRLVTRAKKEALQAA",
    "MGDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSYTAANKNKGIIWGADMGKTCKMVKIFKGDLYRGII"
]

# Classification des séquences
results = classifier(sequences)

# Afficher les résultats
for sequence, result in zip(sequences, results):
    print(f"Sequence: {sequence}")
    print(f"Classification: {result['label']} with score {result['score']:.4f}\n")
