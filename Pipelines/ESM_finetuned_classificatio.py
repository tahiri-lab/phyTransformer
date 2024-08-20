import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Charger le dataset
df = pd.read_csv("Datasets//Uniprot_subcellular_location.csv")

df = df.dropna()  # Supprimer les protéines avec des colonnes manquantes

cytosolic = df['Subcellular location [CC]'].str.contains("Cytosol") | df['Subcellular location [CC]'].str.contains("Cytoplasm")
membrane = df['Subcellular location [CC]'].str.contains("Membrane") | df['Subcellular location [CC]'].str.contains("Cell membrane")

cytosolic_df = df[cytosolic & ~membrane]
membrane_df = df[membrane & ~cytosolic]

cytosolic_sequences = cytosolic_df["Sequence"].tolist()
cytosolic_labels = [0 for _ in cytosolic_sequences]

membrane_sequences = membrane_df["Sequence"].tolist()
membrane_labels = [1 for _ in membrane_sequences]

sequences = cytosolic_sequences + membrane_sequences
labels = cytosolic_labels + membrane_labels

from sklearn.model_selection import train_test_split

train_sequences, test_sequences, train_labels, test_labels = train_test_split(sequences, labels, test_size=0.25, shuffle=True)

# Créer un pipeline de classification
pipe = pipeline("text-classification", model="Rocketknight1/esm2_t6_8M_UR50D-finetuned-localization")

# Map des labels textuels aux valeurs numériques
label_map = {"LABEL_0": 0, "LABEL_1": 1}

# Exécuter les prédictions
predictions = [pipe(sequence)[0]['label'] for sequence in test_sequences]

# Convertir les prédictions en valeurs numériques
predictions_mapped = [label_map.get(pred, -1) for pred in predictions]

# Calculer l'accuracy
accuracy = accuracy_score(test_labels, predictions_mapped)
print(f"L'accuracy du modèle est : {accuracy:.2f}")

# Calculer la matrice de confusion
conf_matrix = confusion_matrix(test_labels, predictions_mapped)

# Visualiser la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Cytosol", "Membrane"], yticklabels=["Cytosol", "Membrane"])
plt.ylabel('Vrais labels')
plt.xlabel('Prédictions')
plt.title('Matrice de confusion')
plt.show()
