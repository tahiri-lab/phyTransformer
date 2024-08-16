#https://huggingface.co/docs/transformers/en/model_doc/esm
from transformers import pipeline

# Charger le pipeline "fill-mask" avec le modèle ESM2
pipe = pipeline("fill-mask", model="facebook/esm2_t6_8M_UR50D")

# La séquence avec le token <mask> pour la prédiction
sequence = "MQIFVKTLTGKTITLEVEPS<mask>TIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

# Exécuter la prédiction
results = pipe(sequence)

# Afficher les résultats
for result in results:
    print(f"Token prédit: {result['token_str']}, Score: {result['score']:.4f}, Séquence complétée: {result['sequence']}")