import gradio as gr
from transformers import pipeline
import torch

# Chargement du pipeline de classification de texte avec le modèle spécifique
pipe = pipeline("text-classification", model="Rocketknight1/esm2_t6_8M_UR50D-finetuned-localization")

def classify_sequence(sequence):
    result = pipe(sequence)
    # Extraction de la localisation prédite à partir du résultat
    localization = result[0]['label']
    return localization

demo = gr.Interface(
    fn=classify_sequence, 
    inputs="text", 
    outputs="text",
    title="Biology Sequence Classifier",
    description="This application determines with a protein sequence given where the localization of this protein sequence comes from. You can try those sequences from each localization:\n'Cell.membrane': 'MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFK',\n'Cytoplasm': 'MDDDIAALVVDNGSGMCKAGFAGDDAPRAVFPSIVG',\n'Endoplasmic.reticulum': 'MKAAVRKVLTVLLLAAAVAGCGNASAEANQNGKPR',\n'Extracellular': 'MGRDGIDTDVFSGPDGKTGQSINNYGGFGADND',\n'Golgi.apparatus': 'MKSVLLLALSLWILPGGQVTQGVDLSSFGNSDLK',\n'Lysosome/Vacuole': 'MKTLLLAILAAWATAEAQTAAPCSGSADAAPTP',\n'Mitochondrion': 'MALWMRLLPLLALLALWGPGPGLSGLALLLAVAP',\n'Nucleus': 'MGLRSGRGKTGGKARAKAKSRSSRAGLQFPVGR',\n'Peroxisome': 'MNLREVRDPLPAHLGRFLRVAAAYRLARFGSD',\n'Plastid': 'MSTIAHRAMVALGEPNAETMGRLEREGAEVRN'",
    examples=None,
    theme=None,
    allow_flagging="never",
    live=False
)

demo.launch(share=True)
