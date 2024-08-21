# phyTransforner
phyTransformer

add branch??

ALL LIBRAIRIES YOU WILL NEED
transformers
datasets
scikit-learn
torch
peft
evaluate
numpy
pandas
huggingface_hub

Where the Datasets come from?
Uniprot subcellular localization comes from a notebook google colab made by facebook on how fine tune the model ESM2

Effectors, fluorescence, fold_classes, neuropeptide, remote_homol, stability, superfamily vient du github
https://github.com/technion-cs-nlp/BiologicalTokenizers

huggingfacecourse - glue 

Shawnincourse - imdb from huggingface

What do pipelines ? Pipelines are model published on huggingface we can use them directly without fine tune but it generally don't answer exactly to our special tasks.

So 1 distilibert for sentiment analysis (text classifcation)
1 distilibert for Masked language Modeling
1 ESM finetuned protein localization we will compare it with our homemade finetuned
1 ESM for Masked Language Modeling

What do fine-tuning ?
Fine-tuning is pre-trained model who is train on a special task (for example text classifcation) with a dataset (for example Uniprot subcellular protein localization)
So there are 3 finetuned models DistilBERT, ESM2 and GPT2
We will compare diferrences with these different model to see which is the best and finally is it worth it to use transformer model than usual algorithm.
