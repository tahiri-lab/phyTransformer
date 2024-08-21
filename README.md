Here’s an improved version of your README:

---

# phyTransformer

Welcome to the **phyTransformer** repository! This project explores the capabilities of Transformer models in the field of biology, particularly focusing on biological sequences. It provides a comprehensive review and practical implementation of these models, comparing their performance on various biological classification tasks.

## Required Libraries

To run the code in this repository, you will need the following libraries:

- `transformers`
- `datasets`
- `scikit-learn`
- `torch`
- `peft`
- `evaluate`
- `numpy`
- `pandas`
- `huggingface_hub`

You can install them using pip:

```bash
pip install transformers datasets scikit-learn torch peft evaluate numpy pandas huggingface_hub
```

## Dataset Sources

The datasets used in this project originate from several sources:

- **Uniprot Subcellular Localization**: This dataset is derived from a Google Colab notebook provided by Facebook, demonstrating how to fine-tune the ESM2 model.
- **Effectors, Fluorescence, Fold Classes, Neuropeptide, Remote Homology, Stability, Superfamily**: These datasets are available from the [BiologicalTokenizers GitHub repository](https://github.com/technion-cs-nlp/BiologicalTokenizers).
- **GLUE and IMDb**: These datasets are sourced from the Hugging Face course repositories.

## Pipelines

Pipelines in Hugging Face allow you to use pre-trained models for tasks without additional fine-tuning. However, they may not always perform optimally for specific, nuanced tasks in biology. Here’s what we used:

- **DistilBERT for Sentiment Analysis**: Used for text classification.
- **DistilBERT for Masked Language Modeling**: Predicts missing words in a sequence.
- **ESM2 Fine-Tuned for Protein Localization**: Directly compared with our custom fine-tuned models.
- **ESM2 for Masked Language Modeling**: Applied to biological sequences.

## Fine-Tuning

Fine-tuning involves training a pre-trained model on a specific dataset to adapt it for a specialized task. In this project, we fine-tuned the following models:

- **DistilBERT**: Fine-tuned for tasks like text classification.
- **ESM2**: Fine-tuned for protein localization.
- **GPT-2**: Fine-tuned for sequence-related tasks.

We compare the performance of these fine-tuned models to evaluate which one performs best on biological data. Additionally, we assess whether using Transformer models provides a significant advantage over traditional algorithms in biological sequence analysis.

## Conclusion

This repository demonstrates the potential of Transformer models in biological research, offering insights into their performance and applicability in real-world biological tasks. We hope this work contributes to the ongoing discussion about the benefits of using advanced machine learning models in bioinformatics.

# .gitignore
I put env where i installed all the libraries, config.py where i put the private hugging face model and some folders which are results from fine-tuning but too big for bieng push in Github