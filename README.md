# AI-based dementia risk assessment

An MSc dissertation project by Sara Khalid. It uses structural MRI and cognitive assessments from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) to compare three machine-learning approaches. A Gradio prototype demonstrates how the trained models present predictions.

## Study design

The main study classified 702 matched ADNI participants as cognitively normal, mild cognitive impairment (MCI), or Alzheimer’s disease (AD). The MRI model combined three axial slices into a three-channel input for MobileNetV2. The cognitive Random Forest used demographic, cognitive, and functional assessments. The late-fusion Random Forest combined those features with the MRI model’s three class probabilities. Participant-level splitting kept data from the same person within one partition.

The separate [longitudinal experiment](docs/longitudinal.md) used baseline MRI and cognitive assessments to estimate the likelihood of progression from MCI to AD within 36 months.

## Reported test results

| Model | Three-class accuracy | 36-month ROC-AUC | 36-month accuracy |
| --- | ---: | ---: | ---: |
| MRI only | 41.84% | 0.567 | 56.8% |
| Cognitive only | 94.33% | 0.810 | 75.7% |
| MRI and cognitive fusion | 91.49% | 0.813 | 75.7% |

The cognitive model had the highest three-class accuracy. Adding MRI gave a small increase in ROC-AUC for the longitudinal task, with the same accuracy on the 37-participant holdout.

## Repository scope

This is a research showcase. The original experiments require authorised ADNI data, the original Colab environment and, for the Gradio prototype, trained model files. The reported results are from the dissertation’s ADNI evaluation. The prototype is for research and demonstration only.

