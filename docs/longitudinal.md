# Predicting progression within 36 months

This experiment asks whether baseline cognitive assessments and structural MRI can distinguish stable MCI from progression to Alzheimer’s disease within 36 months.

The cohort included 252 participants, with 125 stable MCI and 127 progressive MCI cases. The same participant split was used for all models, with 177 for training, 38 for validation and 37 for testing. Future diagnoses defined the outcome label. Model inputs came from baseline assessments, and validation data guided model and threshold selection.

## Original Colab workflow

| Step | Notebook | Purpose |
| --- | --- | --- |
| 1 | `01_Longitudinal_Cohort_Creation.ipynb` | Define eligibility and outcomes, then create the fixed split. |
| 2 | `02_Cognitive_Risk_Model.ipynb` | Fit and evaluate a model using baseline cognitive features. |
| 3 | `03_MRI_Risk_Model.ipynb` | Train the MRI model from preprocessed baseline scans. |
| 4 | `04_Multimodal_Fusion_Risk_Model.ipynb` | Combine predictions and select fusion weights using validation data. |
| 5 | `05_Model_Comparison.ipynb` | Compare the models on the common held-out participants. |

Later notebooks use outputs from earlier stages. Reproducing the workflow requires authorised cognitive data and preprocessed MRI arrays in the paths expected by the original notebooks. The MRI notebook was designed for a Colab T4 GPU.

