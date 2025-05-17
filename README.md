

ADHD Prediction Project - README

Project Description


This project aims to create a support tool for general practitioners (GPs) to assist in diagnosing ADHD in children and adolescents. The system utilizes fMRI brain imaging data alongside socio-demographic information, emotional data, and parenting insights to predict both the sex of the individual and their ADHD diagnosis. Special care is taken to avoid bias toward female patients, as they are frequently underdiagnosed using traditional methods.

How to Run the Code

The code consists of three main notebooks:

adhd_final.ipynb: Main preprocessing and ADHD prediction model
model_eval.ipynb: Model evaluation and performance assessment
gender_bias.ipynb: Analysis and mitigation of gender bias


Run the notebooks in the specified order above using Jupyter Notebook or JupyterLab


Data should be placed in a folder named "Data" containing:

METADATA_A.xlsx
METADATA_B.xlsx
LABELS.xlsx
FUNCTIONAL_CONNECTOME_MATRICES.csv

Assumptions Made

Features with correlation above 0.8 are removed to reduce multicollinearity
kernel PCA is used to reduce dimensionality of brain connectome data
The model makes separate predictions for both sex and ADHD diagnosis
LIME is used for model explainability as required by GDPR
Gender bias is assessed by comparing performance metrics between male and female subjects
