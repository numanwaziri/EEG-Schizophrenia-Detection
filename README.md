# EEG Schizophrenia Detection

This repository contains the code for a project on Schizophrenia Detection using EEG data. The goal is to analyze EEG data and develop a model for detecting schizophrenia based on the provided datasets.

**Import Note:** The majority of the logic, preprocessing, and analysis are implemented in the `util` module. Refer to the `util` module for detailed functions and processing steps.

## Data
Code in this repository assumes that all data is locally downloaded and unzipped into a `dataset` folder. The `dataset` folder should include directories for each of the 81 subjects' data.

- [EEG Schizophrenia Detection Dataset Part 1](https://www.kaggle.com/datasets/broach/button-tone-sz/data)
- [EEG Schizophrenia Detection Dataset Part 2](https://www.kaggle.com/datasets/broach/buttontonesz2)

*Note: No raw data or output files are included in this repository due to size constraints.*

## **Project Result:**
In the domain of traditional models, the Light Gradient Boosting Machine (LGBM) proved to be a robust classifier, surpassing others in this project with a ROC AUC of **95.96%** and an accuracy of **90%**.

## Project Report
The detailed project report can be found [**here**](https://numanwaziri.github.io/posts/eeg-schizophrenia-detection).
