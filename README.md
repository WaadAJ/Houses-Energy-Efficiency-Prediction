# Houses' Energy Efficiency Prediction

A machine learning project that predicts the energy-efficiency rating of residential houses and provides the prediction through a user-friendly chatbot.

## Overview

Energy efficiency is an important consideration for homeowners and consumers when buying or renting a house. However, determining the energy-efficiency profile of a home can be difficult without specialized knowledge.

This project uses **machine learning to estimate the energy-efficiency rating of residential houses**. The selected model was then integrated into a chatbot prototype to make the prediction more accessible and user-friendly.

This project was developed as a senior project in the **Department of Information Systems & Technology**.

## Objectives

The main objectives of this project were to:

- Review existing research related to energy efficiency in residential housing.
- Collect, explore, prepare, and preprocess the dataset.
- Build and evaluate multiple machine learning models.
- Select the best-performing model for predicting the energy-efficiency rating band.
- Integrate the selected model into a chatbot prototype.
- Communicate the results and identify opportunities for future development.

## Methodology

The project followed the following workflow:

**Data Collection → Data Exploration → Data Preprocessing → Model Training → Model Evaluation → Model Selection → Chatbot Integration**

Multiple machine learning models were trained and evaluated to determine the most suitable model for predicting the energy-efficiency rating band.

### Machine Learning Models

The following models were evaluated:

- Support Vector Machine (SVM)
- Random Forest (RF)
- Deep Artificial Neural Network (ANN)
- XGBoost

## Results

| Model | Accuracy |
|:---|---:|
| Support Vector Machine (SVM) | 72% |
| Random Forest (RF) | 86% |
| Deep ANN | 87% |
| **XGBoost** | **89%** |

**XGBoost achieved the highest accuracy of 89%** and was selected for integration into the chatbot prototype.

## Chatbot

The selected XGBoost model was integrated into a chatbot prototype that allows users to provide information about a house and receive an estimated **energy-efficiency rating band**.

The chatbot was designed to make energy-efficiency estimation more accessible to consumers without requiring them to interpret complex technical information.

## Project Structure

```text
Houses-Energy-Efficiency-Prediction/
│
├── data/
│   └── dataset files
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── preprocessing.ipynb
│   └── model_training.ipynb
│
├── models/
│   └── trained model files
│
├── chatbot/
│   └── chatbot implementation
│
├── results/
│   └── evaluation results and visualizations
│
├── README.md
└── requirements.txt
