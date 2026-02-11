# 🧬 Medical AI: Hybrid Diabetes Risk Prediction System
### Kaggle Playground Series S5E12 Solution

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green?style=for-the-badge&logo=xgboost&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> **A Next-Generation Clinical Decision Support System combining Deep Learning (ResNet) and Gradient Boosting Machines to predict diabetes risk with high precision.**

## 🚀 Project Overview

This project was developed for the **Kaggle Playground Series - Season 5, Episode 12**, focusing on predicting diabetes risk using a dataset derived from the *Centers for Disease Control and Prevention (CDC)*.

Unlike traditional prediction models, this solution implements a **Hybrid Ensemble Architecture**. It leverages the strengths of structured data learning (via Gradient Boosting) and complex pattern recognition (via Deep Neural Networks/ResNets) to achieve state-of-the-art performance.

### 🌟 Key Features
* **🧠 Hybrid Intelligence:** Fuses predictions from **XGBoost, LightGBM, CatBoost**, and a **Deep Neural Network (ANN/ResNet)**.
* **📊 Advanced EDA:** Comprehensive Exploratory Data Analysis including violin plots, correlation heatmaps, and pair plots to identify key risk factors (BMI, Age, Hypertension).
* **⚡ Tabular ResNet:** Implementation of Residual Networks specifically designed for tabular data to capture non-linear feature interactions.
* **💻 Interactive Web App:** A professional **Streamlit** dashboard with a custom dark-themed UI, real-time risk calculation, and dynamic health suggestions.
* **⚙️ Intelligent Preprocessing:** Automated scaling, one-hot encoding, and handling of class imbalances.

---

## 🏗️ System Architecture

The model architecture follows a **Weighted Voting Mechanism** to maximize the ROC-AUC score.

```mermaid
graph TD;
    Input[Raw Medical Data] --> Preprocessing[Standard Scaling & Encoding];
    Preprocessing --> GBM[Gradient Boosting Branch];
    Preprocessing --> DL[Deep Learning Branch];
    
    subgraph "Gradient Boosting Machines"
    GBM --> XGB[XGBoost];
    GBM --> LGBM[LightGBM];
    GBM --> CAT[CatBoost];
    end
    
    subgraph "Deep Neural Networks"
    DL --> ANN[Standard ANN];
    DL --> ResNet[Tabular ResNet];
    end
    
    XGB --> Ensemble[Weighted Voting Ensemble];
    LGBM --> Ensemble;
    CAT --> Ensemble;
    ANN --> Ensemble;
    ResNet --> Ensemble;
    
    Ensemble --> Final[Final Probability Score];
    Final --> UI[Streamlit Dashboard];
