# AI Supervised Learning

## Dataset: Steel Plate Defect Prediction

Dataset used was generated from a deep learning model trained on the Steel Plates Faults dataset from UCI.
> https://www.kaggle.com/competitions/playground-series-s4e3/overview

The original dataset of steel plate faults is classified into 7 different types. The goal was to train machine learning for automatic pattern recognition.
> https://archive.ics.uci.edu/dataset/198/steel+plates+faults

### Target variables:

The target variables are 7 types of defects: Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps and Other_Faults.

### Supervised learning problem:

Our objective is to predict the probability of each of the 7 binary targets on a given random steel plate that has 27 features.

### Instructions on how to use program:

If you haven't the necessary libraries installed run:
> pip install pandas numpy matplotlib seaborn

To run the EDA:
> python eda.py

Project developed by:
- Tomás Ferreira de Oliveira
- Diogo Miguel Fernandes Ferreira
- Álvaro Luís Dias Amaral Alvim Torres
