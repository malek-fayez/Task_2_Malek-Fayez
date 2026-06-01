# Project 2: DecodeLabs Internship Track Classifier

## Overview

This project is a supervised machine learning classification system that recommends a suitable DecodeLabs internship track based on a student's skills and interests.

A small educational dataset was manually created for demonstration purposes. The model uses K-Nearest Neighbors to classify students into one of five internship tracks.

## Internship Tracks

The model classifies students into one of the following tracks:

- Artificial Intelligence
- Web Development
- Data Science
- Cyber Security
- Digital Marketing

## Features Used

The model uses the following input features:

- Python skill
- Web development interest
- Data interest
- Cyber security interest
- Digital marketing interest
- Problem-solving ability
- Math interest

## Technologies Used

- Python
- pandas
- scikit-learn

## Machine Learning Concepts Used

- Supervised learning
- Classification
- Dataset creation
- Feature and target separation
- Train-test split
- Feature scaling
- K-Nearest Neighbors classifier
- Accuracy score
- Confusion matrix
- Classification report
- Interactive prediction
- Input validation
- Equal-score handling

## How the System Works

1. A small dataset is created using student skill and interest scores.
2. The dataset is divided into features and target labels.
3. The data is split into training and testing sets.
4. Feature values are scaled using StandardScaler.
5. A KNN model is trained on the training data.
6. The model is evaluated using accuracy, confusion matrix, and classification report.
7. A new student can enter scores interactively.
8. The system validates that all scores are between 0 and 10.
9. If all scores are equal, the system explains that no clear track can be recommended.
10. If the input is valid, the system predicts the most suitable internship track.

## How to Install Requirements

Run:

```bash
py -m pip install pandas scikit-learn
```

## How to Run

```bash
python decodelabs_internship_classifier.py
```

or:

```bash
py decodelabs_internship_classifier.py
```

## Example Input

```text
Python skill: 9
Web development interest: 3
Data interest: 6
Cyber security interest: 2
Digital marketing interest: 1
Problem-solving ability: 9
Math interest: 9
```

## Example Output

```text
Recommended DecodeLabs Internship Track: Artificial Intelligence
```

## Notes

This project uses a small manually created dataset for educational purposes. The model is not intended to be a production-level recommendation system, but it demonstrates the full supervised classification pipeline.