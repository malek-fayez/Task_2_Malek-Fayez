# DecodeLabs Internship Track Classifier Using KNN

# -------------------------------------------------
# Project 2: DecodeLabs Internship Track Classifier
# -------------------------------------------------
# This project is a supervised machine learning classification project.
# The system recommends a suitable DecodeLabs internship track based on
# a student's skills and interests.
#
# A small educational dataset was created using features such as Python
# skill, web interest, data interest, cyber security interest, marketing
# interest, problem-solving ability, and math interest.
#
# The model uses the K-Nearest Neighbors algorithm to classify students
# into one of five tracks: Artificial Intelligence, Web Development,
# Data Science, Cyber Security, and Digital Marketing.
#
# The project includes train-test split, feature scaling, model training,
# model evaluation, interactive user input, range validation, and equal-score
# handling.
# -------------------------------------------------


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# -------------------------------------------------
# 1. Create a small sample dataset
# -------------------------------------------------

data = {
    "python_skill": [
        9, 8, 9, 8, 9, 7,
        4, 3, 5, 4, 2, 3,
        7, 6, 7, 8, 6, 7,
        3, 2, 4, 3, 2, 5,
        2, 3, 1, 4, 3, 2
    ],

    "web_interest": [
        3, 2, 3, 2, 3, 2,
        9, 8, 9, 7, 8, 9,
        2, 3, 2, 3, 2, 3,
        2, 3, 2, 4, 3, 2,
        4, 3, 2, 5, 4, 3
    ],

    "data_interest": [
        6, 5, 6, 5, 6, 5,
        3, 2, 4, 3, 2, 3,
        9, 8, 9, 9, 8, 9,
        2, 3, 2, 4, 3, 2,
        3, 4, 2, 3, 4, 2
    ],

    "security_interest": [
        2, 1, 2, 3, 2, 1,
        2, 1, 2, 3, 2, 1,
        2, 1, 2, 3, 2, 1,
        9, 8, 9, 7, 8, 9,
        2, 3, 2, 1, 2, 3
    ],

    "marketing_interest": [
        1, 2, 1, 2, 1, 2,
        2, 3, 2, 1, 2, 3,
        1, 2, 1, 2, 1, 2,
        2, 1, 2, 3, 2, 1,
        9, 8, 9, 7, 8, 9
    ],

    "problem_solving": [
        9, 9, 8, 9, 8, 9,
        6, 5, 6, 5, 4, 5,
        7, 6, 7, 6, 7, 6,
        7, 8, 7, 8, 7, 8,
        5, 4, 5, 6, 5, 4
    ],

    "math_interest": [
        9, 8, 9, 8, 9, 8,
        4, 3, 4, 3, 4, 3,
        8, 9, 8, 9, 8, 9,
        5, 4, 5, 4, 5, 4,
        3, 2, 3, 2, 3, 2
    ],

    "track": [
        "Artificial Intelligence", "Artificial Intelligence", "Artificial Intelligence",
        "Artificial Intelligence", "Artificial Intelligence", "Artificial Intelligence",

        "Web Development", "Web Development", "Web Development",
        "Web Development", "Web Development", "Web Development",

        "Data Science", "Data Science", "Data Science",
        "Data Science", "Data Science", "Data Science",

        "Cyber Security", "Cyber Security", "Cyber Security",
        "Cyber Security", "Cyber Security", "Cyber Security",

        "Digital Marketing", "Digital Marketing", "Digital Marketing",
        "Digital Marketing", "Digital Marketing", "Digital Marketing"
    ]
}

df = pd.DataFrame(data)

print("Dataset Preview:")
print(df.head())


# -------------------------------------------------
# 2. Separate features and target
# -------------------------------------------------

X = df.drop("track", axis=1)
y = df["track"]


# -------------------------------------------------
# 3. Split data into training and testing sets
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# -------------------------------------------------
# 4. Scale the features
# -------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------------------------------------
# 5. Train the KNN classification model
# -------------------------------------------------

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train_scaled, y_train)


# -------------------------------------------------
# 6. Make predictions
# -------------------------------------------------

predictions = model.predict(X_test_scaled)


# -------------------------------------------------
# 7. Evaluate the model
# -------------------------------------------------

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))


# -------------------------------------------------
# 8. Interactive prediction for a new student
# -------------------------------------------------

print("\nNow let's predict the best DecodeLabs internship track for a new student.")
print("Please enter a score from 0 to 10 for each question.")

def get_valid_score(question):
    while True:
        try:
            score = int(input(question))

            if 0 <= score <= 10:
                return score
            else:
                print("Invalid input. Please enter a number from 0 to 10.")

        except ValueError:
            print("Invalid input. Please enter numbers only.")


python_skill = get_valid_score("Python skill: ")
web_interest = get_valid_score("Web development interest: ")
data_interest = get_valid_score("Data interest: ")
security_interest = get_valid_score("Cyber security interest: ")
marketing_interest = get_valid_score("Digital marketing interest: ")
problem_solving = get_valid_score("Problem-solving ability: ")
math_interest = get_valid_score("Math interest: ")

student_scores = [
    python_skill,
    web_interest,
    data_interest,
    security_interest,
    marketing_interest,
    problem_solving,
    math_interest
]

if len(set(student_scores)) == 1:
    print("\nNew Student Prediction:")
    print("No clear track can be recommended because all scores are equal.")
    print("Please enter more specific scores to get a more accurate recommendation.")
else:
    new_student = pd.DataFrame(
        [student_scores],
        columns=X.columns
    )

    new_student_scaled = scaler.transform(new_student)

    predicted_track = model.predict(new_student_scaled)

    print("\nNew Student Prediction:")
    print("Recommended DecodeLabs Internship Track:", predicted_track[0])