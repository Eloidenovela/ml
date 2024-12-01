import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("datasets/loan/loan.csv")

# X = data.drop(["LoanApproved"], axis=1)
X= data.drop(columns=["LoanApproved", "ApplicationDate"])

y = data["LoanApproved"]
X["EmploymentStatus"] = X["EmploymentStatus"].map({"Employed": 1, "Self-Employed": 2, "Unemployed": 3})
X["EducationLevel"] = X["EducationLevel"].map({'Master': 1, 'Associate': 2, 'Bachelor': 3, 'High School': 4, 'Doctorate': 5})
X["MaritalStatus"] = X["MaritalStatus"].map({'Married': 1, 'Single': 2, 'Divorced': 3, 'Widowed': 4})
X["LoanPurpose"] = X["LoanPurpose"].map({'Home': 1, 'Debt Consolidation': 2, 'Education': 3, 'Other': 4, 'Auto': 5})
X["HomeOwnershipStatus"] = X["HomeOwnershipStatus"].map({'Own':1, 'Mortgage': 2, 'Rent': 3, 'Other': 4})
# X["ApplicationDate"] = pd.to_datetime(X["ApplicationDate"]).astype(int) / 10**9

# print(X["LoanPurpose"].unique))

# print(X.dtypes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
new_data = [
    # [34,39877,497,1,2,13,24221,72,1,1,2,298,0.234362993,2,0,0.229685287,0,Auto,0,31,9,614,1486,60095,12884,3323.083333,0.900163149,5,47211,0.290721,0.318428604,757.6838879,0.317682039,0,54]
]

y_pred = model.predict(X_test)


# acuuracia = accuracy_score(y_test, y_pred)

# print(acuuracia * 100, "%")


import pickle as pkl

with open("training_scripts/001_train/model.pkl", "wb") as file:
    pkl.dump(model, file)