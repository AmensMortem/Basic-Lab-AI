import pandas as pd
import Tree
from sklearn.preprocessing import LabelEncoder

test_cases = pd.DataFrame({
    "Outlook": ["Sunny", "Rain", "Overcast", "Sunny"],
    "Temperature": ["Cool", "Mild", "Hot", "Mild"],
    "Humidity": ["High", "Normal", "High", "Normal"],
    "Wind": ["Strong", "Weak", "Weak", "Strong"],
    "Tennis": ["No", "Yes", "Yes", "Yes"]
})

data = {
    "Outlook": [
        "Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast",
        "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"
    ],
    "Temperature": [
        "Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool",
        "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"
    ],
    "Humidity": [
        "High", "High", "High", "High", "Normal", "Normal", "Normal",
        "High", "Normal", "Normal", "Normal", "High", "Normal", "High"
    ],
    "Wind": [
        "Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong",
        "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"
    ],
    "Tennis": [
        "No", "No", "Yes", "Yes", "Yes", "No", "Yes",
        "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"
    ]
}

df = pd.DataFrame(data, index=["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14"])

unique_values = []
for column in df.columns:
    unique_values += df[column].unique().tolist()
le = LabelEncoder().fit(unique_values)
for column in df.columns:
    df[column] = le.transform(df[column])
x_train = df.drop(["Tennis"], axis=1)
y_train = df["Tennis"]
instance = Tree.DecisionTreeClassifierScratch()
for column in test_cases.columns:
    test_cases[column] = le.transform(test_cases[column])

instance.fit(X=x_train, y=y_train)

x_test = test_cases.drop(["Tennis"], axis=1)
y_test = test_cases["Tennis"]
y_pred = instance.predict(x_test)

train_acc = instance.score(x_train, y_train)
test_acc = instance.score(x_test, y_test)
print(train_acc)
print(test_acc)

print("\nTree structure:")
instance.print_tree()
