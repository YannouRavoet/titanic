import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
#Data
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
#Replace NaN values of: "Age", "Fare"
#Drop rows with NaN values for: "Cabin"
train_data = train_data.fillna({"Age":train_data["Age"].mean(), "Fare":test_data["Fare"].mean()})
test_data = test_data.fillna({"Age":test_data["Age"].mean(), "Fare":test_data["Fare"].mean()})
#train_data = train_data.dropna(subset=["Embarked"])
#test_data = test_data.dropna(subset=["Embarked"])
y = train_data["Survived"]
features = ["Pclass", "Sex","Age","SibSp", "Parch","Fare"]
#turn into categorical data
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])


#Model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)

#Statistics
def perf_measure(y, predictions):
    assert(len(y)==len(predictions))
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(len(y)):
        if y[i] == predictions[i] == 1:
            TP += 1
        elif y[i] != predictions[i] == 0:
            FN += 1
        elif y[i] != predictions[i] == 1:
            FP += 1
        else:
            TN += 1
    return TP,FN,FP,TN

def print_stats():
    tp,fn,fp,tn = perf_measure(y, model.predict(X))
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*(precision*recall)/(precision+recall)
    print(f"accuracy: {accuracy:.3f}")
    print(f"precision: {precision:.3f}")
    print(f"recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")

#Make submission file
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")