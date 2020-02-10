import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn import svm

def import_data():
    train_data = pd.read_csv("kaggle/input/titanic/train.csv")
    test_data = pd.read_csv("kaggle/input/titanic/test.csv")
    return train_data, test_data

def data_wrangling(train_data, test_data):
    #drop Cabin, Ticket, Name column
    train_data.drop('Cabin',axis=1, inplace=True)
    test_data.drop('Cabin',axis=1, inplace=True)
    train_data.drop('Ticket',axis=1, inplace=True)
    test_data.drop('Ticket',axis=1, inplace=True)
    train_data.drop('Name',axis=1, inplace=True)
    test_data.drop('Name',axis=1, inplace=True)
    #Sex to numbers
    train_data['Sex'] = train_data['Sex'].map({'male':0, 'female':1})
    test_data['Sex'] = test_data['Sex'].map({'male':0, 'female':1})
    #Replace NaN values of: "Age", "Fare" with median values
    #Replace NaN values of "Embarked" with S
    train_data = train_data.fillna({"Age":train_data["Age"].dropna().median(), "Fare":train_data["Fare"].dropna().median(), "Embarked":'S'})
    test_data = test_data.fillna({"Age":test_data["Age"].dropna().median(), "Fare":test_data["Fare"].dropna().median(), "Embarked":'S'})
    #check for null values
    print(f"Training dataset has null values: {train_data.isnull().values.any()}")
    print(f"Test dataset has null values: {test_data.isnull().values.any()}")
    return train_data, test_data

def features(train_data, test_data):
    feat = ["Pclass", "Sex","Age","SibSp", "Parch","Fare", "Embarked"]
    y = train_data["Survived"]
    #turn into categorical data
    X = pd.get_dummies(train_data[feat])
    X_test = pd.get_dummies(test_data[feat])
    return X, y, X_test

def fit(X, y):
    model = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_split=2, random_state=1)
    model.fit(X, y)
    scores = model_selection.cross_val_score(model, X, y, scoring='accuracy', cv=50)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return model

def create_submission_file(model, X_test, test_data):
    predictions = model.predict(X_test)
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('my_submission.csv', index=False)
    print("Your submission was successfully saved!")

#Statistics without cross-validation
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

def print_stats(X, y, model):
    tp,fn,fp,tn = perf_measure(y, model.predict(X))
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*(precision*recall)/(precision+recall)
    print(f"accuracy: {accuracy:.3f}")
    print(f"precision: {precision:.3f}")
    print(f"recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
