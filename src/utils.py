import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
import sklearn.tree as tree
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def import_data():
    train_data = pd.read_csv("kaggle/input/titanic/train.csv")
    test_data = pd.read_csv("kaggle/input/titanic/test.csv")
    return train_data, test_data

def data_wrangling(train_data, test_data):
    #Replace NaN values of: "Age", "Fare" with median values and of "Embarked" with S
    train_data = train_data.fillna({"Age":train_data.groupby('Sex')["Age"].transform("median"), "Fare":train_data.groupby('Pclass')["Fare"].transform("median"), "Embarked":'S'})
    test_data = test_data.fillna({"Age":test_data.groupby('Sex')["Age"].transform("median"), "Fare":test_data.groupby('Pclass')["Fare"].transform("median"), "Embarked":'S'})
    #Drop PassengerId in train_data
    train_data.drop('PassengerId', axis=1, inplace=True)
    for dataset in [train_data, test_data]:
        #Age binning
        dataset['AgeBin'] = 0
        dataset.loc[dataset['Age'] <= 16, 'AgeBin'] = 0
        dataset.loc[ (dataset['Age'] > 16) & (dataset['Age'] <= 26), 'AgeBin'] = 1
        dataset.loc[ (dataset['Age'] > 26) & (dataset['Age'] <= 36), 'AgeBin'] = 2
        dataset.loc[ (dataset['Age'] > 36) & (dataset['Age'] <= 62), 'AgeBin'] = 3
        dataset.loc[ (dataset['Age'] > 62), 'AgeBin'] = 4
        #Fare binning
        dataset['FareBin'] = 0
        dataset.loc[dataset['Fare'] <= 17, 'FareBin'] = 0
        dataset.loc[ (dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'FareBin'] = 1
        dataset.loc[ (dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'FareBin'] = 2
        dataset.loc[ (dataset['Fare'] > 100) , 'FareBin'] = 3
        #Cabin mapping
        cabin_distribution = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
        dataset['Cabin'] = dataset['Cabin'].str[:1]
        dataset['Cabin'] = dataset['Cabin'].map(cabin_distribution)
        #create Family Size column
        familysize_distribution = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11:4}
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] +1
        dataset['FamilySize'] = dataset['FamilySize'].map(familysize_distribution)
        #drop Ticket and Name column
        dataset.drop(['Name', 'Ticket'], axis=1, inplace=True)
        #Sex, Embarked to numbers
        dataset['Sex'] = dataset['Sex'].map({'male':0, 'female':1})
        dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2})
    #fillNa for cabin
    train_data["Cabin"].fillna(train_data.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
    test_data["Cabin"].fillna(test_data.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
    #check for null values
    print(f"Training dataset has null values: {train_data.isnull().values.any()}")
    print(f"Test dataset has null values: {test_data.isnull().values.any()}")
    return train_data, test_data

def features(train_data, test_data):
    feat_names = ["Pclass", "Sex","AgeBin", "FareBin", "Embarked", "FamilySize", "Cabin"]
    y = train_data["Survived"]
    #turn into categorical data
    X = pd.get_dummies(train_data[feat_names])
    X_test = pd.get_dummies(test_data[feat_names])
    return X, y, X_test

def fit(X, y):
    #model = AdaBoostClassifier(n_estimators=500)
    #model = RandomForestClassifier(n_estimators=500, max_depth=7, random_state=1)
    #model = DecisionTreeClassifier(max_depth=7, random_state=1)
    model = SVC(gamma='auto')
    model.fit(X, y)
    # feat_names = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Child"]
    # tree.export_graphviz(model, feature_names=feat_names, out_file="../plots/dt.dot")

    #scores = model_selection.cross_val_score(model, X, y, scoring='accuracy', cv=50)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return model

def create_submission_file(model, X_test, test_data):
    predictions = model.predict(X_test)
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('my_submission.csv', index=False)
    print("Your submission was successfully saved!")

#Statistics without cross-validation
def perf_measure(y, predictions):
    assert(len(y)==len(predictions))
    tn, fp, fn, tp =  confusion_matrix(y, predictions).ravel()
    # accuracy = (tp+tn)/(tp+tn+fp+fn)
    # precision = tp/(tp+fp)
    # recall = tp/(tp+fn)
    # f1 = 2*(precision*recall)/(precision+recall)
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)
    return accuracy, precision, recall, f1

def print_stats(X, y, model):
    accuracy, precision, recall, f1 = perf_measure(y, model.predict(X))
    print(f"accuracy: {accuracy:.3f}")
    print(f"precision: {precision:.3f}")
    print(f"recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
