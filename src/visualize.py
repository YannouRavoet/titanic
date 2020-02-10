from utils import import_data, data_wrangling
import matplotlib.pyplot as plt
import numpy as np
#see https://www.youtube.com/watch?v=fS70iptz-XU - Predicting Titanic Survivors with Machine Learning
def plot_basics(train_data):
    plt.figure(figsize=(18,6))
    #Survived percentages
    plt.subplot2grid((3,3),(0,0))
    train_data.Survived.value_counts(normalize=True).plot.bar(alpha=0.5)
    plt.xticks(rotation='horizontal')
    plt.title('Survival Percentages')
    #Age wrt Survived
    plt.subplot2grid((3,3),(0,1))
    plt.scatter(train_data.Survived, train_data.Age, alpha=0.1)
    plt.title('Age wrt Survived')
    #Passenger Class
    plt.subplot2grid((3,3),(0,2))
    train_data.Pclass.value_counts(normalize=True).sort_index().plot.bar(alpha=0.5)
    plt.xticks(rotation='horizontal')
    plt.title('Passenger Class Distribution')
    #Passenger Class wrt Age
    plt.subplot2grid((3, 3), (1, 0), colspan=2)
    for i in [1,2,3]:
        train_data.Age[train_data.Pclass == i].plot.kde()
    plt.title("Passenger Class wrt Age")
    plt.legend(['1st','2nd','3rd'])
    #Embarked percentages
    plt.subplot2grid((3,3),(1,2))
    train_data.Embarked.value_counts(normalize=True).plot.bar(alpha=0.5)
    plt.xticks(ticks=np.arange(3), labels=['Southampton', 'Cherbourg', 'Queenstown'], rotation='horizontal')
    plt.title('Embarked distribution')
    #Gender distribution per Class
    ax = plt.subplot2grid((3,3),(2,0))
    men_data = []
    women_data = []
    for i in [1,2,3]:
        data = train_data.Sex[train_data.Pclass == i].value_counts(normalize=True)
        men_data.append(data[0])
        women_data.append(data[1])
    ind = np.arange(3)
    width = 0.25
    plt.bar(ind, men_data, width, bottom=0, alpha=0.5)
    plt.bar(ind+width, women_data, width, bottom=0, alpha=0.5, color='#FA0000')
    ax.set_xticklabels(['1st','2nd','3rd'])
    plt.title('Gender distribution wrt Passenger Class')
    plt.xticks(ind+ width/2)
    plt.legend(['Men', 'Women'])
    #Survived wrt SibSp
    plt.subplot2grid((3,3),(2,1))
    data = train_data[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    data["Survived"].plot.bar(alpha=0.5)
    plt.xticks(rotation='horizontal')
    plt.title('Survival percentage wrt number of siblings/spouse')
    #Survived wrt Parch
    plt.subplot2grid((3,3),(2,2))
    data = train_data[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    data["Survived"].plot.bar(alpha=0.5)
    plt.xticks(rotation='horizontal')
    plt.title('Survival percentage wrt number of parents/children')
    plt.show()

def plot_gender(train_data):
    plt.figure(figsize=(18, 6))
    # Survived percentages
    plt.subplot2grid((3, 4), (0, 0))
    train_data.Survived.value_counts(normalize=True).plot.bar(alpha=0.5)
    plt.xticks(rotation='horizontal')
    plt.title('Survived percentages')
    # Men Survived
    plt.subplot2grid((3, 4), (0, 1))
    train_data.Survived[train_data.Sex == 'male'].value_counts(normalize=True).plot.bar(alpha=0.5)
    plt.xticks(rotation='horizontal')
    plt.title('Men Survived')
    # Women Survived
    plt.subplot2grid((3, 4), (0, 2))
    train_data.Survived[train_data.Sex == 'female'].value_counts(normalize=True).plot.bar(alpha=0.5, color='#FA0000')
    plt.xticks(rotation='horizontal')
    plt.title('Women Survived')
    # Sex of survived
    plt.subplot2grid((3, 4), (0, 3))
    train_data.Sex[train_data.Survived == 1].value_counts(normalize=True).plot.bar(alpha=0.5, color=['#FA0000', '#2d92cc'])
    plt.xticks(rotation='horizontal')
    plt.title('Sex of survived')
    #Passenger Class wrt Survived
    plt.subplot2grid((3, 4), (1, 0), colspan=4)
    for i in [1,2,3]:
        train_data.Survived[train_data.Pclass == i].plot.kde()
    plt.title("Passenger Class wrt Survived")
    plt.legend(['1st','2nd','3rd'])
    # Rich men
    plt.subplot2grid((3, 4), (2, 0))
    train_data.Survived[(train_data.Sex == 'male') & (train_data.Pclass == 1)].value_counts(normalize=True).plot.bar(alpha=0.5)
    plt.xticks(rotation='horizontal')
    plt.title('Rich men survived')
    # Poor men
    plt.subplot2grid((3, 4), (2, 1))
    train_data.Survived[(train_data.Sex == 'male') & (train_data.Pclass == 3)].value_counts(normalize=True).plot.bar(alpha=0.5)
    plt.xticks(rotation='horizontal')
    plt.title('Poor men survived')
    # Rich women
    plt.subplot2grid((3, 4), (2, 2))
    train_data.Survived[(train_data.Sex == 'female') & (train_data.Pclass == 1)].value_counts(normalize=True).plot.bar(alpha=0.5, color='#FA0000')
    plt.xticks(rotation='horizontal')
    plt.title('Rich women survived')
    # Poor women
    plt.subplot2grid((3, 4), (2, 3))
    train_data.Survived[(train_data.Sex == 'female') & (train_data.Pclass == 3)].value_counts(normalize=True).plot.bar(alpha=0.5, color='#FA0000')
    plt.xticks(rotation='horizontal')
    plt.title('Poor women survived')

    plt.show()

if __name__ == "__main__":
    train_data, test_data = import_data()
    #train_data, test_data = data_wrangling(train_data, test_data)
    plot_basics(train_data)
    plot_gender(train_data)
