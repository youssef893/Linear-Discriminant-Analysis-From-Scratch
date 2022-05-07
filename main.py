import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from Linear_discriminate_analysis import LDA


def get_data():
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target
    x = iris.target_names
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.15, random_state=20)

    return X_train, X_test, y_train, y_test, x


def plot_confusion_matrix(cf_matrix, name):
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title(f'Seaborn {name} Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Flower Category')
    ax.set_ylabel('Actual Flower Category ')
    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Setosa', 'Versicolor', 'Virginia'])
    ax.yaxis.set_ticklabels(['Setosa', 'Versicolor', 'Virginia'])
    # Display the visualization of the Confusion Matrix.
    plt.show()


def lda_model(lda, X_train, Y_train, target_names):
    data_plot = lda.fit_transform(X_train, Y_train)
    plt.figure()
    colors = ['red', 'green', 'blue']

    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(data_plot[Y_train == i, 0], data_plot[Y_train == i, 1], alpha=.8, color=color,
                    label=target_name)

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.show()


def main():
    x_train, x_test, y_train, y_test, target_names = get_data()
    lda = LDA()
    lda.fit(x_train, y_train)
    prediction = lda.predict(x_test)
    lda_model(lda, x_train, y_train, target_names)
    # predicting from the LDA model
    print(f"accuracy from LDA using Perceptron = {accuracy_score(y_test, prediction) * 100:.2f}%")
    print(f"precision_score from LDA using Perceptron = {precision_score(y_test, prediction, average='weighted') * 100:.2f}%")
    print(f"recall_score from LDA using Perceptron = {recall_score(y_test, prediction, average='weighted') * 100:.2f}%")
    plot_confusion_matrix(confusion_matrix(y_test, prediction), 'LDA using perceptron')
    model = LinearDiscriminantAnalysis()
    # fit model
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print(f"accuracy from built in LDA = {accuracy_score(y_test, prediction) * 100:.2f}%")
    print(f"precision_score from built in LDA = {precision_score(y_test, prediction, average='weighted') * 100:.2f}")
    print(f"recall_score from built in LDA = {recall_score(y_test, prediction, average='weighted') * 100:.2f}")
    plot_confusion_matrix(confusion_matrix(y_test, prediction), 'Built in LDA')


if __name__ == "__main__":
    main()
