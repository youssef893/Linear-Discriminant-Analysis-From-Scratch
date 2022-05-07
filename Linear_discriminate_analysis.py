import numpy as np


class LDA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.linear_discriminants = None
        self.weights = np.array([])

    def fit(self, x_train, y_train, learning_Rate=1, epochs=1000):
        labels = set(y_train)  # get number of classes
        n_row, n_col = x_train.shape  # get data shape
        x_train = np.concatenate((np.array(x_train), np.ones((n_row, 1))), axis=1)  # add one for bias
        y_train = np.array(y_train)  # convert it to numpy

        # initialize weights with zeros with shape (labels_size * (features + bias))
        self.weights = np.zeros((len(labels), (n_col + 1)))
        iteration = 0

        while iteration < epochs:  # loop for epochs to update weights
            for index, label in enumerate(y_train):  # loop on all data
                # predict every point and return index of its class
                prediction = np.argmax(np.dot(self.weights, x_train[index]))

                if label != prediction:  # check if prediction not equal prediction to update weights
                    ''' 
                        apply this two equations:
                        w_new = w_old + lr * xi  where w_new is the weights of real label
                        w_new = w_old - lr * xi where w_new is the weight of wrong prediction    
                    '''
                    self.weights[label] += learning_Rate * x_train[index]
                    self.weights[prediction] -= learning_Rate * x_train[index]


            iteration += 1

    def fit_transform(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        # Scatter classes
        mean_total = np.mean(X, axis=0)
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))

        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            SW += (X_c - mean_c).T @ (X_c - mean_c)

            samples_inclass = X_c.shape[0]
            mean_difference = (mean_c - mean_total).reshape((n_features, 1))
            SB += samples_inclass * mean_difference @ mean_difference.T

        vector = np.linalg.inv(SW) @ SB
        eigenvalues, eigenvectors = np.linalg.eig(vector)
        eigenvectors = eigenvectors.T
        indices = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[indices]
        self.linear_discriminants = eigenvectors[:self.n_components]
        return X @ self.linear_discriminants.T

    def predict(self, x):
        n_row, _ = x.shape
        x_test = np.concatenate((np.array(x), np.ones((n_row, 1))), axis=1)
        return np.argmax(np.dot(x_test, self.weights.T), axis=1)
