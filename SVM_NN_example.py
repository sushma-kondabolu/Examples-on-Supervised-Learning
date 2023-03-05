import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pylab as plt


class NN:
    def __init__(self, n_hidden, n_output, epochs, batch_size, learning_rate):
        self.learning_rate = learning_rate
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.epochs = epochs ##30
        self.batch_size = batch_size
        self.layers_size = [self.n_hidden,self.n_output]
        self.parameters = {}
        self.L = len(self.layers_size)
        self.n = 0
        self.costs = []
        self.costsVal = []
        self.accuracy = []

    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s
    
    def sigmoid_deriv(self, x):
        s = 1 / (1 + np.exp(-x))
        d = s * (1 - s)
        return d

    def softmax(self, x):
        expZ = np.exp(x - np.max(x))
        d =  expZ / expZ.sum(axis=0, keepdims=True)
        return d

    
    def feed_forward(self, X):
        cache = {}
        A = X.T
        for l in range(self.L - 1):
            Z = self.parameters["W" + str(l + 1)].dot(A) + self.parameters["b" + str(l + 1)]
            A = self.sigmoid(Z)
            cache["A" + str(l + 1)] = A
            cache["W" + str(l + 1)] = self.parameters["W" + str(l + 1)]
            cache["Z" + str(l + 1)] = Z

        Z = self.parameters["W" + str(self.L)].dot(A) + self.parameters["b" + str(self.L)]
        output = self.softmax(Z)
        cache["A" + str(self.L)] = output
        cache["W" + str(self.L)] = self.parameters["W" + str(self.L)]
        cache["Z" + str(self.L)] = Z
        return cache, output

    def back_propagate(self, X, Y, cache):
        grads = {}
        cache["A0"] = X.T
        A = cache["A" + str(self.L)]
        dZ = A - Y.T
        dW1 = dZ.dot(cache["A" + str(self.L - 1)].T) / self.n
        db1 = np.sum(dZ, axis=1, keepdims=True) / self.n
        dAPrev = cache["W" + str(self.L)].T.dot(dZ)

        grads["dW" + str(self.L)] = dW1
        grads["db" + str(self.L)] = db1

        for l in range(self.L - 1, 0, -1):
            dZ = dAPrev * self.sigmoid_deriv(cache["Z" + str(l)])
            dW2 = 1. / self.n * dZ.dot(cache["A" + str(l - 1)].T)
            db2 = 1. / self.n * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAPrev = cache["W" + str(l)].T.dot(dZ)
            grads["dW" + str(l)] = dW2
            grads["db" + str(l)] = db2
            #grads["dW2"] = dW2
            #rads["db2"] = db2

        return grads

    def init_weights(self, n_input):
        np.random.seed(1)
        for l in range(1, len(self.layers_size)):
            self.parameters["W" + str(l)] = np.random.randn(self.layers_size[l], self.layers_size[l - 1]) / np.sqrt(
                self.layers_size[l - 1])
            self.parameters["b" + str(l)] = np.zeros((self.layers_size[l], 1))
    
    def update_weights(self, grads):
        for l in range(1, self.L + 1):
                self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - self.learning_rate * grads["dW" + str(l)]
                self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - self.learning_rate * grads["db" + str(l)]
    
    def compute_loss(self, y, output):
        #m = y.shape[1]
        m = y.shape[0]
        loss = - (1 / m) * np.sum(np.multiply(y, np.log(output)) + np.multiply(1 - y, np.log(1 - output)))
        return loss

    def iterate_minibatches(self, X, Y, batchsize, shuffle=True):
        assert X.shape[0] == Y.shape[0]
        if shuffle:
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
        yield X[excerpt], Y[excerpt]


    def train(self, X, Y, x_val, y_val):
        np.random.seed(1)
        (n, m) = X.shape
        self.n = n
        self.layers_size.insert(0, X.shape[1])
        self.init_weights(self.n)
        print("Training")
        for loop in range(self.epochs):
            n_iteration = int(X.shape[0]/self.batch_size)
            costList = []
            for iteration in range(n_iteration):
                #costList = []
                for batch in self.iterate_minibatches(X, Y, self.batch_size):
                    X_mini, Y_mini = batch
                    cache, A = self.feed_forward(X_mini)
                    #cost1 = -np.mean(Y_mini * np.log(A.T))
                    cost = self.compute_loss(Y_mini,A.T)
                    costList.append(cost)
                    derivatives = self.back_propagate(X_mini, Y_mini, cache)
                    self.update_weights(derivatives)
            costAvg = sum(costList)/len(costList)
            accuracy_Train=self.test(X, Y)
            print("Ep:",loop+1, "Average Cost: ", costAvg, "Training Accuracy:", accuracy_Train)
            self.accuracy.append(accuracy_Train)
            self.costs.append(costAvg)
        print("Validation set")
        for loop in range(self.epochs):
            n_iteration = int(x_val.shape[0]/self.batch_size)
            costList = []
            for iteration in range(n_iteration):
                for batch in self.iterate_minibatches(x_val, y_val, self.batch_size):
                    X_mini, Y_mini = batch
                    cache, A = self.feed_forward(X_mini)
                    cost = self.compute_loss(Y_mini,A.T)
                    costList.append(cost)
            costAvg = sum(costList)/len(costList)
            accuracy_Val=self.test(X, Y)
            print("Ep:",loop+1, "Average Cost: ", costAvg, "Validation Accuracy:", accuracy_Val)
            self.accuracy.append(accuracy_Val)
            self.costsVal.append(costAvg)

    def test(self, X, Y):
        cache, output = self.feed_forward(X)
        accuracy = self.compute_accuracy(Y, output)
        return accuracy
    
    def compute_accuracy(self, y, output):
        accuracy = (np.argmax(output, axis=0) == np.argmax(y, axis=1)).sum() * 1. / y.shape[0]
        return accuracy

    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.costs)), self.costs, color='r', label='Training Set')
        plt.plot(np.arange(len(self.costsVal)), self.costsVal, color='g', label='Validation Set')
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.show()
    
    def plot_accuracy(self):
        plt.figure()
        plt.plot(np.arange(len(self.accuracy)), self.accuracy)
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.show()

    def one_hot_labels(self, y):
        one_hot_labels = np.zeros((y.size, self.n_output))
        one_hot_labels[np.arange(y.size), y.astype(int)] = 1
        return one_hot_labels


def main():
    nn = NN(n_hidden=300, n_output=10, epochs=30, batch_size=1000, learning_rate=5)
    X, y = fetch_openml('mnist_784', return_X_y=True)
    np.random.seed(100)
    X = (X / 255).astype('float32')
    X_train, y_train = X[0:60000], y[0:60000]
    y_train = nn.one_hot_labels(y_train)
    p = np.random.permutation(60000)
    X_train = X_train[p]
    y_train = y_train[p]
    
    ##Validation set 10,000##
    val_x = X_train[0:10000]
    val_y = y_train[0:10000]
	##Train set 50,000
    train_x = X_train[10000:]
    train_y = y_train[10000:]
	##Test set 10,000
    test_x, y_test = X[60000:], y[60000:]
    test_y = nn.one_hot_labels(y_test)

    nn.train(train_x, train_y, val_x, val_y)
    #print("Train set Accuracy:", nn.test(train_x, train_y))
    #print("Validation set Accuracy:", nn.test(val_x, val_y))
    #print("Test set Accuracy:", nn.test(test_x, test_y))
    accuracy = nn.test(test_x, test_y)
    print(f'Test accuracy: {accuracy}')
    nn.plot_cost()
    nn.plot_accuracy()

if __name__ == '__main__':
    main()
    
