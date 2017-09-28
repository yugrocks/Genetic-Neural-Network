import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

#visualize
plt.scatter(X[:, 0], X[:, 1], c = y, s= 30)
plt.show()

#split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#Scale the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Train a neural net without any regularization first
nb_hdn_neurons = 8
nb_output_neurons = 1 #binary classification
nb_features = 2

def get_model():
    model = Sequential()
    model.add(Dense(output_dim = nb_hdn_neurons, init = 'uniform', activation = 'relu', input_dim = nb_features))
    model.add(Dense(output_dim = nb_output_neurons, init = 'uniform', activation = 'sigmoid',
                               # kernel_regularizer=regularizers.l2(0.2)
                               ))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model


def train_nn():
    model = get_model()
    history = model.fit(X_train,y_train, batch_size=32,validation_data=(X_test, y_test), epochs=1000,verbose=2, 
                        #validation_split=0.1
                        )
    return history, model

#training with backprop
history, model = train_nn()

y_pred0 = model.predict_classes(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm0 = confusion_matrix(y_test, y_pred0)

"""
Min. Loss:
    Training = 0.2528
    Test     = 0.2000
Accuracies:
    Training = 0.9000
    Test     = 0.8962

"""


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict_classes(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Neural net(Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict_classes(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Neural Net(Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
