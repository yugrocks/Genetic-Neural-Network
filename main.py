import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from model import model
from random import random, choice, randint
from numpy import copy
import operator

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

# adding the column of ones in X for the bias term
X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis = 1)
X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis = 1)

# hyperparameters
nb_hdn_neurons = 8
nb_output_neurons = 1 #binary classification
input_size = 3 #including bias
W1_shape = (8, 3)
W2_shape = (1, 9)
initial_population_size = 50
# total parameters = 8*3 + 9*1 = 33


population = [] #the very first population

def initialize_population():
    #convention: each chromosome will have dimensions 1x33
    for i in range(initial_population_size):
        W1 = np.random.randn(W1_shape[0], W1_shape[1])
        W2 = np.random.randn(W2_shape[0], W2_shape[1])
        population.append(np.concatenate((W1.flatten().reshape(1,W1_shape[0]*W1_shape[1] ), W2.flatten().reshape(1,W2_shape[0]*W2_shape[1] )),axis = 1))

def get_weights_from_encoded(individual):
    W1 = individual[:, 0:W1_shape[0]*W1_shape[1]]
    W2 = individual[:, W1_shape[0]*W1_shape[1]:]
    return (W1.reshape(W1_shape[0], W1_shape[1]), W2.reshape(W2_shape[0], W2_shape[1]))

def generate_random_chromosome():
    W1 = np.random.randn(W1_shape[0], W1_shape[1])
    W2 = np.random.randn(W2_shape[0], W2_shape[1])
    return np.concatenate((W1.flatten().reshape(1,W1_shape[0]*W1_shape[1] ), W2.flatten().reshape(1,W2_shape[0]*W2_shape[1] )),axis = 1)
    

def get_losses(population): #The rank function
    losses = []
    for individual in population:
        mdl = model(get_weights_from_encoded(individual))
        losses.append(mdl.loss(X_train, y_train))
    zip1 = zip(losses,population)
    sorted_results = sorted(zip1, key=operator.itemgetter(0))
    sorted_pop = [x for _,x in sorted_results]
    sorted_losses = [_ for _,x in sorted_results]
    return sorted_pop, sorted_losses



def mutate(chromosome, prob):
    if random() >= prob:
        return chromosome, False # No mutation done
    else:
        #mutate each element with a probability of 'prob'
        mutated = False
        chromosome0 = copy(chromosome)
        operators = ['add', 'subtract']
        for i in range(len(chromosome0)):
            if random() < prob:
                if choice(operators) == 'add':
                    chromosome0[i] += random()
                    mutated = True
                else:
                    chromosome0[i] -= random()
                    mutated = True
        return chromosome0, mutated # mutated
    

def crossover(chromosomes, prob):
    # here the argument chromosomes is a list containing two parent chromosomes
    if random() >= prob:
        return chromosomes, False # No crossover done
    else:
        #select a random position from the index, around which the values will be swapped
        indx = randint(1, chromosomes[0].shape[1]-1)
        p0 = copy(chromosomes[0]); p1 = copy(chromosomes[1])
        med = copy(p0)
        p0[:, 0:indx] = p1[:, 0:indx]
        p1[:, 0:indx] = med[:, 0:indx]
        return [p0, p1], True
    

def crossover2(chromosomes, prob):
    # here the argument chromosomes is a list containing two parent chromosomes
    #for every index along the length of both chromosomes, randomly select if it has to be swapped
    p0 = copy(chromosomes[0]); p1 = copy(chromosomes[1])
    crossovered = False
    for i in range(chromosomes[0].shape[1]):
        if random() < prob:
            #swap the numbers at index i
            p0[0, i] = chromosomes[1][0][i]
            p1[0, i] = chromosomes[0][0][i]
            crossovered = True
    return [p0, p1], crossovered


def selectindex():
    return randint(0, 10) #including the 11th element


def evolve(initial_population ,max_iter = 20,min_desired_loss = None, crossover_prob = 0.7, mutation_prob = 0.2, crossover2_prob = 0.2):
    population = initial_population
    for iteration in range(max_iter):
        # create population, 
        # breed, mutate, and keep the 5 best to the next generation unchanged for next generation
        print("Generation ", iteration)
        newpop = []
        sorted_pop, sorted_losses = get_losses(population)
        print("loss = ",sorted_losses[0])
        if min_desired_loss is not None:
            if sorted_losses[0] <= min_desired_loss:
                return sorted_losses[0], population[0]
        # The top five always make it:
        newpop.append(sorted_pop[0]); newpop.append( sorted_pop[1]);newpop.append( sorted_pop[2])
        newpop.append( sorted_pop[3]);newpop.append( sorted_pop[4])
        while len(newpop) < initial_population_size:
                # select any from the top 10 of the population and randomly breed and mutate them
                # First crossover:
                idx1 = selectindex();idx2 = selectindex()
                if idx1 != idx2:
                    children, crossovered = crossover([population[idx1],population[idx2]], prob = crossover_prob)
                    if crossovered and len(newpop) < initial_population_size-1:
                        newpop.extend(children)
                # Mutation:
                idx1 = selectindex()
                child, mutated = mutate(population[idx1], prob = mutation_prob)
                if mutated and len(newpop) < initial_population_size:
                    newpop.append(child)
                # Crossover 2:
                idx1 = selectindex();idx2 = selectindex()
                if idx1 != idx2:
                    children, crossovered = crossover2([population[idx1],population[idx2]], prob = crossover2_prob)
                    if crossovered and len(newpop) < initial_population_size-1:
                        newpop.extend(children)
                #add a random new chromosome by the probability of none of the above hapening
                prob_none =1- ((crossover_prob*(1-mutation_prob)*(1-crossover2_prob) + (1-crossover_prob)*(mutation_prob)*(1-crossover2_prob)+(1-crossover_prob)*(1-mutation_prob)*(crossover2_prob))
                           +(crossover_prob*mutation_prob*(1-crossover2_prob) + (1-crossover_prob)*mutation_prob*crossover2_prob + crossover_prob*(1-mutation_prob)*crossover2_prob)
                           +crossover_prob*mutation_prob*crossover2_prob )
                if random() < prob_none and len(newpop) < initial_population_size:
                    newpop.append(generate_random_chromosome())
        population = list(np.copy(newpop))
    sorted_pop, sorted_losses = get_losses(population)
    return sorted_losses[0], sorted_pop[0]



initialize_population()
loss, weights = evolve(population ,max_iter = 1000,min_desired_loss = 0.2000, crossover_prob = 0.7, mutation_prob = 0.2, crossover2_prob = 0.2)
weights = get_weights_from_encoded(weights)

#Checking accuracy by plotting
model = model(weights)
#Training accuracy:
y_pred_train = model.forward_propagate(X_train)
for i in range(y_pred_train.shape[1]):
    if y_pred_train[0][i] >= 0.5:
        y_pred_train[0][i] = 1
    else:
        y_pred_train[0][i] = 0
        
y_pred_train = y_pred_train.reshape((320,))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred_train)
accuracy_train = (cm[0][0]+cm[1][1])/np.sum(cm)
print("training accuracy = ",accuracy_train)

#Test accuracy:
y_pred_test = model.forward_propagate(X_test)
for i in range(y_pred_test.shape[1]):
    if y_pred_test[0][i] >= 0.5:
        y_pred_test[0][i] = 1
    else:
        y_pred_test[0][i] = 0
        
y_pred_test = y_pred_test.reshape((80,))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_test)
accuracy_test = (cm[0][0]+cm[1][1])/np.sum(cm)
print("test accuracy = ",accuracy_test)

"""The training accuracy comes out to be 90.24 %. 
    The test accuracy is 91.25%. 
    So no overfitting here. """

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.forward_propagate(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Genetic Neural net(Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.forward_propagate(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Genetic Neural net(Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
