import numpy as np
import random
import pickle
import gzip
import os
import urllib.request

def sigma(z):
    return 1.0/(1.0+np.exp(-z))

def sigma_der(z):
    return sigma(z) *(1-sigma(z))

class network:
    def __init__(self,sizes):
        self.sizes=sizes
        self.num_layers =len(sizes)
        self.biases=[np.random.randn(y,1) for y in sizes[1:]]
        self.weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

    def feedforward(self,a):
        for b,w in zip(self.biases,self.weights):
            a=sigma(np.dot(w,a)+b)
        return a

    def backprop(self,x,y):
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        activation=x
        activations=[x]
        zs=[]
        for b,w in zip(self.biases, self.weights):
            z=np.dot(w,activation)+b
            zs.append(z)
            activation=sigma(z)
            activations.append(activation)
        delta=(activations[-1]-y)*sigma_der(zs[-1])
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta,activations[-2].transpose())
        for l in range(2,self.num_layers):
            z=zs[-l]
            sp=sigma_der(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l]=delta
            nabla_w[-l]=np.dot(delta,activations[-l-1].transpose())
        return(nabla_b,nabla_w)

    def update_mini_batch(self,mini_batch,eta):
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w=self.backprop(x,y)
            nabla_b=[nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w=[nw+dnw for nw, dnw in zip(nabla_w,delta_nabla_w)]
        self.weights=[w-(eta/len(mini_batch))*nw
                         for w,nw in zip(self.weights,nabla_w)]
        self.biases=[b-(eta/len(mini_batch))*nb
                         for b,nb in zip(self.biases,nabla_b)]

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def SGD(self,training_data, epochs, mini_batch_size,eta,test_data=None):
        if test_data: n_test = len(test_data)
        n=len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches=[
                training_data[k:k+mini_batch_size]
                for k in range(0,n,mini_batch_size)
                ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")
from PIL import Image

def incarca_imaginea_mea(nume_fisier):
    # Deschidem imaginea și o convertim la grayscale (L)
    img = Image.open(nume_fisier).convert('L')
    # O transformăm într-un array de numere
    data = np.array(img)
    # MNIST are valori între 0 și 1, deci împărțim la 255
    # Și facem "flatten" ca să avem 784 de pixeli
    data = data.reshape(784, 1) / 255.0
    return data

def load_data():
    filename = "mnist.pkl.gz"
    if not os.path.exists(filename):
        print("Downloading MNIST...")
        url = "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz"
        urllib.request.urlretrieve(url, filename)
    with gzip.open(filename, 'rb') as f:
        tr_d, va_d, te_d = pickle.load(f, encoding='latin1')
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

if __name__ == "__main__":
    training_data, test_data = load_data()
    net = network([784, 30, 10])
    

    net.SGD(training_data, 10, 10, 3.0, test_data=test_data)
    
    print("\n--- TESTARE IMAGINE PROPRIE ---")
    try:
        imagine_mea = incarca_imaginea_mea("sase.png")
        rezultat = net.feedforward(imagine_mea)
        cifra_ghicita = np.argmax(rezultat)
        
        print(f"Rețeaua crede că ai desenat cifra: {cifra_ghicita}")
        print(f"Probabilitate: {rezultat[cifra_ghicita][0]*100:.2f}%")
    except FileNotFoundError:
        print("Eroare: Nu am găsit fișierul cifra.png in folder!")