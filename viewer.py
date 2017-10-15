import os
import json
import numpy
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as pp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

prtable = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu']

def builder(indim, oudim):
    model = Sequential()
    model.add(Dense(700, activation = 'selu', input_dim = indim))
    model.add(Dense(700, activation = 'selu'))
    model.add(Dense(700, activation = 'selu'))
    model.add(Dense(700, activation = 'selu'))
    model.add(Dense(700, activation = 'selu'))
    model.add(Dense(700, activation = 'selu'))
    model.add(Dense(700, activation = 'selu'))
    model.add(Dense(oudim, activation = 'linear'))
    return model

def loader(gener, kpoints, indim, oudim):
    models = []
    for kpoint in range(kpoints):
        model = builder(indim, oudim)
        model.load_weights(str(gener) + '/model_' + str(kpoint) + '.k')
        models += [model]
    return models

def transformer(arrays):
    trans = [[] for i in range(len(arrays[0]))]
    for array in arrays:
        for i in range(len(array)):
            trans[i] += [array[i]]
    return trans

def plotter(feature, models, degre, nband, dft):
    eigens = []
    for model in models:
        eigens += list(model.predict(numpy.array([feature])))
    kpoints = len(eigens)
    pp.figure(figsize=(2, 10))
    bands = transformer(eigens)
    def f(x, cs):
        y = 0
        for i in range(len(cs)):
            c = cs[i]
            y += c * x ** i
        return y
    for band in bands[nband[0]:nband[1]]:
        xs = [1 / (kpoints - 1) * i for i in range(kpoints)]
        pp.scatter(xs, band, s = 8)
        cs = list(numpy.polyfit(x = xs, y = band, deg = degre))
        cs.reverse()
        pp.plot(xs, [f(x, cs) for x in xs], lw = 2)
    if dft != None:
        for band in dft[nband[0]:nband[1]]:
            cs = list(numpy.polyfit(x = xs, y = band, deg = degre))
            cs.reverse()
            pp.plot(xs, [f(x, cs) for x in xs], '--', lw = 2)
    pp.show()

def viewer(gener, kpoints):
    with open(str(gener) + '/saved_common_features', 'r') as saved_features:
        features = json.load(saved_features)
    with open(str(gener) + '/saved_common_stands', 'r') as saved_stands:
        stands = json.load(saved_stands)
    with open('saved_trimmed_bands', 'r') as saved_bands:
        bands = json.load(saved_bands)
    models = loader(gener, kpoints, 96, 100)
    while True:
        angle = float(input('angle: '))
        lenth = float(input('lenth: '))
        kinds = input('kinds: ').split(' ')
        numbs = input('numbs: ').split(' ')
        nband = [int(i) for i in input('nband: ').split(' ')]
        degre = float(input('degre: '))
        matid = input('matid: ')
        atoms = [0 for i in range(len(prtable))]
        for i in range(len(kinds)):
            kind = kinds[i]
            atoms[prtable.index(kind)] = int(numbs[i])
        feature = [angle * 0.0174533, lenth] + atoms
        dft = None
        for band in bands:
            if band[0] == matid:
                dft = band[1:]
                break
        plotter(feature, models, degre, nband, dft)

viewer(gener = 0, kpoints = 50)