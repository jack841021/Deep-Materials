import os
import json
import numpy
import operator
from keras.models import Sequential
from keras.layers import Dense
# import matplotlib.pyplot as pp
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

def common_finder(gener, kpoints):
    commons = {}
    for kpoint in range(kpoints):
        with open(str(gener) + '/saved_stands_' + str(kpoint), 'r') as saved_stands:
            stands = json.load(saved_stands)
        for stand in stands:
            if stand in commons.keys():
                commons[stand] += 1
            else:
                commons[stand] = 1
    with open(str(gener) + '/saved_stands_0', 'r') as saved_stands:
        stands = json.load(saved_stands)
    with open(str(gener) + '/saved_features_0', 'r') as saved_features:
        features = json.load(saved_features)
    new_stands = []
    new_features = []
    for stand in commons.keys():
        if commons[stand] == kpoints:
            new_stands += [stand]
            new_features += [features[stands.index(stand)]]
    with open(str(gener) + '/saved_common_stands', 'w') as saved_stands:
        json.dump(new_stands, saved_stands)
    with open(str(gener) + '/saved_common_features', 'w') as saved_features:
        json.dump(new_features , saved_features)
    for kpoint in range(kpoints):
        with open(str(gener) + '/saved_stands_' + str(kpoint), 'r') as saved_stands:
            stands = json.load(saved_stands)
        with open(str(gener) + '/saved_eigens_' + str(kpoint), 'r') as saved_eigens:
            eigens = json.load(saved_eigens)
        new_eigens = []
        for stand in new_stands:
            new_eigens += [eigens[stands.index(stand)]]
        with open(str(gener) + '/saved_common_eigens_' + str(kpoint), 'w') as saved_eigens:
            json.dump(new_eigens, saved_eigens)

def anomaly_finder(gener, kpoints, cutof, matid, error):
    with open(str(gener) + '/saved_common_stands', 'r') as saved_stands:
        stands = json.load(saved_stands)
    with open(str(gener) + '/saved_common_features', 'r') as saved_features:
        features = json.load(saved_features)
    errors = {}
    for stand in stands:
        errors[stand] = 0
    for kpoint in range(kpoints):
        with open(str(gener) + '/saved_common_eigens_' + str(kpoint), 'r') as saved_eigens:
            eigens = json.load(saved_eigens)
        model = builder(len(features[0]), len(eigens[0]))
        model.load_weights(str(gener) + '/model_' + str(kpoint) + '.k')
        predis = model.predict(features)
        for i in range(len(predis)):
            for j in range(len(eigens[i])):
                if eigens[i][j] == 50:
                    cut = j
                    break
            for j in range(cut):
                errors[stands[i]] += (abs(predis[i][j] - eigens[i][j])) / (cut * kpoints)
    anomalies = sorted(errors.items(), key=operator.itemgetter(1), reverse = True)
    if cutof != None:
        anomalies = anomalies[:cutof]
    if matid == True:
        for anomaly in anomalies:
            print(anomaly[0])
    if error == True:
        for anomaly in anomalies:
            print(anomaly[1])
    with open('saved_anomalies', 'w') as saved_anomalies:
        json.dump(anomalies, saved_anomalies)

def semi_stat(parts, incre, width, lenth):
    with open('saved_gaps', 'r') as saved_gaps:
        gaps = json.load(saved_gaps)
    with open('saved_anomalies', 'r') as saved_anomas:
        anomas = json.load(saved_anomas)
    candi = list(gaps.keys())
    chose = []
    other = []
    for anoma in anomas:
        if anoma[0] in candi:
            chose += [anoma[1]]
        else:
            other += [anoma[1]]
    pp.hist(other, bins = [width * i for i in range(int(lenth / width))])
    pp.hist(chose, bins = [width * i for i in range(int(lenth / width))])
    pp.plot([numpy.mean(other), numpy.mean(other)], [0, 750], linewidth = 2)
    pp.plot([numpy.mean(chose), numpy.mean(chose)], [0, 750], linewidth = 2)
    pp.show()
    means = []
    for i in range(parts):
        candi = []
        for key in gaps.keys():
            if gaps[key] >= incre * i and gaps[key] < incre * (i + 1):
                candi += [key]
        error = []
        for anoma in anomas:
            if anoma[0] in candi:
                error += [anoma[1]]
        means += [numpy.mean(error)]
    pp.plot([incre * i for i in range(len(means))], means)
    pp.show()

def line_plotter(gener, atoms):
    with open('saved_anomalies', 'r') as saved_anomalies:
        anomalies = json.load(saved_anomalies)
    etable = {}
    for anomaly in anomalies:
        etable[anomaly[0]] = anomaly[1]
    errors = {}    
    counte = {}
    for atom in prtable:
        errors[atom] = 0
        counte[atom] = 0
    with open(str(gener) + '/saved_common_stands', 'r') as saved_stands:
        stands = json.load(saved_stands)
    with open(str(gener) + '/saved_common_features', 'r') as saved_features:
        features = json.load(saved_features)
    for i in range(len(features)):
        feature = features[i][2:]
        for j in range(len(feature)):
            if feature[j] > 0:
                errors[prtable[j]] += etable[stands[i]]
                counte[prtable[j]] += 1
    for key in errors.keys():
        if errors[key] > 0:
            errors[key] /= counte[key]
    y = []
    a = []
    for atom in atoms:
        y += [errors[atom]]
        a += [counte[atom] / max(counte.values())]
    z = [i * j for i, j in zip(y, a)]
    x = [i for i in range(len(z))]
    pp.plot(x, y, linestyle = 'solid', marker = 'o')
    pp.plot(x, z, linestyle = 'solid', marker = 'o')
    for i in range(len(atoms)):
        pp.annotate(atoms[i], xy = (x[i], y[i] + 0.01))
    pp.show()

def pg_getter(element):
    periods = [2, 8, 8, 18, 18, 32, 32]
    a = prtable.index(element) + 1
    for i in range(len(periods)):
        if a <= periods[i]:
            p = i + 1
            g = a
            if element == 'He':
                g += 30
            elif g >= 3:
                if p == 2 or p == 3:
                    g += 24
                elif p == 4 or p == 5:
                    g += 14
            return p, g
        else:
            a -= periods[i]

def plane_plotter(gener, origin, empty):
    with open('saved_anomalies', 'r') as saved_anomalies:
        anomalies = json.load(saved_anomalies)
    etable = {}
    for anomaly in anomalies:
        etable[anomaly[0]] = anomaly[1]
    errors = {}    
    counte = {}
    for atom in prtable:
        errors[atom] = 0
        counte[atom] = 0
    with open(str(gener) + '/saved_common_stands', 'r') as saved_stands:
        stands = json.load(saved_stands)
    with open(str(gener) + '/saved_common_features', 'r') as saved_features:
        features = json.load(saved_features)
    for i in range(len(features)):
        feature = features[i][2:]
        for j in range(len(feature)):
            if feature[j] > 0:
                errors[prtable[j]] += etable[stands[i]]
                counte[prtable[j]] += 1
    for key in errors.keys():
        if errors[key] > 0:
            errors[key] /= counte[key]
    table = numpy.full((7, 32), empty)
    for atom in prtable:
        p, g = pg_getter(atom)
        i, j = p - 1, g - 1
        table[i][j] = errors[atom] * counte[atom] / max(counte.values())
        if origin == True:
            table[i][j] = counte[atom]
    pp.matshow(table, cmap = 'magma')
    pp.show()

def kpoint_stat(gener, kpoints):
    with open(str(gener) + '/saved_common_features', 'r') as saved_features:
        features = json.load(saved_features)
    mean = []
    stdv = []
    for kpoint in range(kpoints):
        with open(str(gener) + '/saved_common_eigens_' + str(kpoint), 'r') as saved_eigens:
            eigens = json.load(saved_eigens)
        model = builder(len(features[0]), len(eigens[0]))
        model.load_weights(str(gener) + '/model_' + str(kpoint) + '.k')
        predis = model.predict(features)
        error = []
        for i in range(len(predis)):
            diffe = 0
            for j in range(len(eigens[i])):
                if eigens[i][j] == 50:
                    cut = j
                    break
            for j in range(cut):
                diffe += (abs(predis[i][j] - eigens[i][j])) / cut
            error += [diffe]
        mean += [numpy.mean(error)]
        stdv += [numpy.std(error)]
    pp.plot([i for i in range(kpoints)], mean, linestyle = 'solid', marker = 'o')
    pp.plot([i for i in range(kpoints)], stdv, linestyle = 'solid', marker = 'o')
    pp.show()

def bands_cleaner(previ, gener, ids):
    with open('saved_trimmed_bands', 'r') as saved_trimmed_bands:
        bands = json.load(saved_trimmed_bands)
    new_bands = []
    for band in bands:
        if band[0] not in ids:
            new_bands += [band]
    with open(str(gener) + '/saved_trimmed_bands', 'w') as saved_trimmed_bands:
        json.dump(new_bands, saved_trimmed_bands)
    print(len(bands))
    print(len(new_bands))

common_finder(gener = 0, kpoints = 50)
# bands_cleaner(previ = 0, gener = 0, ids = ['mp-23026', 'mp-2534', 'mp-406'])

# atom_counter(gener = 0, atoms = ['H' , 'Li', 'Na', 'K' , 'Rb'])
# atom_counter(gener = 0, atoms = ['Be', 'Mg', 'Ca', 'Sr', 'Ba'])
# atom_counter(gener = 0, atoms = ['B' , 'Al', 'Ga', 'In', 'Tl'])
# atom_counter(gener = 0, atoms = ['C' , 'Si', 'Ge', 'Sn', 'Pb'])
# atom_counter(gener = 0, atoms = ['O' , 'S' , 'Se', 'Te', 'Po'])
# atom_counter(gener = 0, atoms = ['F' , 'Cl', 'Br', 'I' , 'At'])
