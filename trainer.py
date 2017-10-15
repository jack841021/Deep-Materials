import json
from numpy import array
from numpy.random import choice
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import operator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def splitter(gener, kpoint):
    with open(str(gener) + '/saved_features_' + str(kpoint), 'r') as saved_features:
        features = json.load(saved_features)
    with open(str(gener) + '/saved_eigens_' + str(kpoint), 'r') as saved_eigens:
        eigens = json.load(saved_eigens)
    #--------------------------------------------------
    valid_choice = set(choice(len(features), size=int(len(features) * 0.01), replace=False))
    train_choice = set(i for i in range(len(features))) - valid_choice
    #--------------------------------------------------#
    train_features = []
    train_eigens = []
    for i in train_choice:
        train_features += [features[i]]
        train_eigens += [eigens[i]]
    valid_features = []
    valid_eigens = []
    for i in valid_choice:
        valid_features += [features[i]]
        valid_eigens += [eigens[i]]
    #--------------------------------------------------#
    with open(str(gener) + '/saved_train_features_' + str(kpoint), 'w') as saved_train_features:
        json.dump(train_features, saved_train_features)
    with open(str(gener) + '/saved_train_eigens_' + str(kpoint), 'w') as saved_train_eigens:
        json.dump(train_eigens, saved_train_eigens)
    with open(str(gener) + '/saved_valid_features_' + str(kpoint), 'w') as saved_valid_features:
        json.dump(valid_features, saved_valid_features)
    with open(str(gener) + '/saved_valid_eigens_' + str(kpoint), 'w') as saved_valid_eigens:
        json.dump(valid_eigens, saved_valid_eigens)
    # print('KPOINT', str(kpoint) + ':', len(train_features), 'training data and', len(valid_features), 'validation data have been created.')

def opener(gener, kpoint):
    with open(str(gener) + '/saved_train_features_' + str(kpoint), 'r') as saved_train_features:
        train_inputs = array(json.load(saved_train_features))
    with open(str(gener) + '/saved_train_eigens_' + str(kpoint), 'r') as saved_train_eigens:
        train_outputs = array(json.load(saved_train_eigens))
    with open(str(gener) + '/saved_valid_features_' + str(kpoint), 'r') as saved_valid_features:
        valid_inputs = array(json.load(saved_valid_features))
    with open(str(gener) + '/saved_valid_eigens_' + str(kpoint), 'r') as saved_valid_eigens:
        valid_outputs = array(json.load(saved_valid_eigens))
    return train_inputs, train_outputs, valid_inputs, valid_outputs

def builder(indim, drate, oudim):
    model = Sequential()
    model.add(Dense(700, activation = 'selu', input_dim = indim))
    model.add(Dropout(drate))
    model.add(Dense(700, activation = 'selu'))
    model.add(Dropout(drate))
    model.add(Dense(700, activation = 'selu'))
    model.add(Dropout(drate))
    model.add(Dense(700, activation = 'selu'))
    model.add(Dropout(drate))
    model.add(Dense(700, activation = 'selu'))
    model.add(Dropout(drate))
    model.add(Dense(700, activation = 'selu'))
    model.add(Dropout(drate))
    model.add(Dense(700, activation = 'selu'))
    model.add(Dropout(drate))
    model.add(Dense(oudim, activation = 'linear'))
    return model

def trainer(gener, start, end, drate, learn, batch, epoch, verbo, diffe):
    for kpoint in range(start, end):
        while True:
            splitter(gener, kpoint)
            train_inputs, train_outputs, valid_inputs, valid_outputs = opener(gener, kpoint)
            model = builder(len(train_inputs[0]), drate, len(train_outputs[0]))
            optimizer = SGD(lr = learn, momentum = 0.99, decay = 0, nesterov = True)
            model.compile(loss = 'mean_absolute_error', optimizer = optimizer)
            history = [[], []]
            lock = 0
            for i in range(int(epoch / 100)):
                hist = model.fit(x = train_inputs, y = train_outputs, batch_size = batch, epochs = 100, shuffle = True, validation_data = (valid_inputs, valid_outputs), verbose = verbo)
                thist, vhist = hist.history['loss'], hist.history['val_loss']
                tmean, vmean = sum(thist[90:100]) / 10, sum(vhist[90:100]) / 10
                history[0] += thist
                history[1] += vhist
                if abs(tmean - vmean) >= diffe:
                    print('KPOINT', str(kpoint) + ': Bad data.')
                    lock = 1
                    break
            if lock == 0:
                loc2 = 1
                for j in range(100):
                    hist = model.fit(x = train_inputs, y = train_outputs, batch_size = batch, epochs = 1, shuffle = True, validation_data = (valid_inputs, valid_outputs), verbose = verbo)
                    thist, vhist = hist.history['loss'], hist.history['val_loss']
                    tloss, vloss = thist[0], vhist[0]
                    history[0] += thist
                    history[1] += vhist
                    if abs(tloss - vloss) < 0.1:
                        model.save_weights(str(gener) + '/model_' + str(kpoint) + '.k')
                        with open(str(gener) + '/saved_history_' + str(kpoint), 'w') as saved_history:
                            json.dump(history, saved_history)
                        print('KPOINT', str(kpoint) + ':', tloss, 'and', vloss, 'are fine.\n')
                        loc2 = 0
                        break
                if loc2 == 0:
                    break

def tester(gener, kpoint, drate, learn, batch, epoch, verbo, diffe):
    while True:
        splitter(gener, kpoint)
        train_inputs, train_outputs, valid_inputs, valid_outputs = opener(gener, kpoint)
        model = builder(len(train_inputs[0]), drate, len(train_outputs[0]))
        optimizer = SGD(lr = learn, momentum = 0.99, decay = 0, nesterov = True)
        model.compile(loss = 'mean_absolute_error', optimizer = optimizer)
        lock = 0
        for i in range(int(epoch / 100)):
            history = model.fit(x = train_inputs, y = train_outputs, batch_size = batch, epochs = 100, shuffle = True, validation_data = (valid_inputs, valid_outputs), verbose = verbo)
            thist, vhist = history.history['loss'], history.history['val_loss']
            tmean, vmean = sum(thist[90:100]) / 10, sum(vhist[90:100]) / 10
            if abs(tmean - vmean) >= diffe:
                print('Bad data.')
                lock = 1
                break
        if lock == 0:
            print(epoch, diffe, tmean, vmean, '\n')
            return True

trainer(gener = 0, start = 10, end = 50, drate = 0.01, learn = 0.001, batch = 50, epoch = 700, verbo = 0, diffe = 0.2)
