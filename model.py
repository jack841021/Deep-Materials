import numpy, json
from keras.models import Sequential
from keras.layers import Dense
import plotly.plotly as plotly
import plotly.graph_objs as go

with open('saved_train_features', 'r') as saved_train_features:
    train_inputs = numpy.array(json.load(saved_train_features))
with open('saved_train_bands', 'r') as saved_train_bands:
    train_outputs = numpy.array(json.load(saved_train_bands))
with open('saved_valid_features', 'r') as saved_valid_features:
    valid_inputs = numpy.array(json.load(saved_valid_features))
with open('saved_valid_bands', 'r') as saved_valid_bands:
    valid_outputs = numpy.array(json.load(saved_valid_bands))
with open('saved_evalu_features', 'r') as saved_evalu_features:
    evalu_inputs = numpy.array(json.load(saved_evalu_features))
with open('saved_evalu_bands', 'r') as saved_evalu_bands:
    evalu_outputs = numpy.array(json.load(saved_evalu_bands))

model = Sequential()
model.add(Dense(input_dim =  207, output_dim =  400, activation =   'linear'))
model.add(Dense(                  output_dim =  600, activation = 'softplus'))
model.add(Dense(                  output_dim =  800, activation = 'softplus'))
model.add(Dense(                  output_dim = 1000, activation = 'softplus'))
model.add(Dense(                  output_dim = 1200, activation = 'softplus'))
model.add(Dense(                  output_dim = 1400, activation = 'softplus'))
model.add(Dense(                  output_dim = 1600, activation = 'softplus'))
model.add(Dense(                  output_dim = 1400, activation = 'softplus'))
model.add(Dense(                  output_dim = 1200, activation = 'softplus'))
model.add(Dense(                  output_dim = 1000, activation =   'linear'))

from keras.optimizers import SGD
optimizer = SGD(lr = 0.001, momentum = 1, decay = 0, nesterov = False)
model.compile(loss = 'mean_absolute_error', optimizer = optimizer)

history = model.fit(x = train_inputs, y = train_outputs, batch_size = 100, nb_epoch = 40, shuffle = True, validation_data = (valid_inputs, valid_outputs), verbose = 1)

# loss = history.history['loss']
# vali = history.history['val_loss']
# x = [i for i in range(70)]
# y_loss = go.Scatter(x = x, y = loss)
# y_vali = go.Scatter(x = x, y = vali)
# trace = [loss, vali]
# figure = dict(data = trace)
# py.iplot(figure, filename = 'loss')

y = model.predict(evalu_inputs[0:1])[0]
x = [i * 0.1 for i in range(100)]
z = []
for j in range(10):
    z += [y[j * 100 : (j + 1) * 100]]
traces = []
for y_ in z:
    traces += [go.Scatter(x = x, y = y_, line = dict(shape = 'spline'))]
figure = dict(data = traces)
plotly.iplot(figure, filename = '40-1')