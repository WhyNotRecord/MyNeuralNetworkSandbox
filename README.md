# MyNeuralNetworkSandbox
My playground for experimenting with neural networks

Here I have learnt the basics of Recurrent Neural Networks (LSTM) and fine tuned MLPack instruments for user-friendy experience.
Here's the example output of my beautiful callback for training RNN:

Optimizer params: 100 epochs, 0.0005 learning rate, 32 batchsize.
Model created: 2 layers, 64 neurons, 0 dropout.
LSTMType -> LSTMType -> LeakyReLUType -> LinearType -> Out

---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0
☼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▲▼▼▼▼▲▲▼▼▼▲▼▲▲►▲►▲►▼▲►▼▼▼▼▲▼▼▼▼▼▼▼▼▼▼▲▼▼▼▼▲▼▲▼▲▼▲►►▼▼▼▼▲►▼▲▲▲▼

First 10 loss values skipped are: 1.34271 1.20875 0.59418 0.18246 0.11399 0.08533 0.06764 0.05553 0.04577 0.03831
     0.0324 ┼┐
     0.0309 ┤│
     0.0294 ┤│
     0.0280 ┤└┐
     0.0265 ┤ │
     0.0250 ┤ │
     0.0235 ┤ └┐
     0.0220 ┤  │
     0.0205 ┤  └┐
     0.0191 ┤   │
     0.0176 ┤   └┐
     0.0161 ┤    │
     0.0146 ┤    └┐
     0.0131 ┤     └┐
     0.0116 ┤      │
     0.0101 ┤      └┐
     0.0087 ┤       └┐
     0.0072 ┤        └─┐
     0.0057 ┤          └──┐
     0.0042 ┤             └───────────────┐
     0.0027 ┤                             └───────────────────────────────────────────────────────────

Final loss: 0.00272163

Mean Squared Error on prediction data points in training set := 0.156425
Deviation in percents (min, avg, max): 0.000002 6.783822 82.083452

Mean Squared Error on prediction data points in test set := 0.104036
Deviation in percents (min, avg, max): 0.000025 4.511276 53.290589
