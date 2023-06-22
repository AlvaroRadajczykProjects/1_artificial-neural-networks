import numpy as np

import implementation

def Sigmoid(x):
	return 1 / (1 + np.exp(-x))

def d_Sigmoid(x):
	return Sigmoid(x) * (1 - Sigmoid(x))

def Tanh(x):
	return np.tanh(x)

def d_Tanh(x):
	return 1-np.tanh(x)**2

def pReLU(x):
	return max(0.01*x,x)

def d_pReLU(x):
	if x < 0: return 0.01
	else: return 1

def MSE(y_pred, y_exp):
	return (y_pred - y_exp)**2

def d_MSE(y_pred, y_exp):
	return 2*(y_pred - y_exp)

network = implementation.network_builder( 2, [3, 4], 4 )

fun_list = [ Tanh for i in range(len(network[0])) ]
dfun_list = [ d_Tanh for i in range(len(network[0])) ]

examples = [
	[ [ 0, 0 ], [1, 0, 0, 0] ],
	[ [ 0, 1 ], [0, 0, 1, 0] ],
	[ [ 1, 0 ], [0, 1, 0, 0] ],
	[ [ 1, 1 ], [0, 0, 0, 1] ]
]

epochs = 3000

"""
	we wont always obtain a good prediction. involved factors are:
		- learning rate
		- random values which initialized weight values
		- activation functions
		- cost/error function
		- ...
	however, the major part of program runs should give decent results almost with sigmoid function
	for example, using pReLU without modifying the weight random values causes less frequent good predictions. However, when networks learns ok, the error significantly reduces
	in addition, Tanh is slower computing, but average results are better than average sigmoid results
"""

for i in range(epochs):
	if i%(1000-1) == 0: implementation.back_propagation_one_epoch_softmax( examples, network, 0.01, 0.9, 0.99, 1e-8, fun_list, dfun_list, True )
	else: implementation.back_propagation_one_epoch_softmax( examples, network, 0.01, 0.9, 0.99, 1e-8, fun_list, dfun_list, False )

def parsePred(pred):
	ret = []
	for i in pred:
		if i >= 0.5: ret.append(1)
		else: ret.append(0)
	return ret

print( "prediction for [0, 0] is:", parsePred( implementation.make_prediction( [0, 0], network, fun_list ) ) )
print( "prediction for [0, 1] is:", parsePred( implementation.make_prediction( [0, 1], network, fun_list ) ) )
print( "prediction for [1, 0] is:", parsePred( implementation.make_prediction( [1, 0], network, fun_list ) ) )
print( "prediction for [1, 1] is:", parsePred( implementation.make_prediction( [1, 1], network, fun_list ) ) )

