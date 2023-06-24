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

network = implementation.network_builder( 2, [10,10,10], 4 )

fun_list = [ pReLU for i in range(len(network[0])) ]
dfun_list = [ d_pReLU for i in range(len(network[0])) ]

examples = [
	[ [ 0, 0 ], [1, 0, 0, 0] ],
	[ [ 0, 1 ], [0, 0, 1, 0] ],
	[ [ 1, 0 ], [0, 1, 0, 0] ],
	[ [ 1, 1 ], [0, 0, 0, 1] ]
]

max_epochs = 3000

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

#se producen menos errores, pero a veces incluso se queda quieto en un mal mínimo local! aumentando las neuronas en la capa de salida reduce las posibilidades de que eso pase

perror = 100
i = 0
seguir = True

while seguir:
	i+=1
	if i%1000==0: 
		cerror = implementation.back_propagation_one_epoch_softmax( examples, network, 0.001, 0.9, 0.99, 1e-8, fun_list, dfun_list, True )
		print( "Aumenta el error?", cerror > perror )
		if cerror > perror or i > max_epochs: seguir = False
		perror = cerror
	else: implementation.back_propagation_one_epoch_softmax( examples, network, 0.001, 0.9, 0.99, 1e-8, fun_list, dfun_list, False )

"""for i in range(epochs):
	if i%(1000-1) == 0: implementation.back_propagation_one_epoch_softmax( examples, network, 0.01, 0.9, 0.99, 1e-8, fun_list, dfun_list, True )
	else: implementation.back_propagation_one_epoch_softmax( examples, network, 0.01, 0.9, 0.99, 1e-8, fun_list, dfun_list, False )"""

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

