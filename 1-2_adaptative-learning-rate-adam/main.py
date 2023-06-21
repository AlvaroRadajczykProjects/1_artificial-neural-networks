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

network = implementation.network_builder( 2, [4, 3], 1 )

fun_list = [ Tanh for i in range(len(network[0])) ]
dfun_list = [ d_Tanh for i in range(len(network[0])) ]

examples = [
	[ [ 0, 0 ], [1] ],
	[ [ 0, 1 ], [0] ],
	[ [ 1, 0 ], [0] ],
	[ [ 1, 1 ], [1] ]
]

epochs = 200000

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

#suele fallar menos. Lo que si, con una tasa de aprendizaje bastante más baja, es capaz de aprender mejor, más rápido
#si la tasa de aprendizaje es muy baja y hay suficientes épocas, a veces es capaz de recuperarse de algunos mínimos locales malos

error = 100
while error > 1e-3:
	for i in range(1000):
		if i==999: 
			error = implementation.theorical_back_propagation_adam( examples, network, 0.001, 0.9, 0.99, 1e-8, fun_list, dfun_list, MSE, d_MSE, True )
		else: implementation.theorical_back_propagation_adam( examples, network, 0.001, 0.9, 0.99, 1e-8, fun_list, dfun_list, MSE, d_MSE, False )

print( "prediction for [0, 0] is:", implementation.make_prediction( [0, 0], network, fun_list )[0] > 0.5 )
print( "prediction for [0, 1] is:", implementation.make_prediction( [0, 1], network, fun_list )[0] > 0.5 )
print( "prediction for [1, 0] is:", implementation.make_prediction( [1, 0], network, fun_list )[0] > 0.5 )
print( "prediction for [1, 1] is:", implementation.make_prediction( [1, 1], network, fun_list )[0] > 0.5 )
