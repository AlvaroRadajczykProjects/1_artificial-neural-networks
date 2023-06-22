from copy import deepcopy
from numpy import random
from numpy import log
#import random
import numpy as np
import math

import utils

"""
	Build an empty network with biases initialized to 0 and weights initialized to a random number (can be easily modified)
	
	input_layer -> number of input layer nodes
	hidden_layers -> python list with the number of each hidden layer nodes
	output_layer -> number of output layer nodes
	
	return => python list with two elements as lists, the first one contains all vectors representing the bias of each network layer (except input layer),
		  and the second one contains all matrixes representing the weights of each union of two network layers
"""
def network_builder( input_layer, hidden_layers , output_layer ):
	nodes = [input_layer] + hidden_layers + [output_layer]
	network = [ [], [] ]
	for i in range(len(nodes)-1):
		network[0].append([ 0 for j in range(nodes[i+1]) ])
		network[1].append(random.normal( loc=0, scale=1, size=( nodes[i], nodes[i+1] ) ).tolist()) #2/nodes[i+1]
		#network[1].append( utils.random_matrix(nodes[i], nodes[i+1], 3, -3) )
	return network

"""
	Build an empty gradient vector representation with the same shape of a network as argument (biases errors as network biases, and weights errors as network weights).
	All element values equal to 0
	
	network -> a network built with network_builder( input_layer, hidden_layers , output_layer ) function
	
	return => gradient vector with same shape as network
"""
def gradient_vector_builder( network ):
	vgrad = [ [], [] ]
	for i in range(len(network[0])):
		vgrad[0].append([ 0 for j in range(len(network[0][i])) ])
		vgrad[1].append([ [ 0 for k in range(len(network[1][i][j])) ] for j in range(len(network[1][i])) ])
	return vgrad

"""
	Build and empty ADAM vector representation with the same shape of a network as argument (biases errors as network biases, and weights errors as network weights).
	All element values equal to 0. The difference between gradient_vector_builder is there's a respective momentum and velocity copy for each weight matrix or bias vector
	instead of one.
	
	Momentum = [
		[] -> Momentum_weights
		[] -> Momentum_biases	
	],
	Velocity = [
		[] -> Velocity_weights
		[] -> Velocity_biases	
	]
	
	network -> a network built with network_builder( input_layer, hidden_layers , output_layer ) function
	
	return => initialized ADAM vector representation
"""
def adam_vector_builder( network ):
	#Momentum, Velocity -> [[], []]
	#Inside momentum and velocity, first one stores x_weights (matrices), and second stores x_biases (vectors), where x is Momentum xor Velocity
	vadam = [ [[], []], [[], []] ]
	for i in range(len(network[0])):
		#biases
		vadam[0][1].append([ 0 for j in range(len(network[0][i])) ])
		vadam[1][1].append([ 0 for j in range(len(network[0][i])) ])
		#weights
		vadam[0][0].append([ [ 0 for k in range(len(network[1][i][j])) ] for j in range(len(network[1][i])) ])
		vadam[1][0].append([ [ 0 for k in range(len(network[1][i][j])) ] for j in range(len(network[1][i])) ])
	return vadam

"""
	Returns the results of network forward propagacion. In this case, last activation function is softmax instead another one passed as argument
	
	input_list -> python list with network input values 
	network -> a network built with network_builder( input_layer, hidden_layers , output_layer ) function
	fun_list -> python list with unary fuction of activation functions desired to use (for example, def Sigmoid(x): return 1 / (1 + np.exp(-x)) )
	
	return =>
		- zl: python list of lists, which each element is a python list with the sum of the output value multiplyed by the weight of the edge of the nodes on previous layer, for each 
		      node in the current layer (except input layer)
		- al: same zl list but applying the corresponding activation function for each element in zl (except input layer)
"""
def forward_propagation_softmax( input_list, network, fun_list ):
	al = [ input_list ]
	zl = []
	for i in range(len(network[0])):
		zl.append( utils.matrix_product([al[i]],network[1][i])[0] )
		utils.vectors_add( zl[i], network[0][i], 1 )
		if i < ( len(network[0]) - 1 ): 
			al.append( [zl[i][j] for j in range(len(zl[i])) ] )
			utils.apply_function_each_element_vector( al[i+1], fun_list[i] )
	#Here we apply softmax
	sumat = sum( np.exp(i) for i in zl[-1] )
	al.append( [ np.exp(i)/sumat for i in zl[-1] ] )
	return al[1:], zl

"""
	Returns a list with the bias errors for each neuron in output layer when using softmax as activation function and cross entrophy as loss function of this layer
	
	exp_output_list -> a list with expected output list values
	output_layer_al -> a list with output values obtained in forward propagation, whose output layer was applied softmax function
	
	return => a list with the bias errors for each neuron in output layer
"""
def obtainErrorBiasOutputLayerSoftmax( exp_output_list, output_layer_al ):
	biases = []
	for i in range(len(exp_output_list)):
		if exp_output_list[i] == 1: biases.append( output_layer_al[i] - 1 )
		else: biases.append( output_layer_al[i] )
	return biases

"""
	Returns the value of the cost function of cross entrophy when activation function in output layer is Softmax
	
	exp_output_list -> a list with expected output list values
	output_layer_al -> a list with output values obtained in forward propagation, whose output layer was applied softmax function
	
	return => the value of the cost function
"""
def obtainCostFunctionValueSoftmax( exp_output_list, output_layer_al ):
	res = 0
	for i in range(len(exp_output_list)): res += -1*exp_output_list[i]*log( output_layer_al[i] )
	return res

"""
	Returns a gradient vector like the one built with def network_builder( input_layer, hidden_layers , output_layer ) but with error values for biases and weights instead of
	zeroes. In this case, last activation function is softmax instead another one passed as argument, so INPUT AND OUTPUT VALUES SHOULD BE 0 OR 1, and there should be as number
	of output elements as number of expected classes 
	
	examples -> python list that contains all training examples, each one as a python list with two elements, the first one represents the input values, and the second one the expected
		    output values (for example, [ [ [a, b], [c] ], [ [d, e], [f] ] ] contains two training examples, the first one has a,b as inputs and c as output, and second one has
		    d,e as inputs and f as output ) 
	network -> a network built with network_builder( input_layer, hidden_layers , output_layer ) function
	fun_list -> python list with unary fuction of activation functions desired to use (for example, def Sigmoid(x): return 1 / (1 + np.exp(-x)) )
	dfun_list -> python list with unary function of derivate activation functions desired to use (for example, def d_Sigmoid(x): return Sigmoid(x) * (1 - Sigmoid(x)) )
    
    	return => a gradient vector with same shape as the one built in def gradient_vector_builder( network ) but with bias and weight error values instead of zeroes
"""
def calculated_gradient_vector_softmax( exp_output_list, al_list, zl_list, network, dfun_list ):
	#the gradient vector this function will return
	ret = [ [], [] ]
	#define wl_list for better comprehensionn
	wl_list = network[1]
	#calculate the bias error in the output layer
	error_bias = obtainErrorBiasOutputLayerSoftmax( exp_output_list, al_list[-1] )
	#back propagation process
	for L in range(len(al_list)-1,0,-1):
		#save bias errors for current layer
		ret[0].insert(0, [i for i in error_bias])
		#calculate and save weight errors for current layer
		ret[1].insert(0, utils.matrix_product( utils.matrix_traspose([al_list[L-1]]), [error_bias] ) )
		#if we are in the last layer
		if L > 1: 
			#here, wl_list index is L-1 instead of L like theory, but refers to the same weight matrix in L layer
			x = utils.matrix_product( [error_bias], utils.matrix_traspose(wl_list[L-1]) )[0]
			#calculate bias errors for previous layer. dfun_list index is L-2, but refers to the same theorical L-1 layer
			error_bias = [ x[i] * dfun_list[L-2](zl_list[L-1][i]) for i in range(len(x)) ]
	return ret

def updateAdamVectorValues( vadam, vgrad, b1=0.9, b2=0.99 ):
	for m in range(len( vgrad[1] )):
		for r in range(len( vgrad[1][m] )):
        		for c in range(len( vgrad[1][m][r] )):
        			#momentum
            			vadam[0][0][m][r][c] = b1 * vadam[0][0][m][r][c] + (1 - b1) * vgrad[1][m][r][c]
            			#velocity
            			vadam[1][0][m][r][c] = b2 * vadam[1][0][m][r][c] + (1 - b2) * vgrad[1][m][r][c] * vgrad[1][m][r][c]
	for v in range(len( vgrad[0] )):
		for e in range(len( vgrad[0][v] )):
			#momentum
			vadam[0][1][v][e] = b1 * vadam[0][1][v][e] + (1 - b1) * vgrad[0][v][e]
			#velocity
			vadam[1][1][v][e] = b2 * vadam[1][1][v][e] + (1 - b2) * vgrad[0][v][e] * vgrad[0][v][e]

"""
	Adjust ADAM momentum and velocity and return the error value
	
	momentum -> momentum value without adjust
	velocity -> velocity value without adjust
	
	return => error value
"""
def adjustAdamValue( momentum, velocity, b1=0.9, b2=0.99, learning_rate=0.01, epsilon=1e-8 ):
	fit_momentum = momentum / (1 - b1)
	fit_velocity = velocity / (1 - b2)
	return (learning_rate * fit_momentum) / ( epsilon + math.sqrt( fit_velocity ) )

"""
	Makes network learn with ADAM vector
"""
def learnNetworkAdamVector( network, vadam, b1=0.9, b2=0.99, learning_rate=0.01, epsilon=1e-8 ):
	for m in range(len( network[1] )):
    		for r in range(len( network[1][m] )):
        		for c in range(len( network[1][m][r] )):
            			network[1][m][r][c] -= adjustAdamValue( vadam[0][0][m][r][c], vadam[1][0][m][r][c], b1, b2, learning_rate, epsilon )
	for v in range(len( network[0] )):
    		for e in range(len( network[0][v] )):
    			network[0][v][e] -= adjustAdamValue( vadam[0][1][v][e], vadam[1][1][v][e], b1, b2, learning_rate, epsilon )

"""
	Computes theorical back propagation to the network from a set of examples. Prints on console the estimated error obtained by
	the desired cost function as argument. In this case, last activation function is softmax instead another one passed as argument,
	so INPUT VALUES SHOULD BE 0 OR 1
	
	examples -> python list that contains all training examples, each one as a python list with two elements, the first one represents the input values, and the second one the expected
		    output values (for example, [ [ [a, b], [c] ], [ [d, e], [f] ] ] contains two training examples, the first one has a,b as inputs and c as output, and second one has
		    d,e as inputs and f as output ) 
	network -> a network built with network_builder( input_layer, hidden_layers , output_layer ) function
	learning_rate -> number that represents the fixed learning rate desired to use
	fun_list -> python list with unary fuction of activation functions desired to use (for example, def Sigmoid(x): return 1 / (1 + np.exp(-x)) )
	dfun_list -> python list with unary function of derivate activation functions desired to use (for example, def d_Sigmoid(x): return Sigmoid(x) * (1 - Sigmoid(x)) )
	err_fun -> python binary function of error function which first parameter represents the network output value, and the second the expected output value
		   (for example def MSE(y_pred, y_exp): return (y_pred - y_exp)**2)
	derr_fun -> python binary function of derivate error function which first parameter represents the network output value, and the second the expected output value
		    (for example def d_MSE(y_pred, y_exp): return 2*(y_pred - y_exp))
        show_error -> boolean, if true, shows error in this iteration, otherwhise, don't
"""
def back_propagation_one_epoch_softmax( examples, network, learning_rate, b1, b2, epsilon, fun_list, dfun_list, show_error ):
	#randomly sort all examples
	random.shuffle(examples)
	#where errors will be stored
	error = []
	#store the vgrad as mean of rest of vgrads obtained for each example
	vgrad = gradient_vector_builder( network )
	#store all the ADAM optimization values in a built ADAM vector
	vadam = adam_vector_builder( network )
	for i in examples:
		x = i[0] #desired input
		y = i[1] #expected output
		#forward results
		al_list, zl_list = forward_propagation_softmax( x, network, fun_list )
		al_list.insert(0,x)
		zl_list.insert(0,x)
		#calculate error for this example and iteration
		error.append( obtainCostFunctionValueSoftmax( y, al_list[-1] ) )
		#obtain the gradient vector
		calc = calculated_gradient_vector_softmax( y, al_list, zl_list, network, deepcopy(dfun_list) )
		#multiply each error by learning_rate
		for j in range(len(vgrad[0])): utils.vectors_add(vgrad[0][j], calc[0][j], 1)
		for j in range(len(vgrad[1])): utils.matrixes_add(vgrad[1][j], calc[1][j], 1)	
	#apply learning rate divided by number of examples
	for j in range(len(vgrad[0])): utils.vector_mul(vgrad[0][j],1/len(examples))
	for j in range(len(vgrad[1])): utils.matrix_mul(vgrad[1][j],1/len(examples))
	#update the ADAM vector
	updateAdamVectorValues( vadam, vgrad, 0.9, 0.99 )
	#sub ADAM value obtained with momentum and velocity for each weight error
	learnNetworkAdamVector( network, vadam, b1, b2, learning_rate, epsilon )
	#show error if requested
	if show_error: print( "Softmax and cross entrophy error:", sum(error) )

"""
	Make a prediction in a network and returns it as a python list
	
	input_values -> python list with input values desired to use for making a prediction
	network -> a network built with network_builder( input_layer, hidden_layers , output_layer ) function
	fun_list -> python list with unary fuction of activation functions desired to use (for example, def Sigmoid(x): return 1 / (1 + np.exp(-x)) )
	
	return => prediction as a python list of values
"""
def make_prediction( input_values, network, fun_list ):
	al_list, zl_list = forward_propagation_softmax( input_values, network, fun_list )
	return al_list[-1]
