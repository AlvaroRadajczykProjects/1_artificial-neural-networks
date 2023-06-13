#from numpy import random
import random

"""
	Returns the product of two matrixes as matrix1*matrix2
	
	matrix1 -> matrix represented as a python list of lists
	matrix2 -> matrix represented as a python list of lists
	
	return => matrix represented as a python list of lists
"""
def matrix_product( matrix1, matrix2 ):
	res = [[0 for x in range(len(matrix2[0]))] for y in range(len(matrix1))]
	for i in range(len(matrix1)):
		for j in range(len(matrix2[0])):
			for k in range(len(matrix2)):
				res[i][j] += matrix1[i][k] * matrix2[k][j]
	return res

"""
	Returns the transposed matrix of m matrix
	
	m -> matrix represented as a python list of lists
	
	return => matrix represented as a python list of lists
"""
def matrix_traspose(m):
	ret = [ [ 0 for j in range(len(m)) ] for i in range(len(m[0])) ]
	for i in range(len(m)):
		for j in range(len(m[0])):
			ret[j][i] = m[i][j]
	return ret

"""
	Add vector v2 with vector v1, multiplying each v2 element by a factor
	
	v1 -> vector represented as a python list
	v2 -> vector represented as a python list
	factor -> a desired factor to multiply each element of v2
	
	return => vector represented as a python list	
"""
def vectors_add(v1,v2,factor):
	for i in range(len(v1)):
		v1[i] += factor*v2[i]

"""
	Add matrix m2 with matrix m1, multiplying each m2 element by a factor
	
	v1 -> matrix represented as a python list of lists
	v2 -> matrix represented as a python list of lists
	factor -> a desired factor to multiply each element of v2
	
	return => matrix represented as a python list of lists
"""
def matrixes_add(m1,m2,factor):
	for i in range(len(m1)):
		for j in range(len(m1[i])):
			m1[i][j] += factor*m2[i][j]

"""
	Multiply each element of v vector by a factor
"""
def vector_mul(v,factor):
	for i in range(len(v)):
		v[i] *= factor

"""
	Multiply each element of m matrix by a factor
"""
def matrix_mul(m,factor):
	for i in range(len(m)):
		for j in range(len(m[i])):
			m[i][j] *= factor

"""
	Apply a function to each element of v vector
	
	v -> vector represented as a python list
	function -> unary python function
"""
def apply_function_each_element_vector(v,function):
	for i in range(len(v)):
		v[i] = function(v[i])

"""
	Apply a function to each element of m matrix
	
	m -> matrix represented as a python list of lists
	function -> unary python function
"""
def apply_function_each_element_matrix(m,function):
	for i in range(len(m)):
		for j in range(len(m[i])):
			m[i][j] = function(m[i][j])

"""
	Returns a matrix with desired rows and cols, which each element is a random number
	between maxran and minran (included)
"""
def random_matrix(rows,cols,maxran,minran):
	return [ [ random.uniform(maxran, minran) for j in range(cols) ] for i in range(rows) ]
