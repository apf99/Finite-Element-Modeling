import numpy as np 
from scipy.linalg import eigh
import math
from matplotlib import pyplot as plt 
import time

def bar2(num_elems):
	restrained_dofs = [0, ]

	# element mass and stiffnessm matrices
	m = np.array([[2,1],[1,2]]) / (6.0 * num_elems)
	k = np.array([[1,-1],[-1,1]]) * float(num_elems)

	# construct global mass and stiffness matrices
	M = np.zeros((num_elems+1,num_elems+1))
	K = np.zeros((num_elems+1,num_elems+1))

	# for each element, change to global coordinates
	for i in range(num_elems):
		M_temp = np.zeros((num_elems+1,num_elems+1))
		K_temp = np.zeros((num_elems+1,num_elems+1))
		M_temp[i:i+2, i:i+2] = m
		K_temp[i:i+2, i:i+2] = k
		M += M_temp
		K += K_temp

	# remove the fixed degrees of freedom
	for dof in restrained_dofs:
		for i in [0,1]:
			M = np.delete(M, dof, axis=i)
			K = np.delete(K, dof, axis=i)

	evals, evecs = eigh(K,M)
	frequencies = np.sqrt(evals)
	return M, K, frequencies, evecs

def bar3(num_elems):
	restrained_dofs = [0, ]

	# element mass and stiffnessm matrices
	m = np.array([[4,2,-1],[2,16,2],[-1,2,4]]) / (30.0 * num_elems)  
	k = np.array([[7,-8,1],[-8,16,-8],[1,-8,7]]) / 3.0 * num_elems  

	# construct global mass and stiffness matrices
	M = np.zeros((2*num_elems+1,2*num_elems+1))
	K = np.zeros((2*num_elems+1,2*num_elems+1))

	# for each element, change to global coordinates
	for i in range(num_elems):
		M_temp = np.zeros((2*num_elems+1,2*num_elems+1))
		K_temp = np.zeros((2*num_elems+1,2*num_elems+1))

		first_node = 2*i
		M_temp[first_node:first_node+3, first_node:first_node+3] = m   # 3 x 3
		K_temp[first_node:first_node+3, first_node:first_node+3] = k   # 3 x 3
		M += M_temp
		K += K_temp

	# remove the fixed degrees of freedom
	for dof in restrained_dofs:
		for i in [0,1]:
			M = np.delete(M, dof, axis=i)
			K = np.delete(K, dof, axis=i)

	evals, evecs = eigh(K,M)
	frequencies = np.sqrt(evals)
	return M, K, frequencies, evecs

# 2 node bar element
print '2 Node bar element'
exact_frequency = math.pi/2
errors = []
for i in range(1,11):
	start = time.clock()
	M, K, frequencies, evecs = bar2(i)
	time_taken = time.clock() - start
	error = (frequencies[0] - exact_frequency) / exact_frequency * 100.0
	errors.append( (i, error) )
	print 'Num Elems: {} \tFrequency: {}\tError: {}% \tShape: {} \tTime: {}'.format( i, round(frequencies[0],4), round(error, 4), K.shape, round(time_taken*1000, 3) )

print 'Exact Freq:', round(exact_frequency, 4)

element  = np.array([x[0] for x in errors])
error2   = np.array([x[1] for x in errors])


# 3 node bar element
print '3 Node bar element'
exact_frequency = math.pi/2
errors = []
for i in range(1,11):
	start = time.clock()
	M, K, frequencies, evecs = bar3(i)
	time_taken = time.clock() - start
	error = (frequencies[0] - exact_frequency) / exact_frequency * 100.0
	errors.append( (i, error) )
	print 'Num Elems: {} \tFrequency: {}\tError: {}% \tShape: {} \tTime: {}'.format( i, round(frequencies[0],4), round(error, 4), K.shape, round(time_taken*1000, 4) )

print 'Exact Freq:', round(exact_frequency, 4)

element  = np.array([x[0] for x in errors])
error3   = np.array([x[1] for x in errors])


# plot the result
plt.plot(element, zip(error2, error3), 'o-')
plt.xlim(1, element[-1])
plt.xlabel('Number of Elements')
plt.ylabel('Errror (%)')
plt.show()

