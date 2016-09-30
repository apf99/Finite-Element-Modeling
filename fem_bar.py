import numpy as np
from scipy.linalg import eigh
import math
from matplotlib import pyplot as plt

def bar(num_elems):
	restrained_dofs = [0,]

	# element mass and stiffness matrices for a bar
	m = np.array([[2,1],[1,2]]) / (6. * num_elems)
	k = np.array([[1,-1],[-1,1]]) * float(num_elems)

	# construct global mass and stiffness matrices
	M = np.zeros((num_elems+1,num_elems+1))
	K = np.zeros((num_elems+1,num_elems+1))

	# assembly of elements
	for i in range(num_elems):
		M_temp = np.zeros((num_elems+1,num_elems+1))
		K_temp = np.zeros((num_elems+1,num_elems+1))
		M_temp[i:i+2,i:i+2] = m
		K_temp[i:i+2,i:i+2] = k
		M += M_temp
		K += K_temp

	# remove the fixed degrees of freedom
	for dof in restrained_dofs:
		for i in [0,1]:
			M = np.delete(M, dof, axis=i)
			K = np.delete(K, dof, axis=i)

	# eigenvalue problem
	evals, evecs = eigh(K,M)
	frequencies = np.sqrt(evals)
	return M, K, frequencies, evecs


exact_frequency = math.pi/2
results = []
for i in range(1,11):
	M, K, frequencies, evecs = bar(i)
	error = ( frequencies[0] - exact_frequency ) / exact_frequency * 100.0
	results.append( (i, error) )
	print 'Num Elems: {} \tFund. Frequency: {} \t Error: {}%'.format(i, round(frequencies[0],3), round(error,3))

print 'Exact frequency: ', round(exact_frequency,3)

# plot the results
elements = np.array([x[0] for x in results])
errors   = np.array([x[1] for x in results])

plt.plot(elements,errors, 'o-')
plt.xlim(elements[0], elements[-1])
plt.xlabel('Number of Elements')
plt.ylabel('Error (%)')
plt.show()

