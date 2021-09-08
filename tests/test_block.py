
import numpy as np

A = np.array([[1,0],
     [0,2]])
B = np.array([3])
C = np.kron(A, B)
D = np.outer(A, B)
E = np.block([[A, np.zeros((2, 1))],
             [np.zeros((1, 2)), B]])
print(A, B, C, D)

# example: https://numpy.org/doc/stable/reference/generated/numpy.block.html
A = np.eye(2) * 2
B = np.eye(1) * 3
F = np.block([[A, np.zeros((2, 1))],
          [np.zeros((1, 2)), B]])

# https://www.delftstack.com/howto/python/print-matrix-python/
a = np.array([[1,2,3],[3,4,5],[7,8,9]])
print(a)

for row in a:
    print ('  '.join(map(str, row)))

print('\n'.join([''.join(['{:4}'.format(item) for item in row])
                     for row in a]))

s = [[str(e) for e in row] for row in a]
lens = [max(map(len, col)) for col in zip(*s)]
fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
table = [fmt.format(*row) for row in s]
print ('\n'.join(table))

