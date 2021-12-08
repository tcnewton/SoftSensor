import numpy as np
from numpy import hstack

# split a univariate sequence into samples
# sequence = (input variables, output)
def split_sequences(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):            
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence):
            break
    # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix, :-1], sequence[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
#transform 3d sequence in 2d sequence in order 'F'
def ravel_sequence (inputseq):
    z = list()
    for s in range(len(inputseq)):
        l = np.ravel(inputseq[s],order='F')
        z.append(l)
    return z

# create a dataset for test
dataset = np.array([[10,20,30,40,50,60],[11,21,31,41,51,61],[12,22,32,42,52,62],
[13,23,33,43,53,63]])
#time steps
n_steps = 2
# convert into input/output
X, y = split_sequences(dataset, n_steps)
teste = ravel_sequence(X)
print("shape x = {}".format(np.shape(X)))
print("shape X transformed = {}".format(np.shape(teste)))
print("shape y = {}".format(np.shape(y)))
# summarize the data
for i in range(len(X)):
    print(X[i], y[i],teste[i])