import numpy as np

data = np.load('./Data/single_variable_classification/testlabel.npy')

print(data.shape)
print(data.dtype)
print(data[:5])
