import numpy as np

a = np.zeros( (3, 4, 5 ) )

b = np.zeros( a.shape + ( 6, 7 ) )

print(b.shape)
