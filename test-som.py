import matplotlib  
matplotlib.use('Agg') 
import matplotlib.pylab as plt
from matplotlib.pyplot import savefig
# import sompy as sompy
import pandas as pd
import numpy as np
from time import time
import sompy

dlen = 200
Data1 = pd.DataFrame(data= 1*np.random.rand(dlen,768))
#fig = plt.figure()
#plt.plot(Data1[:,0],Data1[:,1],'ob',alpha=0.2, markersize=4)

#fig.set_size_inches(7,7)
mapsize = [1,8]
som = sompy.SOMFactory.build(Data1, mapsize, mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='sompy')  # this will use the default parameters, but i can change the initialization and neighborhood methods
som.train(n_job=1, verbose='info', train_rough_len=200, train_finetune_len=500)  # verbose='debug' will print more, and verbose=None wont print anything

#v = sompy.mapview.View2DPacked(50, 50, 'test',text_size=8)  
# could be done in a one-liner: sompy.mapview.View2DPacked(300, 300, 'test').show(som)
#v.show(som, what='codebook', which_dim=[0,1], cmap=None, col_sz=6) #which_dim='all' default
# v.save('2d_packed_test')


savefig('test.png')
