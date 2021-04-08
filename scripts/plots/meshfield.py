import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
#plt.figure(1)
a, x, tot = np.loadtxt('fields.csv', delimiter=',', unpack=True)
fig=plt.figure(1)
gamma=10
tf_n=tot/-gamma**4
a=-a/180*np.pi
x=x
cm = plt.cm.get_cmap('jet')
#sc = ax.scatter(x, a, c=tf_n, marker='o', s=5, cmap=cm ) 
sc = plt.scatter(a, x, c=tf_n, vmin=-10, vmax=10, marker='o', s=1, cmap=cm )
#ax = fig.add_subplot(111)
plt.colorbar(sc)
#ax.set_facecolor('grey' )
xx=0.005
aa=0.0005
plt.ylim([-xx,xx]) 
plt.xlim([-aa,aa])
plt.ylabel('x')
plt.xlabel('a')
#plt.title('field on the wavefronts')
#l = plt.legend(loc='upper right')
plt.grid(True)
#plt.show()
#plt.savefig('csr_mesh_field.png')

# define grid.
#pos =np.array([x, a]).T
#grid_x, grid_a = np.mgrid[-0.05:0.05:500j, -0.005:0.005:500j]
#grid_z0 = griddata( pos, tf_n, (grid_x, grid_a), method='nearest')

plt.figure(2)
# define grid.                                                                                                                                                                      
xi = np.linspace(-0.005,0.005,100)
ai = np.linspace(-0.0005,0.0005,100)
pos =np.array([a, x]).T
# grid the data.
zi = griddata((a, x), tf_n, (ai[None,:], xi[:,None]), method='cubic')
# contour the gridded data, plotting dots at the randomly spaced data points.
levels = np.linspace(-10.0, 10.0, 11)
CS = plt.contour(ai,xi,zi,levels=levels,linewidths=0.5,colors='k')
CS = plt.contourf(ai,xi,zi,levels=levels, cmap=plt.cm.jet)
plt.colorbar() # draw colorbar
# plot data points.
#plt.scatter(x,a,marker='o',c='b',s=1)
plt.ylim(-xx,xx)
plt.xlim(-aa,aa)
#plt.title('griddata test (%d points)' % npts)
plt.show()



