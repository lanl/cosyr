from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

pos_x, pos_y = np.loadtxt('wavefronts.csv', delimiter=',', unpack=True)
traj_x, traj_y = np.loadtxt('trajectory.csv', delimiter=',', unpack=True)
vfld, afld, tf = np.loadtxt('field.csv', delimiter=',', unpack=True)
tf_n=tf/-10000

fig, ax= plt.subplots()
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(pos_x, pos_y, tf_n,  marker='o')

cm = plt.cm.get_cmap('jet')
#sc=ax.scatter(pos_x,pos_y,c=tf, marker='o',s=3,cmap=cm )
sc=ax.scatter(pos_x,pos_y,c=tf_n, vmin=-8, vmax=8, marker='o',s=2,cmap=cm )
plt.colorbar(sc)           
#ax.set_facecolor('grey' )

#sc=ax.scatter(pos_x,pos_y,c=tf_n, vmin=-5, vmax=5,marker='.',s=10,cmap=cm )
#plt.colorbar(sc)

#plt.xlim([-1,1])
#plt.ylim([-2,0])

plt.xlabel('x')
plt.ylabel('y')
plt.title('Synchrontron radiation')
l = plt.legend(loc='upper right')
ax.grid(True)
plt.show()
plt.savefig('csr_circular_field.png')

