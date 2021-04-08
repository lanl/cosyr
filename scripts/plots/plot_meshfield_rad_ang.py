from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
atil, xtil ,tot= np.loadtxt('comoving_mesh_rad_ang.csv', delimiter=',', unpack=True)
#vel_f, acc_f, tot = np.loadtxt('comoving_mesh_field.csv', delimiter=',', unpack=True)

fig, ax= plt.subplots(1)
tf_n=tot/(-1e4)
cm = plt.cm.get_cmap('jet')
#coolwarm
sc = ax.scatter( xtil, atil, c=tf_n, vmin=-5, vmax=5, marker='.', s=5, cmap=cm ) 

#sc = ax.scatter(xtil, atil, c=tot, vmin=-1000, vmax=1000, marker='.', s=10, cmap=cm )
#plt.ylim(0.995,1.005)
#plt.xlim(-0.01,0.01)
plt.colorbar(sc)
ax.set_facecolor('grey' )
plt.xlabel('x')
plt.ylabel('y')
plt.title('field on the mesh ')
l = plt.legend(loc='upper right')
ax.grid(True)
plt.show()
plt.savefig('csr_mesh_field_rad_ang.png')

#fig=plt.figure(2)
#ax = fig.add_subplot(111, projection='3d')
#m_x = np.linspace( 0.995, 1.005, 401 )
#m_a = np.linspace( -0.05 , 0.05, 401)
#z= np.reshape(tf_n, (-1,401))#
#x, y = np.meshgrid(m_x, m_a,sparse=True)
#dem3d=ax.plot_surface(x,y,z,cmap=cm )
#plt.show()








