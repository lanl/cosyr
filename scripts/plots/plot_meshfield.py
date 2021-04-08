from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
pos_x, pos_y = np.loadtxt('trajectory.csv', delimiter=',', unpack=True)
m_pos_x, m_pos_y = np.loadtxt('comoving_mesh_pos.csv', delimiter=',', unpack=True)
vel_f, acc_f, tot = np.loadtxt('comoving_mesh_field.csv', delimiter=',', unpack=True)
fig, ax= plt.subplots()
tf_n=tot/-10000
ax.scatter(pos_x,pos_y, s=3, facecolor='black', label="trajectory")
cm = plt.cm.get_cmap('jet')
sc = ax.scatter(m_pos_x, m_pos_y, c=tf_n, vmin=-10, vmax=10, marker='.', s=10, cmap=cm ) 
#sc = ax.scatter(m_pos_x, m_pos_y, c=tot, vmin=-10000, vmax=10000, marker='.', s=10, cmap=cm )
plt.colorbar(sc)
ax.set_facecolor('grey' )
#plt.xlim([-1,1])
#plt.ylim([-2,0])
plt.xlabel('x')
plt.ylabel('y')
plt.title('field on the mesh ')
l = plt.legend(loc='upper right')
ax.grid(True)
plt.show()
plt.savefig('csr_mesh_field.png')








