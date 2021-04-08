from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

particleid = "_1"
pos_x, pos_y = np.loadtxt('trajectory'+particleid+'.csv', delimiter=',', unpack=True)
m_pos_x, m_pos_y = np.loadtxt('comoving_mesh_pos.csv', delimiter=',', unpack=True)
w_pos_x, w_pos_y = np.loadtxt('wavefronts'+particleid+'.csv', delimiter=',', unpack=True)
firstwave_x = w_pos_x[:20]
firstwave_y = w_pos_y[:20]

fig, ax= plt.subplots()
ax.scatter(m_pos_x,m_pos_y, s=6, facecolor='lightseagreen', label="mesh")
ax.scatter(w_pos_x,w_pos_y, s=1, facecolor='brown', label="wavefronts")
ax.scatter(pos_x,pos_y, s=1, facecolor='blue', label="tracjectory")
ax.scatter(firstwave_x,firstwave_y, s=4, facecolor='darkblue', label="first_time_step")


#plt.xlim([-0.6,0.6])
#plt.ylim([-1,0.2])
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Synchrontron radiation')
l = plt.legend(loc='upper right')
ax.grid(True)
plt.show()
plt.savefig('csr_mesh.png')
