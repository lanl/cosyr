from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

pos_x, pos_y = np.loadtxt('wavefronts.csv', delimiter=',', unpack=True)

# 
firstwave_x = pos_x[:600]
firstwave_y = pos_y[:600]
#lastwave_x=pos_x[16500:17000]
#lastwave_y=pos_y[16500:17000]

traj_x, traj_y = np.loadtxt('trajectory.csv', delimiter=',', unpack=True)

fig, ax= plt.subplots()

ax.scatter(pos_x,pos_y, s=1, facecolor='lightseagreen', label="wavefronts")
#ax.scatter(lastwave_x,lastwave_y, s=1, facecolor='darkorange', label="last_time_step")
ax.scatter(firstwave_x,firstwave_y, s=1, facecolor='darkblue', label="first_time_step")
ax.scatter(traj_x,traj_y, s=1, facecolor='brown', label="trajectory")

#plt.xlim([-1,1])
#plt.ylim([-1,1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('traj and wavefronts')
l = plt.legend(loc='upper right')
ax.grid(True)
plt.show()
plt.savefig('csr_wavefronts.png')
