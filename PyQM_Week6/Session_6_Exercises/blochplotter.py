import scipy
import scipy.linalg as linalg
import scipy.optimize as opt
from scipy.fft import fft, fftfreq, fftshift

import numpy as np
import numpy.random as rnd

import matplotlib.pyplot as plt

def blochplot(theta,phi):
    #Defining the state
    psi = [np.cos(theta/2.0),np.exp(1j*phi)*np.sin(theta/2.0)]
    
    #printing it's probability amplitudes
    print('psi = {}|0> + {}|1>'.format(psi[0],psi[1]))
    print('psi prob = {} , {}'.format(np.round(abs(psi[0])**2,2),np.round(abs(psi[1])**2,2)))

    #Using spherical coordinates
    x=np.cos(phi)*np.sin(theta)
    y=np.sin(phi)*np.sin(theta)
    z=np.cos(theta)

    #Plotting a sphere of radius 1, with a black point at it's centre
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_axes([0,0,1,1],projection='3d')
    ax.grid(False)
    ax.set_axis_off()
    
    u, v = np.mgrid[0:2*np.pi:200j, 0:np.pi:100j]
    xsphere = np.cos(u)*np.sin(v)
    ysphere = np.sin(u)*np.sin(v)
    zsphere = np.cos(v)
    
    ax.plot_wireframe(xsphere, ysphere, zsphere, color="b",lw=0.3)
    ax.scatter([0], [0], [0], color="black", s=50)
    
    #Draw a new axis
    ax.plot([-2,2],[0,0],[0,0],color='black',linestyle='--')
    ax.plot([0,0],[-2,2],[0,0],color='black',linestyle='--')
    ax.plot([0,0],[0,0],[-2,2],color='black',linestyle='--')
    
    #Plotting Text Labels
    ax.text(0, -0.0, 1.3, "|0>", color='black',size=15)
    ax.text(0, -0.0, -1.5, "|1>", color='black',size=15)
    
    
    #ax.text(0, -1.5, -0.2, "|+>", color='black',size=15)
    #ax.text(0, 1.4, 0.2, "|->", color='black',size=15)

    #Plotting a vector of the quantum state.
    ax.plot([0,x],[0,y],[0,z], color='r')
    ax.scatter([x], [y], [z], color="red", s=50,marker= 'o')
    
    plt.show()