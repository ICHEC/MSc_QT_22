class schrodinger:
    def __init__(self,x,V,m=1,a=0,t=1):
        self.x = x
        self.V = V
        self.size = x.size
        self.h = x[1]-x[0]
        
        #Define the time span over which the integration will take place
        self.t0 = 0.0    #initial time
        self.tf = t    #final time
        self.dt = self.h    #Time steps
        self.t_eval = np.arange(self.t0, self.tf, self.dt)  #Time span
    
        self.hbar=1
        self.m = m
        self.a = a
    
        #Define the Laplace Operator
        self.L = scipy.sparse.diags([1, -2, 1], [-1, 0, 1], shape=(self.size, self.size)) / self.h**2
        
        #Defining the initial state psi
        self.f = 8*(2*np.pi)
        self.sigma = 1.0 
        self.packet = (1.0/np.sqrt(self.sigma*np.sqrt(np.pi)))*np.exp(-(self.x-self.a)**2/(2.0*self.sigma**2))*np.exp(1j*self.f*self.x)

        #Normalise
        self.packet = self.packet/(np.sqrt(sum((abs(self.packet))**2)))
    
        def wave_fun(t, psi):
            return -1j*(-(self.hbar/(2*self.m))*self.L.dot(psi) + (self.V/self.hbar)*psi)
    
        #Solve the integral problem
        self.sol = integrate.solve_ivp(wave_fun, t_span = [self.t0, self.tf], y0 = self.packet, t_eval = self.t_eval)
        
        self.F = np.zeros(shape=(self.size,self.sol.y.size),dtype='complex')
        self.w = fftfreq(self.size, self.h)


        for i, t in enumerate(self.sol.t):
            self.F[:,i] = fftpack.fft(self.sol.y[:,i])
            self.F[:,i] = self.F[:,i]/(np.sqrt(sum((abs(self.F[:,i])**2))))
            self.sol.y[:,i] = self.sol.y[:,i]/(np.sqrt(sum((abs(self.sol.y[:,i])**2))))

    def animater(self):
        fig = plt.figure()
        ax1 = plt.subplot(1,1,1)
        
        limit = 5*max(np.abs(self.sol.y[:,0])**2)
        scaler = 1/(max(self.V)/limit)

        ax1.set_xlim(self.x[0], self.x[-1])
        ax1.set_ylim(0, limit)
        ax1.set_xlabel('x')
        ax1.set_ylabel('psi(x)')

        title = ax1.set_title('The Quantum Harmonic Osciallator')

        line1, = ax1.plot([], [], "--")
        line2, = ax1.plot([], [],color='red')
        
        
        def init():
            line1.set_data(self.x, self.V*scaler)
            return line1,


        def animate(i):
            line2.set_data(self.x, np.abs(self.sol.y[:,i])**2)
            title.set_text('t = {0:1.3f}'.format(self.sol.t[i]))
            return self.line1,
        
        self.anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(self.sol.t), interval=50, blit=True)    
        plt.show()
        
    def f_animater(self):
        fig = plt.figure()
        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(212)
        
        limit = 2.5*max(self.F[:,0])
        scaler = 1/(max(self.V)/limit)

        ax1.set_xlim(self.x[0], self.x[-1])
        ax1.set_ylim(-limit,limit)
        ax1.set_xlabel('x')
        ax1.set_ylabel('psi(x)')
        
        ax2.set_xlim(self.x[0], self.x[-1])
        ax2.set_ylim(-limit, limit)
        ax2.set_xlabel('w=p/h')
        ax2.set_ylabel('psi(w)')

        title = ax1.set_title('The Quantum Harmonic Osciallator')

        line1, = ax1.plot([], [], "--")
        line2, = ax1.plot([], [],color='red')
        line3, = ax2.plot([], [],color='blue')
        line4, = ax2.plot([], [], "--")
        
        def init():
            line1.set_data(self.x, self.V*scaler)
            line4.set_data(self.x, self.V*scaler)
            return line1,line4


        def animate(i):
            line2.set_data(self.x, self.sol.y[:,i].real)
            line3.set_data(self.w, self.F[:,i].real)
            title.set_text('t = {0:1.3f}'.format(self.sol.t[i]))
            return line2,line3
        
        self.anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(self.sol.t), interval=50, blit=True)    
        plt.show()
        
    def plot(self):
        fig = plt.figure()
        ax = plt.subplot(111)
        limit = 5*max(np.abs(self.sol.y[:,0])**2)
        scaler = 1/(max(self.V)/limit)

        #Plot the solutions at various time intervals
        for i, t in enumerate(self.sol.t):
            if i%20==0:
                ax.plot(self.x, np.abs(self.sol.y[:,i])**2,label='t = {}'.format(t)) 

        #Plot the potential, adjusting size for visual aid
        ax.plot(self.x, self.V*scaler, "--", label='V')   # Plot Potential

        #Plot the legend outside the graph
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3, fancybox=True, shadow=True)

        ax.set_xlabel('x')
        ax.set_ylabel('psi(x)')
        plt.show()
        
h = 0.01
x = np.arange(-10, 10, h)

m = 1
w = 2*np.pi
a = 0    
V = 0.5*m*(w**2)*(x-a)**2

osc = schrodinger(x,V) 

osc.plot()

osc.f_animater()

h = 0.01
x = np.arange(0, 10, h)
size = x.size

hbar = 1 
m=1

x_Vmin = 5         # center of V(x)
T      = 1  

omega = 2 * np.pi / T
k = omega**2 * m
V = np.zeros(size)
V[int(V.size/2):] = x_Vmin*250

step = schrodinger(x,V,a=2.5,t=0.5)

step.plot()
step.f_animater()