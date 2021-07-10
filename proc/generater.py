import  numpy as np

#1D Gaussian Signal Generate
class ShiftedGause1D(object):
    def __init__(self,coef,mu,sigma,len=32):
        print(type(len))
        self.x = np.arange(len)
        self.length = len
        self.mu = mu
        self.sigma = sigma
        self.discriminator = 2*sigma**2
        self.coef = coef

    def gen_signal(self):
        ret = np.zeros(self.x.shape[0])
        for i in range(self.length):
            ret[i] = self.coef * np.exp(-(self.x[i]-self.mu)**2/self.discriminator)
        return ret

#1D signam generate
class signalOnbit(object):
    def __init__(self,len,size,stride=1):
        self.base = np.zeros(len)
        self.len = len
        self.iter = -1
        self.size = size
        self.stride = stride
    
    def __call__(self):
        self.iter += 1 
        start = (int)(self.stride*self.iter) % self.size
        ret = self.base.copy()
        for i in range(self.size):
            ind = start+i
            ret[ind]=1/self.size
        return ret

