import  numpy as np
from abc import ABCMeta, abstractmethod

#abstract base class for signal gen
class baseSignal(object,metaclass=ABCMeta):
    m_len = 0
    m_x = np.arange(0)
    m_y = np.arange(0)
    m_begin = 0
    m_end = 0

    def __init__(self,begin,end,len=32):
        if(len != 1):
            self.m_len = len
            self.m_begin = begin
            self.m_end = end
            self.m_x = np.linspace(self.m_begin,self.m_end,self.m_len,dtype=np.float)
            self.m_y = np.zeros(self.m_len)
    
    @abstractmethod
    def calc(self):
        pass

    def generate(self):
        self.calc()
        return self.m_y

#1D Gaussian Signal Generate
class GaussSignal(baseSignal):
    m_mu = 0.0
    m_sigma = 1.0
    m_coef = 0.0
    discriminator=1
    def __init__(self,coef,mu,sigma,begin,end,len=32):
        super().__init__(begin,end,len)
        self.m_mu = mu
        self.m_sigma = sigma
        self.m_coef = coef
        self.discriminator = 2*sigma**2
    
    def calc(self):
        print(self.m_x.shape[0])
        print(self.m_y.shape[0])
        for i in range(self.m_len):
            self.m_y[i] = self.m_coef * np.exp(-(self.m_x[i]-self.m_mu)**2/self.discriminator)

#1D Normalized Rectangle Signal Generate    
class NormRectSignal(baseSignal):
    m_upx = 0.0
    m_width = 1.0
    def __init__(self,upx,width,begin,end,len=32):
        super().__init__(begin,end,len)
        self.m_upx = upx
        self.m_width = width

    def calc(self):
        for i in range(self.m_len):
            if (self.m_x[i] >= self.m_upx )and( self.m_x[i] <= (self.m_upx +self.m_width)):
                self.m_y[i] = 1.0/self.m_width