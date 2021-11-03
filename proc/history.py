from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

# abc for histories
class baseHist(object,metaclass=ABCMeta):
    m_x = np.arrange(0)
    m_Error = np.arrange(0)
    m_dir = ""
    def __init__(self,dir_path):
        self.m_dir = dir_path

    def addHist(self,x,y):
        self.m_x = np.append(self.m_x,x)
        self.m_y = np.append(self.m_Error,y)
    
    @abstractmethod
    def out(self):
        pass

#Error History
class ErrorHist(baseHist):
    m_fig = plt.figure(figsize=(6,5))
    def __init__(self,dir_path):
        super().__init__(dir_path)

    def out(self):
        fname = self.m_dir + "\\ErrorHist.png"
        self.m_fig.plot(self.m_x,self.m_y)
        self.m_fig.xlabel("iter")
        self.m_fig.ylabel("Error")
        self.m_fig.savefig(fname)
        self.m_fig.close()


