from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

# abc for histories
class baseHist(object,metaclass=ABCMeta):
    m_x = np.arange(0)
    m_Error = np.arange(0)
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
    #m_fig = plt.figure(figsize=(6,5))
    def __init__(self,dir_path):
        super().__init__(dir_path)

    def out(self):
        fname = self.m_dir + "\\ErrorHist.png"
        fig = plt.figure(figsize=(6,5))
        plt.plot(self.m_x,self.m_y)
        plt.xlabel("iter")
        plt.ylabel("Error")
        plt.savefig(fname)
        plt.close()

#process History
class ProcHist(baseHist):
    #m_fig = plt.figure(figsize=(6,5))
    m_title = ""
    m_bScat = False
    def __init__(self,dir_path,title,scat=False):
        super().__init__(dir_path)
        self.m_title = title
        self.m_bScat = scat
    
    def out(self):
        fname = self.m_dir + "\\" + self.m_title +".png"
        fig = plt.figure(figsize=(6,5))
        ax = plt.axes()
        # ax.set_xlim([0,self.m_x.max()])
        # ax.set_ylim([0,None])
        if self.m_bScat:
            plt.scatter(self.m_x,self.m_y)
        else:
            plt.plot(self.m_x,self.m_y)
        plt.xlabel("iter")
        plt.ylabel(self.m_title)
        plt.savefig(fname)
        plt.close()


