import numpy as np

class Greedy1D(object):
    def __init__(self,A,b,epsilon=1e-4):
        """
        A : Dictionary Matrix (n x m, n<m)
          : Normalised basis use. 
        b : Observe Vector (n x 1)
        epsilon : Threshold for Error
        """
        self.A=A
        self.b=b
        self.epsilon = epsilon
        self.itrhist=[]
        self.reshist=[]
        self.addhist=[]

    def OMP(self):
        """
        OMP : Orthogornal Matching Pursuit"
        x : Sparse Representation (m x 1)
        S : Support of x (OneHotVector)
        r : Residual error
        """
        #Initialize
        self.reshist=[]
        self.addhist=[]
        self.itrhist=[]
        cur_itr = 0
        x = np.zeros(self.A.shape[1])
        S = np.zeros(self.A.shape[1],dtype=np.uint8)
        r = self.b.copy()
        rr = np.dot(r,r)
        target = 0

        for _ in range(self.A.shape[1]):
            #Calc Error
            err = rr - np.dot(self.A[:,S==0].T,r)**2

            #Update Support
            ndx = np.where(S==0)[0]
            target = ndx[err.argmin()]
            S[target] = 1

            #Update Representation
            As = self.A[:,S==1]
            pinv = np.linalg.pinv(np.dot(As,As.T))
            x[S==1] = np.dot(As.T,np.dot(pinv,self.b))

            #Update Residual error
            r = self.b - np.dot(self.A, x)
            rr = np.dot(r, r)
            self.itrhist.append(cur_itr)
            self.reshist.append(rr)
            self.addhist.append(target)
            cur_itr+=1
            if rr < self.epsilon:
                break
        
        return x,S



