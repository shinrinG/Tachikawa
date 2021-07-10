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

    def OMP(self):
        """
        OMP : Orthogornal Matching Pursuit"
        x : Sparse Representation (m x 1)
        S : Support of x (OneHotVector)
        r : Residual error
        """
        #Initialize

        x = np.zeros(self.A.shape[1])
        S = np.zeros(self.A.shape[1],dtype=np.uint8)
        r = self.b.copy()
        rr = np.dot(r,r)

        for _ in range(self.A.shape[1]):
            #Calc Error
            err = rr - np.dot(self.A[:,S==0].T,r)**2

            #Update Support
            ndx = np.where(S==0)[0]
            S[ndx[err.argmin()]] = 1

            #Update Representation
            As = self.A[:,S==1]
            pinv = np.linalg.pinv(np.dot(As,As.T))
            x[S==1] = np.dot(As.T,np.dot(pinv,self.b))

            #Update Residual error
            r = self.b - np.dot(self.A, x)
            rr = np.dot(r, r)
            if rr < self.eps:
                break
        
        return x,S



