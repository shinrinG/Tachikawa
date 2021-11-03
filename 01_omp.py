from proc.generater import GaussSignal,NormRectSignal
from proc.history import ErrorHist, ProcHist
import numpy as np
from commons.utils import *
from proc.omp1d import Greedy1D

#Global params
TRY_TAG = "01"
EXP_ROOT = "C:\\Test\\OMP"
#SignalSectionShared
SIGNAL_LENGTH = 64
BASIS_NUM = 65
X_MIN = 0
X_MAX = 63
#SignalSection1
MU = 32
SIGMA = 1
COEFF = 1
#SignalSection2
WIDTH = 1
#OMPSection
EPS = 1e-4

#main
if __name__=="__main__":

    #00.initialize
    logname = EXP_ROOT + "\\01"+TRY_TAG+"_log.txt"
    log = logger(logname)
    log.write("Process Start")

    #01.createSignals
    #01-1.ObservedSignal
    gsig = GaussSignal(COEFF,MU,SIGMA,X_MIN,X_MAX,SIGNAL_LENGTH)
    obs = gsig.generate()
    #01-2.BasisSignal
    bsig_list = []
    bsig = NormRectSignal(0,WIDTH,X_MIN,X_MAX,SIGNAL_LENGTH)
    for i in range(SIGNAL_LENGTH):
        bsig.update(i,WIDTH)
        basis = bsig.generate()
        x = np.copy(basis)
        bsig_list.append(x)
    bsig.update(0,WIDTH)
    basis = bsig.generate()
    x = np.copy(basis)
    bsig_list.append(x)

    #02.createMatrixes
    dicA = bsig_list[0].reshape(SIGNAL_LENGTH,1)
    for i in range(len(bsig_list)):
        if i==0:
            continue
        dicA = np.hstack((dicA,bsig_list[i].reshape(SIGNAL_LENGTH,1)))
    log.write("get dictionary.")
    #03.Optimize
    log.write("OptimizeStart.")
    omp = Greedy1D(dicA,obs)
    spr,sup = omp.OMP()
    message = "sparce code ...\n"
    for i in range(len(spr)):
        message += str(spr[i])
        message += ","
    log.write(message)
    log.write("OptimizeEnd.")

    #04.ReconstructSignal
    recon = np.dot(dicA,spr)

    #05.CheckHistory
    #05-1 optimize history (Error)
    his_er = ErrorHist(EXP_ROOT)
    his_er.addHist(omp.itrhist,omp.reshist)
    his_er.out()
    #05-2 optimize history (SupportSelect)
    his_sel = ProcHist(EXP_ROOT,"SupportSelect",scat=True)
    his_sel.addHist(omp.itrhist,omp.addhist)
    his_sel.out()
    #05-3 origin signal
    his_org = ProcHist(EXP_ROOT,"OriginSignal")
    his_org.addHist(np.linspace(X_MIN,X_MAX,SIGNAL_LENGTH),obs)
    his_org.out()
    #05-4 reconst signal
    his_rcn = ProcHist(EXP_ROOT,"ReconstSignal")
    his_rcn.addHist(np.linspace(X_MIN,X_MAX,SIGNAL_LENGTH),recon)
    his_rcn.out()
    #05-5 sparse code
    his_spc = ProcHist(EXP_ROOT,"SparseCode")
    his_spc.addHist(np.arange(0,spr.shape[0],1),spr)
    his_spc.out()