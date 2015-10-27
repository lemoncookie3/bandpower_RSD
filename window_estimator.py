import numpy as np
from numpy import zeros, sqrt, pi
from numpy.linalg import inv
from numpy import vectorize
from scipy.interpolate import interp1d
from covariance_class2 import *
import pp, sys, time, datetime
from multiprocessing import Process, Queue
import matplotlib.cm as cm
import matplotlib.pyplot as plt

#   Initialization

print '\nwindow estimator'
print datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

# int range
KMIN = 0.001
KMAX = 502.32

# r scale
RMIN = .1
RMAX = 2000.
# k scale
kmin = KMIN
kmax = KMAX
# BAO+RSD(r 24 ~ 152) (k: 0.01 ~ 0.2))
# BAO only(r 29 ~ 200) (k : 0.02 ~ 0.3)


# REID (0.01~ 180) corresponding k value :(0.02 ~ 361.28)
# REID convergence condition : kN = 61, rN = 151, subN = 101
# REID convergence condition : kN = 101, rN = 101, subN = 121
kN = 61  #converge perfectly at 151, 2by2 components converge at 121 # the number of k bins. the sample should be odd
rN = 151 #101 for Reid # number of r bins
subN = 51 # to keep sdlnk for 61 k bins, take subN = 248 #101 for Reid

Window = Window(KMIN, KMAX, RMIN, RMAX, kN, rN, subN)
#Window.compile_fortran_modules() ## run only for the first time running

#file = open('matterpower_z_0.55.dat')
file = open('camb_LINmatterpower_z0.0.dat')
Window.MatterPower(file)
Window.Shell_avg_band()


import matplotlib.pyplot as plt

unwindowed_P = Window.unwindowed_P()
unwindowed_dCdp = Window.unwindowed_dCdp()

windowed_P = Window.windowed_P()
dCdp = Window.dCdp()

windowed_Xi = Window.windowed_Xi()
dCXidp = Window.dCXidp()

covariance_P = Window.Covaiance_window()
dPdp = Window.dPdp()

covariance_Xi = Window.Covaiance_window_Xi()
dXidp = Window.dXidp()



def PS_plotting():
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    V = 4 * pi/3. * RMAX**3
    fig = plt.figure()
    #plt.loglog(Window.kcenter, dCdp[0].diagonal())
    plt.loglog(Window.kcenter, Window.Pmlist, label ='No Window')
    plt.loglog(Window.kcenter, Window.Signal_P.diagonal() , label ='Signal')
    plt.loglog(Window.kcenter, Window.Noise_P.diagonal() , label ='Shot Noise')
    plt.loglog(Window.kcenter, windowed_P.diagonal() , label ='Signal + Shot Noise')
    plt.legend(loc=3)
    plt.title('Power Spectrum')
    #plt.show()
    pdf=PdfPages( 'power_spectrums.pdf' )
    pdf.savefig(fig)
    pdf.close()
    print ' pdf file saved :  power_spectrums.pdf'
PS_plotting()

#plt.loglog(Window.kcenter, dCdp[25].diagonal())
#plt.show()


Cross1 = CrossCoeff( Window.Signal_P )
Cross2 = CrossCoeff( windowed_P )
Cross3 = CrossCoeff( Window.Signal_Xi )
Cross4 = CrossCoeff( windowed_Xi )
"""
Contour_plot( Window.kcenter, Cross1, title='Signal P_ij (no shot noise)', pdfname='Signal_P.pdf' )
Contour_plot( Window.kcenter, Cross2, title='Signal + Noise P_ij', pdfname='C_P.pdf' )
Contour_plot( Window.rcenter, Cross3, title='Signal Xi_ij (no shot noise)', pdfname='Signal_Xi.pdf', basename = 'log10(r)' )
Contour_plot( Window.rcenter, Cross4, title='Signal + Noise Xi_ij', pdfname='C_Xi.pdf',basename = 'log10(r)' )
"""
#Fisher_P = Window.Quadratic_Fisher(windowed_P, dCdp)
#Fisher_Xi = Window.Quadratic_Fisher(windowed_Xi, dCXidp)
Fisher_P_Traditional = np.dot( np.dot( dPdp, np.linalg.inv(covariance_P) ), np.transpose(dPdp))
Fisher_Xi_Traditional = np.dot( np.dot( dXidp, np.linalg.inv(covariance_Xi) ), np.transpose(dXidp))
Fisher_P_unwindowed = Window.Quadratic_Fisher(unwindowed_P, unwindowed_dCdp)


#CovP = np.linalg.inv(Fisher_P)
CovP = Window.Quadratic_Cov(windowed_P,dCdp)
CovXi = Window.Quadratic_Cov(windowed_Xi, dCXidp)
CovP_traditional = np.linalg.inv(Fisher_P_Traditional)
CovXi_traditional = np.linalg.inv(Fisher_Xi_Traditional)
CovP_unwindowed = Window.Quadratic_Cov(unwindowed_P, unwindowed_dCdp) #np.linalg.inv(Fisher_P_unwindowed)

CrossP = CrossCoeff( CovP )
CrossXi = CrossCoeff( CovXi )
Contour_plot( Window.kcenter, CovP, title='Cov_P ', pdfname='Cov_P.pdf' )
Contour_plot( Window.kcenter, CrossP, title='corr(Cov_P) ', pdfname='corr_Cov_P.pdf' )
Contour_plot( Window.rcenter, CovXi, title='Cov_Xi ', pdfname='Cov_Xi.pdf', basename = 'log10(r)' )
Contour_plot( Window.rcenter, CrossXi, title='corr(Cov_Xi) ', pdfname='corr_Cov_Xi.pdf', basename = 'log10(r)' )

DataCov = Window.PowerCovMatrix()
Sigma_Pdata = np.sqrt(DataCov.diagonal())
Sigma_P = np.sqrt(CovP.diagonal())
Sigma_Xi = np.sqrt(CovXi.diagonal())
Fractional_Pdata = Sigma_Pdata/Window.Pmlist
Fractional_P = Sigma_P/Window.Pmlist
Fractional_Xi = Sigma_Xi/Window.Pmlist
Fractional_P_Traditional = np.sqrt(CovP_traditional.diagonal())/Window.Pmlist
Fractional_Xi_Traditional = np.sqrt(CovXi_traditional.diagonal())/Window.Pmlist
Fractional_P_unwindowed = np.sqrt(CovP_unwindowed.diagonal())/Window.Pmlist

makedirectory('plots/window/')
Linear_plot(Window.kcenter, ['No window', 'No window (Quad)', 'P (Quadratic)', 'Xi(Quadratic)' ,'P (Traditional)'], Fractional_Pdata, Fractional_P_unwindowed, Fractional_P, Fractional_Xi, Fractional_P_Traditional, scale = 'log', xmin = 10**(-3), xmax = 10**(3), ymin = 10**(-5), ymax = 1., pdfname='plots/window/frac_kN{}_rN{}_R{}.pdf'.format(kN, rN, Window.RMAX) )

