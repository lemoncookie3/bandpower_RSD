#===========================================================================
#
#   RSD bandpower fractional error and contour ellipse for b and f
#
#   This program calculates fractional errors on bandpower and parameter b and f
#
#   Correlation multipoles are obtained from
#   Redshift distortion power spectrum including Finger of God
#   P (k,mu) = Pm(k) (b+f*mu^2)^2 D(k,mu,s)
#   Pm(k) is linear matter power spectrum data obtained from camb (matterpower_z_0.55.dat)
#
#   * 8-25-2015
#   * Calling function :
#
#   * Output data is stored in ""
#   * Plot :
#
#==========================================================================
import numpy as np
from numpy import zeros, sqrt, pi
from numpy.linalg import inv
from numpy import vectorize
from scipy.interpolate import interp1d
from covariance_class2 import *

import pp, sys, time
from multiprocessing import Process, Queue

#   Initialization

print '\n estimator 2'

KMIN = 0.0001
KMAX = 502.32
    
# r scale
RMIN = 0.1 # 6. #29. # 24. #0.1 for Reid   #1.15 * np.pi / self.kmax
RMAX = 180. #200.  #628.32 # for Reid  #1.15 * np.pi / self.kmin
    
# REID (0.01~ 180) corresponding k value :(0.02 ~ 361.28)
# REID convergence condition : kN = 61, rN = 151, subN = 101
# REID convergence condition : kN = 101, rN = 101, subN = 121
kN = 61  #converge perfectly at 151, 2by2 components converge at 121 # the number of k bins. the sample should be odd
rN = 601  #101 for Reid # number of r bins
subN = 101 #101 for Reid

RSDPower = RSD_covariance(KMIN, KMAX, RMIN, RMAX, kN, rN, subN)
file = open('matterpower_z_0.55.dat')
#file = open('matterpower.dat')
RSDPower.MatterPower(file)
RSDPower.Shell_avg_band()

    
rcut_max = len(RSDPower.rcenter)-1
rcut_min = 0 #25#len(RSDPower.rcenter)-1
kcut_min = 0 #12 # 45 #150 #90#104
kcut_max = len(RSDPower.kmax)-1 #24  #90
    
# 21,31 (61) 45, 73 (151) for BAO only (k : 0.02 ~ 0.3)
# 18, 29(61) for BAO+RSD (k: 0.01 ~ 0.2))
# 12, 24(41) for big scale and small scale
# 21, __(61) for Reid range # REID (0.01~ 180) corresponding k value :(0.02 ~ 361.28)
    
    
print "kcut_min :", RSDPower.kmin[kcut_min], "  kcut_max :", RSDPower.kmax[kcut_max]
print "rcut_min :", RSDPower.rmin[rcut_max], "  rcut_max :", RSDPower.rmax[rcut_min]


def main():


    #multipole band ================================================================
    # shall averaged in each k bin
    
    multipole_bandpower0 = RSDPower.multipole_P(0.0)
    multipole_bandpower2 = RSDPower.multipole_P(2.0)
    multipole_bandpower4 = RSDPower.multipole_P(4.0)
    multipole_bandpower = np.concatenate((multipole_bandpower0,multipole_bandpower2,multipole_bandpower4), axis=0)

    # multipole derivatives ========================================================
    
    
    ## devP
    RSDPower.RSDband_derivative_P_All()
    # dPb0, dPf0, dPs0, dPb2, dPf2, dPs2, dPb4, dPf4, dPs4 = RSDPower.RSDband_derivative_P_All()

    RSDPower.derivative_Xi_band_All()
    #dxip0, dxip2, dxip4 = RSDPower.derivative_Xi_band_All()

    RSDPower.derivative_Xi_All()
    #RSDPower.RSDband_derivative_xi_All()

    Compare_Derivatives()
    
    
    #covariance matrix============================================================

    ##calling function for covariance matrix components Cp_ll'
    #RSDPower.RSDband_covariance_PP_All()

    #RSDPower.RSDband_covariance_PXi_All()
    ## parallel python applied.
    ## check! it could not be retrieved by order
    
    
    start_time = time.time()
    RSDPower.RSD_covariance_Allmodes()
    print "elapsed time :", time.time()-start_time
    
    matricesXi = [RSDPower.covariance00, RSDPower.covariance02, RSDPower.covariance04, np.transpose(RSDPower.covariance02), RSDPower.covariance22, RSDPower.covariance24,np.transpose(RSDPower.covariance04), np.transpose(RSDPower.covariance24), RSDPower.covariance44]
    print np.sum(matricesXi)
    


    #error_Total( kcut_min ,kcut_max )
    #stop
    
    #-------------------------------------------------------
    # step 2 k loop
    """
    step2_process = 2
    print "multi_processing for k loop : ", step2_process, " workers"
    
    numberlist_k = np.arange(1,len(RSDPower.kcenter)-1)
    numberlist_k_split = np.array_split(numberlist_k, step2_process)
    
    def Reid_step2_loop(q, order, input):
        rr, error_b_determin, error_f_determin, error_b_marginal, error_f_marginal, error_b_determin2, error_f_determin2, error_b_marginal2, error_f_marginal2 = Reid_step2(input)
        DAT = np.array(np.concatenate((rr, error_b_determin, error_f_determin, error_b_marginal, error_f_marginal, error_b_determin2, error_f_determin2, error_b_marginal2, error_f_marginal2), axis = 0)).reshape(9,len(input))
        q.put(( order, DAT ))
    
    loop_queue = Queue()
    loop_processes = [Process(target=Reid_step2_loop, args=(loop_queue, z[0], z[1])) for z in zip(range(step2_process+1), numberlist_k_split)]
    for p in loop_processes:
        p.start()
    for p in loop_processes:
        p.join()
    loop_result = [loop_queue.get() for p in loop_processes]
    loop_result.sort()
    loop_result_list = [ loop[1] for loop in loop_result ]
    loops = loop_result_list[0]
    for i in range(1, step2_process):
        loops = np.concatenate((loops, loop_result_list[i]), axis =1 )
    
    rr = loops[0]
    error_b_determin = loops[1]
    error_f_determin = loops[2]
    error_b_marginal = loops[3]
    error_f_marginal = loops[4]
    error_b_determin2 = loops[5]
    error_f_determin2 = loops[6]
    error_b_marginal2 = loops[7]
    error_f_marginal2 = loops[8]


    output_b_pdf = 'plots/Reid/TwoFractionalError_b_dlnr{:>1.2f}_dlnk{:>1.2f}.pdf'.format(RSDPower.dlnr, RSDPower.dlnk)
    output_f_pdf = 'plots/Reid/TwoFractionalError_f_dlnr{:>1.2f}_dlnk{:>1.2f}.pdf'.format(RSDPower.dlnr, RSDPower.dlnk)
    plot_title = 'RSDFractional Error on b and f (Two Step)\n z=0.55, dlnr = {:>1.3f}, dlnk = {:>1.3f}, kN = {}, rN = {}'.format(RSDPower.dlnr, RSDPower.dlnk, RSDPower.n, RSDPower.n2)

    Linear_plot( rr, ['det', 'marg', 'det2', 'marg2'], error_b_determin, error_b_marginal, error_b_determin2, error_b_marginal2, pdfname = output_b_pdf, title = plot_title, xmin=0.0, xmax=60.0, ymin=0.0005, ymax=5*10**(-2), scale='semilogy' )
    Linear_plot( rr, ['det', 'marg', 'det2', 'marg2'], error_f_determin, error_f_marginal, error_f_determin2, error_f_marginal2, pdfname = output_f_pdf, title = plot_title, xmin=0.0, xmax=60.0, ymin=0.0, ymax=0.07 )
    """
    #-------------------------------------------------------
    # step 1 r loop

    step1_process = 12
    print "multi_processing for r loop : ", step1_process, " workers"
    
    numberlist_r = np.arange(1,len(RSDPower.rcenter)-1)
    numberlist_r_split = np.array_split(numberlist_r, step1_process)
    

    def Reid_step1_loop(q, order, input):
        rr, error_b_determin, error_f_determin, error_b_marginal, error_f_marginal, error_b_determin2, error_f_determin2, error_b_marginal2, error_f_marginal2 = Reid_step1(input)
        DAT = np.array(np.concatenate((rr, error_b_determin, error_f_determin, error_b_marginal, error_f_marginal, error_b_determin2, error_f_determin2, error_b_marginal2, error_f_marginal2), axis = 0)).reshape(9,len(input))
        q.put(( order, DAT ))
    
    loop_queue = Queue()
    loop_processes = [Process(target=Reid_step1_loop, args=(loop_queue, z[0], z[1])) for z in zip(range(step1_process), numberlist_r_split)]
    for p in loop_processes:
        p.start()
    for p in loop_processes:
        p.join()
    loop_result = [loop_queue.get() for p in loop_processes]
    loop_result.sort()
    loop_result_list = [ loop[1] for loop in loop_result ]
    loops = loop_result_list[0]
    for i in range(1, step1_process):
        loops = np.concatenate((loops, loop_result_list[i]), axis =1 )

    rr = loops[0]
    error_b_determin = loops[1]
    error_f_determin = loops[2]
    error_b_marginal = loops[3]
    error_f_marginal = loops[4]
    error_b_determin2 = loops[5]
    error_f_determin2 = loops[6]
    error_b_marginal2 = loops[7]
    error_f_marginal2 = loops[8]
    
    
    output_b_pdf2 = 'plots/Reid/OneFractionalError_b_dlnr{:>1.2f}_dlnk{:>1.2f}.pdf'.format(RSDPower.dlnr, RSDPower.dlnk)
    output_f_pdf2 = 'plots/Reid/OneFractionalError_f_dlnr{:>1.2f}_dlnk{:>1.2f}.pdf'.format(RSDPower.dlnr, RSDPower.dlnk)
    plot_title2 = 'RSDFractional Error on b and f (One Step)\n z=0.55, dlnr = {:>1.3f}, dlnk = {:>1.3f}, kN = {}, rN = {}'.format(RSDPower.dlnr, RSDPower.dlnk, RSDPower.n, RSDPower.n2)
    
    Linear_plot( rr, ['det', 'marg', 'det2', 'marg2'], error_b_determin, error_b_marginal, error_b_determin2, error_b_marginal2, pdfname = output_b_pdf2, title = plot_title2, xmin=0.0, xmax=60.0, ymin=0.0005, ymax=5*10**(-2), scale='semilogy' )
    Linear_plot( rr, ['det', 'marg', 'det2', 'marg2'], error_f_determin, error_f_marginal, error_f_determin2, error_f_marginal2, pdfname = output_f_pdf2, title = plot_title2, xmin=0.0, xmax=60.0, ymin=0.0, ymax=0.07 )



def error():
    
    l1 = len(RSDPower.kcenter)-1
    l2 = len(RSDPower.rcenter)-1
    
    matricesPP_total = [covariance_PP00, covariance_PP02, covariance_PP04,covariance_PP02, covariance_PP22, covariance_PP24,covariance_PP04, covariance_PP24, covariance_PP44]
    """
    matricesPXi = [covariance_PXi00, covariance_PXi02, covariance_PXi04, covariance_PXi20, covariance_PXi22, covariance_PXi24,covariance_PXi40, covariance_PXi42, covariance_PXi44]
    matricesXi = [covariance00, covariance02, covariance04, np.transpose(covariance02), covariance22, covariance24,np.transpose(covariance04), np.transpose(covariance24), covariance44]
    """
    
    
    matricesPP = [covariance_PP00[kcut_min:kcut_max+1,kcut_min:kcut_max+1],\
                  covariance_PP02[kcut_min:kcut_max+1,kcut_min:kcut_max+1], \
                  covariance_PP04[kcut_min:kcut_max+1,kcut_min:kcut_max+1], \
                  covariance_PP02[kcut_min:kcut_max+1,kcut_min:kcut_max+1],\
                  covariance_PP22[kcut_min:kcut_max+1,kcut_min:kcut_max+1], \
                  covariance_PP24[kcut_min:kcut_max+1,kcut_min:kcut_max+1],\
                  covariance_PP04[kcut_min:kcut_max+1,kcut_min:kcut_max+1], \
                  covariance_PP24[kcut_min:kcut_max+1,kcut_min:kcut_max+1], \
                  covariance_PP44[kcut_min:kcut_max+1,kcut_min:kcut_max+1]]
    matricesPXi = [covariance_PXi00[kcut_min:kcut_max+1,rcut_min:rcut_max+1], \
                   covariance_PXi02[kcut_min:kcut_max+1,rcut_min:rcut_max+1],\
                   covariance_PXi04[kcut_min:kcut_max+1,rcut_min:rcut_max+1],\
                   covariance_PXi20[kcut_min:kcut_max+1,rcut_min:rcut_max+1], \
                   covariance_PXi22[kcut_min:kcut_max+1,rcut_min:rcut_max+1], \
                   covariance_PXi24[kcut_min:kcut_max+1,rcut_min:rcut_max+1],\
                   covariance_PXi40[kcut_min:kcut_max+1,rcut_min:rcut_max+1],\
                   covariance_PXi42[kcut_min:kcut_max+1,rcut_min:rcut_max+1],\
                   covariance_PXi44[kcut_min:kcut_max+1,rcut_min:rcut_max+1]]
    matricesXi = [covariance00[rcut_min:rcut_max+1,rcut_min:rcut_max+1],\
                  covariance02[rcut_min:rcut_max+1,rcut_min:rcut_max+1], \
                  covariance04[rcut_min:rcut_max+1,rcut_min:rcut_max+1],\
                  np.transpose(covariance02[rcut_min:rcut_max+1,rcut_min:rcut_max+1]), \
                  covariance22[rcut_min:rcut_max+1,rcut_min:rcut_max+1],\
                  covariance24[rcut_min:rcut_max+1,rcut_min:rcut_max+1],\
                  np.transpose(covariance04[rcut_min:rcut_max+1,rcut_min:rcut_max+1]),\
                  np.transpose(covariance24[rcut_min:rcut_max+1,rcut_min:rcut_max+1]),\
                  covariance44[rcut_min:rcut_max+1,rcut_min:rcut_max+1]]
    
    matrices2P_cut = [dPb0[kcut_min:kcut_max+1], dPb2[kcut_min:kcut_max+1],\
                      dPb4[kcut_min:kcut_max+1], dPf0[kcut_min:kcut_max+1], \
                      dPf2[kcut_min:kcut_max+1], dPf4[kcut_min:kcut_max+1], \
                      dPs0[kcut_min:kcut_max+1], dPs2[kcut_min:kcut_max+1],\
                      dPs4[kcut_min:kcut_max+1]]
                  
    matrices2P = [dPb0, dPb2, dPb4, dPf0, dPf2, dPf4, dPs0, dPs2, dPs4]
    
    
    
    #   derivative matrices
    derivative_P0 = np.identity(len(RSDPower.kcenter))[:,kcut_min:kcut_max+1]
    Pzeros = np.zeros((len(RSDPower.kcenter),kcut_max-kcut_min+1))
    derivative_P = np.concatenate((np.concatenate((derivative_P0, Pzeros, Pzeros),axis=1 ),\
                                   np.concatenate((Pzeros, derivative_P0,Pzeros),axis=1 ),\
                                   np.concatenate((Pzeros, Pzeros,derivative_P0),axis=1 )), axis=0)
    Xizeros = np.zeros((len(RSDPower.kcenter),rcut_max - rcut_min + 1))

    derivative_correl_avg = np.concatenate(( np.concatenate((dxip0,Xizeros,Xizeros), axis=1),\
                                            np.concatenate((Xizeros,dxip2,Xizeros), axis=1),\
                                            np.concatenate((Xizeros,Xizeros,dxip4), axis=1)),axis=0 )
                                            
    Derivatives = np.concatenate((derivative_P,derivative_correl_avg), axis=1)

    

    # for 3 modes
    C_matrix3PP = CombineCovariance3(l1, matricesPP)
    C_matrix3PXi = CombineCrossCovariance3(l1, l2, matricesPXi)
    #C_matrix3PXi = np.zeros((3 * len(RSDPower.kcenter), 3 * len(RSDPower.rcenter)))
    C_matrix3Xi = CombineCovariance3(l2, matricesXi)
    C_matrix3PP_total = CombineCovariance3(l1, matricesPP_total)
    
    zeromatrix = np.zeros(( 3 * (kcut_max-kcut_min + 1) , 3 * (rcut_max-rcut_min + 1 )))
    C_matrix3 = np.concatenate((np.concatenate((C_matrix3PP, C_matrix3PXi), axis=1),np.concatenate((np.transpose(C_matrix3PXi), C_matrix3Xi), axis=1)), axis = 0)
    C_matrix3_nf = np.concatenate((np.concatenate((C_matrix3PP, zeromatrix), axis=1),np.concatenate((np.transpose(zeromatrix), C_matrix3Xi), axis=1)), axis = 0)


    # for 2 modes

    derivative_P2 = np.concatenate((np.concatenate((derivative_P0, Pzeros),axis=1 ),\
                                    np.concatenate((derivative_P0,Pzeros),axis=1 )), axis=0)
    derivative_correl_avg2 = np.concatenate(( np.concatenate((dxip0,Xizeros), axis=1),\
                                            np.concatenate((dxip2,Xizeros), axis=1)),axis=0 )
    Derivatives2 = np.concatenate((derivative_P2,derivative_correl_avg2), axis=1)
    C_matrix2PP = CombineCovariance2(l1, matricesPP)
    C_matrix2PXi = CombineCrossCovariance2(l1, l2, matricesPXi)
    #C_matrix2PXi = np.zeros((2 * len(RSDPower.kcenter), 2 * len(RSDPower.rcenter)))
    C_matrix2Xi = CombineCovariance2(l2, matricesXi)
    C_matrix2PP_total = CombineCovariance2(l1, matricesPP_total)
    
    zeromatrix2 = np.zeros(( 2 * (kcut_max-kcut_min + 1) , 2 * (rcut_max-rcut_min + 1 )))
    C_matrix2 = np.concatenate((np.concatenate((C_matrix2PP, C_matrix2PXi), axis=1),np.concatenate((np.transpose(C_matrix2PXi), C_matrix2Xi), axis=1)), axis = 0)
    C_matrix2_nf = np.concatenate((np.concatenate((C_matrix2PP, zeromatrix2), axis=1),np.concatenate((np.transpose(zeromatrix2), C_matrix2Xi), axis=1)), axis = 0)

    
    
    """ multipole band power covariance obtained from C_tot"""
    # marginalized s, three modes
    Fisher_bandpower_P = inv(C_matrix3PP_total)
    Fisher_bandpower_Xi = FisherProjection(derivative_correl_avg, C_matrix3Xi)
    Fisher_bandpower = FisherProjection(Derivatives, C_matrix3)
    Fisher_bandpower_nf = FisherProjection(Derivatives, C_matrix3_nf)

    # marginalized s, two modes
    Fisher_bandpower_P2 = inv(C_matrix2PP_total)
    Fisher_bandpower_Xi2 = FisherProjection(derivative_correl_avg2, C_matrix2Xi)
    Fisher_bandpower2 = FisherProjection(Derivatives2, C_matrix2)
    Fisher_bandpower2_nf = FisherProjection(Derivatives2, C_matrix2_nf)
    

    bandpower_base = multipole_bandpower #[0:len(RSDPower.kcenter)+1]
    
    """
    k_base = np.concatenate((RSDPower.kcenter,RSDPower.kcenter,RSDPower.kcenter),axis=1)
    r_base = np.concatenate((RSDPower.rcenter,RSDPower.rcenter,RSDPower.rcenter),axis=1)
    
    Contour_plot( RSDPower.kcenter, CrossCoeff( k_base, C_matrix3PP_total), pdfname='CPP.pdf' )
    Contour_plot( RSDPower.kcenter, CrossCoeff( k_base, inv(Fisher_bandpower_Xi)), pdfname='CXi.pdf' )
    Contour_plot( RSDPower.kcenter, CrossCoeff( k_base, inv(Fisher_bandpower)), pdfname='Ctot.pdf' )
    Contour_plot( RSDPower.kcenter, CrossCoeff( k_base, inv(Fisher_bandpower_nf)), pdfname='Cnf.pdf' )
    """

    """ fractional error for bandpower 00 """
    error_P = FractionalErrorBand( bandpower_base, C_matrix3PP_total)[0:len(RSDPower.kcenter)]
    error_Xi = FractionalErrorBand( bandpower_base, inv(Fisher_bandpower_Xi))[0:len(RSDPower.kcenter)]
    error_Ctot = FractionalErrorBand( bandpower_base, inv(Fisher_bandpower))[0:len(RSDPower.kcenter)]
    error_Cnf = FractionalErrorBand( bandpower_base, inv(Fisher_bandpower_nf))[0:len(RSDPower.kcenter)]


    """ bandpower error linear plots """
    makedirectory('plots/RSD/band_error')
    title = 'Fractional Error, l=0 \n  k = ({:>3.4f}, {:>3.4f}), r = ({:>0.4f}, {:>6.2f}) \n dlnr = {:>3.4f}, dlnk = {:>3.4f}, sdlnk = {:>3.4}'.format(RSDPower.kmin[kcut_min], RSDPower.kmax[kcut_max], RSDPower.rmax[rcut_min], RSDPower.rmin[rcut_max], RSDPower.dlnr, RSDPower.dlnk, RSDPower.sdlnk[2])
    pdfname = 'plots/RSD/band_error/fractional00_k{:>3.4f}_{:>3.4f}_r_{:>3.4f}_{:>3.4f}_rN{}kN{}.pdf'.format(RSDPower.kmin[kcut_min], RSDPower.kmax[kcut_max], RSDPower.rmax[rcut_min], RSDPower.rmin[rcut_max], RSDPower.n2, RSDPower.n)
    pdfname2 = 'plots/RSD/band_error/mag_fractional00_k{:>3.4f}_{:>3.4f}_r_{:>3.4f}_{:>3.4f}_rN{}kN{}.pdf'.format(RSDPower.kmin[kcut_min], RSDPower.kmax[kcut_max], RSDPower.rmax[rcut_min], RSDPower.rmin[rcut_max], RSDPower.n2, RSDPower.n)
    """
    # full plot
    Linear_plot( RSDPower.kcenter,['P','Xi','C_tot', 'C_nf'], error_P ,error_Xi,error_Ctot, error_Cnf, title = title, pdfname = pdfname, ymax = 10**11, ymin = 10**(-5), xmax = 10**(3), xmin = 10**(-5), scale='log' )
    
    # magnified plot
    Linear_plot2( RSDPower.kcenter,['P','Xi','C_tot', 'C_nf'], error_P, error_Xi, error_Ctot, error_Cnf, title = title, pdfname = pdfname2, ymax = 10., ymin = 10**(-3), xmax = 10., xmin = 10**(-3), scale='log')
    """
    
    # full plot
    Linear_plot2( RSDPower.kcenter,  RSDPower.kcenter[kcut_min:kcut_max+1], error_P[kcut_min:kcut_max+1], ['P','Xi','C_tot', 'C_nf'], error_P ,error_Xi,error_Ctot, error_Cnf, title = title, pdfname = pdfname, ymax = 10**11, ymin = 10**(-5), xmax = 10**(3), xmin = 10**(-5), scale='log' )
    
    # magnified plot
    Linear_plot2( RSDPower.kcenter,  RSDPower.kcenter[kcut_min:kcut_max+1], error_P[kcut_min:kcut_max+1], ['P','Xi','C_tot', 'C_nf'], error_P, error_Xi, error_Ctot, error_Cnf, title = title, pdfname = pdfname2, ymax = 10., ymin = 10**(-3), xmax = 10., xmin = 10**(-3), scale='log')
    

    XP, XP2 = CombineDevXi(l1, matrices2P)
    XP_cut, XP2_cut = CombineDevXi(l1, matrices2P_cut)
    
    """ projection to b and f space"""
    # Fisher, all modes, s marginalized
    FisherPP = FisherProjection(XP_cut, C_matrix3PP)
    FisherXi = FisherProjection(XP, inv(Fisher_bandpower_Xi))
    FisherC_nf = FisherProjection(XP, inv(Fisher_bandpower_nf))
    FisherC = FisherProjection(XP, inv(Fisher_bandpower))
    
    # Fisher, two modes, s marginalized
    FisherPP2 = FisherProjection(XP2_cut, C_matrix2PP)
    FisherXi2 = FisherProjection(XP2, inv(Fisher_bandpower_Xi2))
    FisherC2_nf = FisherProjection(XP2, inv(Fisher_bandpower2_nf))
    FisherC2 = FisherProjection(XP2, inv(Fisher_bandpower2))
    
    # Fisher, all modes, s determined
    FisherPP_d = FisherProjection(XP_cut[0:2,:], C_matrix3PP)
    FisherXi_d = FisherProjection(XP[0:2,:], inv(Fisher_bandpower_Xi))
    FisherC_nf_d = FisherProjection(XP[0:2,:], inv(Fisher_bandpower_nf))
    FisherC_d = FisherProjection(XP[0:2,:], inv(Fisher_bandpower))

    
    Cov_PP = inv(FisherPP)
    Cov_Xi = inv(FisherXi)
    Cov_C_nooff = inv(FisherC_nf)
    Cov_C = inv(FisherC)
    
    Cov_PP2 = inv(FisherPP2)
    Cov_Xi2 = inv(FisherXi2)
    Cov_C2_nooff = inv(FisherC2_nf)
    Cov_C2 = inv(FisherC2)

    Cov_PP_d = inv(FisherPP_d)
    Cov_Xi_d = inv(FisherXi_d)
    Cov_C_nooff_d = inv(FisherC_nf_d)
    Cov_C_d = inv(FisherC_d)  



  
    print '- - - - - - - - - - - - - - - - -'
    print '  Cov_PP :', np.sum(Cov_PP)
    print '  Cov_Xi :', np.sum(Cov_Xi)
    print '  Cov_tot:', np.sum(Cov_C)
    print '  Cov_nf :', np.sum(Cov_C_nooff)
    print '- - - - - - - - - - - - - - - - -'
    print Cov_PP[0:2,0:2], Cov_Xi[0:2,0:2], Cov_C_nooff[0:2,0:2], Cov_C[0:2,0:2]
    print '- - - - - - - - - - - - - - - - -'
    
    makedirectory('plots/RSD/ellipse')
    # ellipse, all modes, marginalized
    title = 'Confidence Ellipse for b and f, l = 0,2,4 \n rmin={:>3.5f} rmax={:>3.2f} kmin={:>3.5f} kmax={:>3.2f}'.format(RSDPower.rmin[rcut_max],RSDPower.rmax[rcut_min],RSDPower.kmin[kcut_min],RSDPower.kmax[kcut_max] )
    pdfname = 'plots/RSD/ellipse/ConfidenceEllipseTotal_r{:>3.2f}_{:>3.2f}_k{:>3.2f}_{:>3.2f}_rN{}kN{}.pdf'.format(RSDPower.rmin[rcut_max],RSDPower.rmax[rcut_min],RSDPower.kmin[kcut_min],RSDPower.kmax[kcut_max], RSDPower.n2, RSDPower.n )
    labellist = ['$C_{P}$','$C_{Xi}$','$C_{nooff}$','$C_{total}$']
    confidence_ellipse(RSDPower.b, RSDPower.f, labellist, Cov_PP[0:2,0:2], Cov_Xi[0:2,0:2], Cov_C_nooff[0:2,0:2], Cov_C[0:2,0:2], title = title, pdfname = pdfname)
    # ellipse, two modes, marginalized
    title2 = 'Confidence Ellipse for b and f, l = 0,2 \n rmin={:>3.5f} rmax={:>3.2f} kmin={:>3.5f} kmax={:>3.2f}'.format(RSDPower.rmin[rcut_max],RSDPower.rmax[rcut_min],RSDPower.kmin[kcut_min],RSDPower.kmax[kcut_max] )
    pdfname2 = 'plots/RSD/ellipse/ConfidenceEllipseTwomodes_r{:>3.2f}_{:>3.2f}_k{:>3.2f}_{:>3.2f}_rN{}kN{}.pdf'.format(RSDPower.rmin[rcut_max],RSDPower.rmax[rcut_min],RSDPower.kmin[kcut_min],RSDPower.kmax[kcut_max], RSDPower.n2, RSDPower.n )
    confidence_ellipse(RSDPower.b, RSDPower.f, labellist, Cov_PP2[0:2,0:2], Cov_Xi2[0:2,0:2], Cov_C2_nooff[0:2,0:2], Cov_C2[0:2,0:2], title = title2, pdfname = pdfname2)
    # ellipse, all modes, determined
    title3 = 'Confidence Ellipse for b and f, l = 0,2,4  ($\sigma$ determined) \n rmin={:>3.5f} rmax={:>3.2f} kmin={:>3.5f} kmax={:>3.2f}'.format(RSDPower.rmin[rcut_max],RSDPower.rmax[rcut_min],RSDPower.kmin[kcut_min],RSDPower.kmax[kcut_max] )
    pdfname3 = 'plots/RSD/ellipse/ConfidenceEllipseTotal_magi_r{:>3.2f}_{:>3.2f}_k{:>3.2f}_{:>3.2f}_rN{}kN{}.pdf'.format(RSDPower.rmin[rcut_max],RSDPower.rmax[rcut_min],RSDPower.kmin[kcut_min],RSDPower.kmax[kcut_max], RSDPower.n2, RSDPower.n )
    labellist = ['$C_{P}$','$C_{Xi}$','$C_{nooff}$','$C_{total}$']
    confidence_ellipse(RSDPower.b, RSDPower.f, labellist, Cov_PP_d, Cov_Xi_d, Cov_C_nooff_d, Cov_C_d, title = title3, pdfname = pdfname3)

def Only_ellipse():

    #   Only BAO
    
    Cov_PP = np.array([[  6.86832133e-06,  -1.33496438e-05], [ -1.33496438e-05,   9.66966052e-05]])
    Cov_Xi = np.array([[  9.37043305e-05,  -8.41838735e-05], [ -8.41838735e-05,   2.43455137e-04]])
    Cov_C_nooff = np.array([[  6.84923001e-06,  -1.13472249e-05], [ -1.13472249e-05,   3.32219203e-05]])
    Cov_C = np.array([[  6.34306466e-06,  -1.02788983e-05], [ -1.02788983e-05,   3.08986048e-05]])
    
    
    #   RSD+BAO
    """
    Cov_PP = np.array([[  1.62180913e-05,  -3.31981202e-05], [ -3.31981202e-05,   3.01064140e-04]])
    Cov_Xi = np.array([[  7.44753835e-05,  -2.73763795e-05], [ -2.73763795e-05,   1.50450214e-04]])
    Cov_C_nooff = np.array([[  1.25579867e-05,  -1.87313793e-05], [ -1.87313793e-05,   5.89167036e-05]])
    Cov_C = np.array([[  1.50627641e-05,  -2.46236785e-05], [ -2.46236785e-05,   7.15578712e-05]])
    """


    title = 'Confidence Ellipse for b and f, l = 0,2,4 \n rmin={:>3.5f} rmax={:>3.2f} kmin={:>3.5f} kmax={:>3.2f}'.format(RSDPower.rmin[rcut_max],RSDPower.rmax[rcut_min],RSDPower.kmin[kcut_min],RSDPower.kmax[kcut_max] )
    pdfname = 'plots/RSD/ellipse/ConfidenceEllipseTotal_r{:>3.2f}_{:>3.2f}_k{:>3.2f}_{:>3.2f}_rN{}kN{}.pdf'.format(RSDPower.rmin[rcut_max],RSDPower.rmax[rcut_min],RSDPower.kmin[kcut_min],RSDPower.kmax[kcut_max], RSDPower.n2, RSDPower.n )
    labellist = ['$C_{P}$','$C_{Xi}$','$C_{nooff}$','$C_{total}$']
    confidence_ellipse(RSDPower.b, RSDPower.f, labellist, Cov_PP, Cov_Xi, Cov_C_nooff, Cov_C, title = title, pdfname = pdfname)

def error_only_CP(l):
    
    matricesPP = [RSDPower.covariance_PP00, RSDPower.covariance_PP02, RSDPower.covariance_PP04,RSDPower.covariance_PP02, RSDPower.covariance_PP22, RSDPower.covariance_PP24,RSDPower.covariance_PP04, RSDPower.covariance_PP24, RSDPower.covariance_PP44]
    matrices2P = [RSDPower.dPb0, RSDPower.dPb2, RSDPower.dPb4, RSDPower.dPf0, RSDPower.dPf2, RSDPower.dPf4, RSDPower.dPs0, RSDPower.dPs2, RSDPower.dPs4]
    
    C_matrix3 = CombineCovariance3(l, matricesPP)
    C_matrix2 = CombineCovariance2(l, matricesPP)

    Xi, Xi2 = CombineDevXi(l, matrices2P)

    Fisher_marginal = FisherProjection(Xi, C_matrix3)
    Fisher_determin = Fisher_marginal[0:2,0:2] #FisherProjection(Xi2, C_matrix2)
    Fisher_marginal2 = FisherProjection(Xi2, C_matrix2)
    Fisher_determin2 = Fisher_marginal2[0:2,0:2] #FisherProjection(Xi2, C_matrix2)
    
    """
    Cross = CrossCoeff( C_matrix3 )
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    plt.imshow( Cross )
    plt.show()
    """
    
    Cov_marginal = inv(Fisher_marginal)
    Cov_determin = inv(Fisher_determin)
    Cov_marginal2 = inv(Fisher_marginal2)
    Cov_determin2 = inv(Fisher_determin2)

    error_b_marginal, error_f_marginal = FractionalError( RSDPower.b, RSDPower.f, Cov_marginal)
    error_b_determin, error_f_determin = FractionalError( RSDPower.b, RSDPower.f, Cov_determin)
    error_b_marginal2, error_f_marginal2 = FractionalError( RSDPower.b, RSDPower.f, Cov_marginal2)
    error_b_determin2, error_f_determin2 = FractionalError( RSDPower.b, RSDPower.f, Cov_determin2)
    
    rr = 1.15 * np.pi/RSDPower.kcenter[l]
    return rr, error_b_determin, error_f_determin, error_b_marginal, error_f_marginal, error_b_determin2, error_f_determin2, error_b_marginal2, error_f_marginal2

def error_only_Cxi(l):
    matricesXi = [covariance00, covariance02, covariance04, np.transpose(covariance02), covariance22, covariance24,np.transpose(covariance04), np.transpose(covariance24), covariance44]
    
    Xizeros = np.zeros((len(RSDPower.kcenter),rcut_max - rcut_min + 1))
    matrices2P = [dPb0, dPb2, dPb4, dPf0, dPf2, dPf4, dPs0, dPs2, dPs4]
    matrices2Xi = [dxip0, Xizeros,Xizeros,Xizeros,dxip2,Xizeros,dxip4,Xizeros,Xizeros]
    
    
    l_det = len(RSDPower.rcenter)-1
    C_matrix3 = CombineCovariance3(l_det, matricesXi)
    C_matrix2 = CombineCovariance2(l_det, matricesXi)

    Xi = np.concatenate(( np.concatenate((dxip0,Xizeros,Xizeros), axis=0),np.concatenate((Xizeros,dxip2,Xizeros), axis=0), np.concatenate((Xizeros,Xizeros,dxip4), axis=0)),axis=1 )
    Xi2 = np.concatenate(( np.concatenate((dxip0,Xizeros), axis=0), np.concatenate((Xizeros,dxip2), axis=0)),axis=1 )
    XP, XP2 = CombineDevXi(l, matrices2P)
    
    Fisher_bandpower3 = FisherProjection(Xi, C_matrix3)[0:l+1,0:l+1]
    Fisher_bandpower2 = FisherProjection(Xi2, C_matrix2)[0:l+1,0:l+1]
    
    Fisher_marginal = FisherProjection(XP, inv(Fisher_bandpower3))
    Fisher_determin = Fisher_marginal[0:2,0:2] #FisherProjection(Xi2, C_matrix2)
    Fisher_marginal2 = FisherProjection(XP2, inv(Fisher_bandpower2))
    Fisher_determin2 = Fisher_marginal2[0:2,0:2] #FisherProjection(Xi2, C_matrix2)
    
    error_b_marginal, error_f_marginal = FractionalError( RSDPower.b, RSDPower.f, inv(Fisher_marginal))
    error_b_determin, error_f_determin = FractionalError( RSDPower.b, RSDPower.f, inv(Fisher_determin))
    error_b_marginal2, error_f_marginal2 = FractionalError( RSDPower.b, RSDPower.f, inv(Fisher_marginal2))
    error_b_determin2, error_f_determin2 = FractionalError( RSDPower.b, RSDPower.f, inv(Fisher_determin2))

    # confidence ellipse
    
    Cov_marginal = inv(Fisher_marginal)
    Cov_determin = inv(Fisher_determin)
    Cov_marginal2 = inv(Fisher_marginal2)
    Cov_determin2 = inv(Fisher_determin2)

    return RSDPower.rcenter[l], error_b_determin, error_f_determin, error_b_marginal, error_f_marginal, error_b_determin2, error_f_determin2, error_b_marginal2, error_f_marginal2

def error_ellipse():
    
    matricesPP = [covariance_PP00, covariance_PP02, covariance_PP04,covariance_PP02, covariance_PP22, covariance_PP24,covariance_PP04, covariance_PP24, covariance_PP44]
    matricesPXi = [covariance_PXi00, covariance_PXi02, covariance_PXi04, covariance_PXi20, covariance_PXi22, covariance_PXi24,covariance_PXi40, covariance_PXi42, covariance_PXi44]
    matricesXi = [covariance00, covariance02, covariance04, np.transpose(covariance02), covariance22, covariance24,np.transpose(covariance04), np.transpose(covariance24), covariance44]
    
    matrices2P = [dPb0, dPb2, dPb4, dPf0, dPf2, dPf4, dPs0, dPs2, dPs4]
    #matrices2Xi = [dxib0, dxib2, dxib4, dxif0, dxif2, dxif4, dxis0, dxis2, dxis4]
    
    l1 = len(RSDPower.kcenter)-1
    l2 = len(RSDPower.kcenter)-1
    
    C_matrix3PP = CombineCovariance3(l1, matricesPP)
    C_matrix3PXi = CombineCrossCovariance3(l1, l2, matricesPXi)
    C_matrix3Xi = CombineCovariance3(l2, matricesXi)
    
    C_matrix2PP = CombineCovariance2(l1, matricesPP)
    C_matrix2PXi = CombineCrossCovariance2(l1, l2, matricesPXi)
    C_matrix2Xi = CombineCovariance2(l2, matricesXi)
    
    C_matrix3 = np.concatenate((np.concatenate((C_matrix3PP, C_matrix3PXi), axis=1),np.concatenate((np.transpose(C_matrix3PXi), C_matrix3Xi), axis=1)), axis = 0)
    C_matrix2 = np.concatenate((np.concatenate((C_matrix2PP, C_matrix2PXi), axis=1),np.concatenate((np.transpose(C_matrix2PXi), C_matrix2Xi), axis=1)), axis = 0)
    
    
    #   derivative matrices
    derivative_P = np.identity(3 * len(RSDPower.kcenter)) #[:,kmin_cut:kmax_cut+1]
    zeros = np.zeros((len(RSDPower.kcenter),len(RSDPower.rcenter)))
    derivative_correl_avg = np.concatenate(( np.concatenate((dxip0,zeros,zeros), axis=0),\
                                            np.concatenate((zeros,dxip2,zeros), axis=0),\
                                            np.concatenate((zeros,zeros,dxip4), axis=0)),axis=1 )
    Derivatives = np.concatenate((derivative_P,derivative_correl_avg), axis=1)
                                            
    derivative_P2 = np.identity(2 * len(RSDPower.kcenter))
    derivative_correl_avg2 = np.concatenate(( np.concatenate((dxip0,zeros), axis=0),\
                                             np.concatenate((zeros,dxip2), axis=0)),axis=1 )
    Derivatives2 = np.concatenate((derivative_P2,derivative_correl_avg2), axis=1)

    """ multipole band power covariance obtained from C_tot"""
    C_bandpower = FisherProjection(Derivatives, C_matrix3)
    C_bandpower2 = FisherProjection(Derivatives2, C_matrix2)

    XP, XP2 = CombineDevXi(l1, matrices2P)
    #XXi, XXi2 = CombineDevXi(l2, matrices2Xi)
    #Xi = np.concatenate((XP, XXi), axis = 1)
    #Xi2 = np.concatenate((XP2, XXi2), axis = 1)
    
    
    """ projection to b and f space"""
    Fisher_marginal = FisherProjection(XP, C_bandpower)
    Fisher_determin = Fisher_marginal[0:2,0:2] #FisherProjection(Xi2, C_matrix2)
    Fisher_marginal2 = FisherProjection(XP2, C_bandpower2)
    Fisher_determin2 = Fisher_marginal2[0:2,0:2] #FisherProjection(Xi2, C_matrix2)
    
    Cov_marginal = inv(Fisher_marginal)
    Cov_determin = inv(Fisher_determin)
    Cov_marginal2 = inv(Fisher_marginal2)
    Cov_determin2 = inv(Fisher_determin2)
    
    covlist = [Cov_marginal[0:2,0:2], Cov_determin, Cov_marginal2[0:2,0:2],Cov_determin2]
    confidence_ellipse(covlist, RSDPower.b, RSDPower.f)
def error_Total( kcut_min ,kcut_max ):
    
    l1 = len(RSDPower.kcenter)-1
    l2 = len(RSDPower.rcenter)-1
    
    import pp, sys, time
    print "\n MultiProcessing Start :"
    
    ppservers = ()
    #ppservers = ("140.254.91.255","",)
    
    if len(sys.argv) > 1:
        ncpus = int(sys.argv[1])
        # Creates jobserver with ncpus workers
        job_server = pp.Server(ncpus, ppservers=ppservers)
    else:
        # Creates jobserver with automatically detected number of workers
        job_server = pp.Server(ppservers=ppservers)
    print "Starting pp with", job_server.get_ncpus(), "workers \n"
    
    
    #calling function for covariance matrix components Cp_ll'
    covariance_PP00 = np.array(RSDPower.RSDband_covariance_PP(0.0,0.0))
    covariance_PP02 = np.array(RSDPower.RSDband_covariance_PP(0.0,2.0))
    covariance_PP04 = np.array(RSDPower.RSDband_covariance_PP(0.0,4.0))
    covariance_PP22 = np.array(RSDPower.RSDband_covariance_PP(2.0,2.0))
    covariance_PP24 = np.array(RSDPower.RSDband_covariance_PP(2.0,4.0))
    covariance_PP44 = np.array(RSDPower.RSDband_covariance_PP(4.0,4.0))
    
    
    inputs1 = ((0.0, 0.0,),(0.0, 2.0,),(0.0, 4.0,),(2.0, 0.0,),(2.0, 2.0,),(2.0, 4.0,),(4.0, 0.0,),(4.0, 2.0,),(4.0, 4.0,))
    jobs1 = [ job_server.submit(RSDPower.RSDband_covariance_PXi, input, (Ll,avgBessel,)) for input in inputs1]
    result1=[]
    for job in jobs1:
        re = job()
        result1.append(re)
    
    covariance_PXi00 = result1[0]
    covariance_PXi02 = result1[1]
    covariance_PXi04 = result1[2]
    covariance_PXi20 = result1[3]
    covariance_PXi22 = result1[4]
    covariance_PXi24 = result1[5]
    covariance_PXi40 = result1[6]
    covariance_PXi42 = result1[7]
    covariance_PXi44 = result1[8]

    
    print "kcut_min :", RSDPower.kmin[kcut_min], "  kcut_max :", RSDPower.kmax[kcut_max]
    
    matricesPP = [covariance_PP00[kcut_min:kcut_max+1,kcut_min:kcut_max+1],\
                  covariance_PP02[kcut_min:kcut_max+1,kcut_min:kcut_max+1], \
                  covariance_PP04[kcut_min:kcut_max+1,kcut_min:kcut_max+1], \
                  covariance_PP02[kcut_min:kcut_max+1,kcut_min:kcut_max+1],\
                  covariance_PP22[kcut_min:kcut_max+1,kcut_min:kcut_max+1], \
                  covariance_PP24[kcut_min:kcut_max+1,kcut_min:kcut_max+1],\
                  covariance_PP04[kcut_min:kcut_max+1,kcut_min:kcut_max+1], \
                  covariance_PP24[kcut_min:kcut_max+1,kcut_min:kcut_max+1], \
                  covariance_PP44[kcut_min:kcut_max+1,kcut_min:kcut_max+1]]
    matricesPXi = [covariance_PXi00[kcut_min:kcut_max+1,rcut_min:rcut_max+1], \
                   covariance_PXi02[kcut_min:kcut_max+1,rcut_min:rcut_max+1],\
                   covariance_PXi04[kcut_min:kcut_max+1,rcut_min:rcut_max+1],\
                   covariance_PXi20[kcut_min:kcut_max+1,rcut_min:rcut_max+1], \
                   covariance_PXi22[kcut_min:kcut_max+1,rcut_min:rcut_max+1], \
                   covariance_PXi24[kcut_min:kcut_max+1,rcut_min:rcut_max+1],\
                   covariance_PXi40[kcut_min:kcut_max+1,rcut_min:rcut_max+1],\
                   covariance_PXi42[kcut_min:kcut_max+1,rcut_min:rcut_max+1],\
                   covariance_PXi44[kcut_min:kcut_max+1,rcut_min:rcut_max+1]]
                   
    matricesXi = [RSDPower.covariance00[rcut_min:rcut_max+1,rcut_min:rcut_max+1],\
                  RSDPower.covariance02[rcut_min:rcut_max+1,rcut_min:rcut_max+1], \
                  RSDPower.covariance04[rcut_min:rcut_max+1,rcut_min:rcut_max+1],\
                  np.transpose(RSDPower.covariance02[rcut_min:rcut_max+1,rcut_min:rcut_max+1]), \
                  RSDPower.covariance22[rcut_min:rcut_max+1,rcut_min:rcut_max+1],\
                  RSDPower.covariance24[rcut_min:rcut_max+1,rcut_min:rcut_max+1],\
                  np.transpose(RSDPower.covariance04[rcut_min:rcut_max+1,rcut_min:rcut_max+1]),\
                  np.transpose(RSDPower.covariance24[rcut_min:rcut_max+1,rcut_min:rcut_max+1]),\
                  RSDPower.covariance44[rcut_min:rcut_max+1,rcut_min:rcut_max+1]]
                  
    matrices2P = [RSDPower.dPb0[kcut_min:kcut_max+1], RSDPower.dPb2[kcut_min:kcut_max+1],\
                  RSDPower.dPb4[kcut_min:kcut_max+1], RSDPower.dPf0[kcut_min:kcut_max+1], \
                  RSDPower.dPf2[kcut_min:kcut_max+1], RSDPower.dPf4[kcut_min:kcut_max+1], \
                  RSDPower.dPs0[kcut_min:kcut_max+1], RSDPower.dPs2[kcut_min:kcut_max+1],\
                  RSDPower.dPs4[kcut_min:kcut_max+1]]
    #matrices2Xi = [dxib0[rcut_min:rcut_max+1], dxib2[rcut_min:rcut_max+1], dxib4[rcut_min:rcut_max+1], dxif0[rcut_min:rcut_max+1], dxif2[rcut_min:rcut_max+1], dxif4[rcut_min:rcut_max+1], dxis0[rcut_min:rcut_max+1], dxis2[rcut_min:rcut_max+1], dxis4[rcut_min:rcut_max+1]]
    
    
    
    C_matrix3PP = CombineCovariance3(l1, matricesPP)
    C_matrix3PXi = CombineCrossCovariance3(l1, l2, matricesPXi)
    C_matrix3Xi = CombineCovariance3(l2, matricesXi)
    C_matrix3PXi_zero = np.zeros((3 * (kcut_max - kcut_min + 1), 3 * (rcut_max - rcut_min + 1)))
    
    C_matrix2PP = CombineCovariance2(l1, matricesPP)
    C_matrix2PXi = CombineCrossCovariance2(l1, l2, matricesPXi)
    C_matrix2Xi = CombineCovariance2(l2, matricesXi)
    C_matrix2PXi_zero = np.zeros((2 * (kcut_max - kcut_min + 1) , 2 * (rcut_max - rcut_min + 1)))
    
    C_matrix3 = np.concatenate((np.concatenate((C_matrix3PP, C_matrix3PXi), axis=1),np.concatenate((np.transpose(C_matrix3PXi), C_matrix3Xi), axis=1)), axis = 0)
    C_matrix2 = np.concatenate((np.concatenate((C_matrix2PP, C_matrix2PXi), axis=1),np.concatenate((np.transpose(C_matrix2PXi), C_matrix2Xi), axis=1)), axis = 0)
    C_matrix3_nooff = np.concatenate((np.concatenate((C_matrix3PP, C_matrix3PXi_zero), axis=1),np.concatenate((np.transpose(C_matrix3PXi_zero), C_matrix3Xi), axis=1)), axis = 0)
                  
                  
    #   derivative matrices
    derivative_P = np.identity(3 * len(RSDPower.kcenter)) #[:,kmin_cut:kmax_cut+1]
    zeros = np.zeros((len(RSDPower.kcenter),len(RSDPower.rcenter)))
    derivative_correl_avg = np.concatenate(( np.concatenate((RSDPower.dxip0,zeros,zeros), axis=0),\
                                            np.concatenate((zeros,RSDPower.dxip2,zeros), axis=0),\
                                            np.concatenate((zeros,zeros,RSDPower.dxip4), axis=0)),axis=1 )
    Derivatives = np.concatenate((derivative_P,derivative_correl_avg), axis=1)
                                                          
    derivative_P2 = np.identity(2 * len(RSDPower.kcenter))
    derivative_correl_avg2 = np.concatenate(( np.concatenate((RSDPower.dxip0,zeros), axis=0),\
                                             np.concatenate((zeros,RSDPower.dxip2), axis=0)),axis=1 )
    Derivatives2 = np.concatenate((derivative_P2,derivative_correl_avg2), axis=1)
                                                                                                   
    """ multipole band power covariance obtained from C_tot """
        #  bandpower_Xi : inv(bandpowerC) from Cxi
        #  bandpower : inv(bandpowerC) from C_tot
        #  bandpower_nooff : inv(banpowerC) from C_nooff
        #   ** all marginalized, three modes
        #
    Fisher_bandpower_Xi = FisherProjection(derivative_correl_avg, C_matrix3Xi)
    Fisher_bandpower = FisherProjection(Derivatives, C_matrix3)
    Fisher_bandpower_nooff = FisherProjection(Derivatives, C_matrix3_nooff)
    #Fisher_bandpower2 = FisherProjection(Derivatives2, C_matrix2)
    #Fisher_bandpower2_nooff = FisherProjection(Derivatives2, C_matrix2_nooff)
    
    XP, XP2 = CombineDevXi(l1, matrices2P) # dP/dq derivative matrices

    """ projection to b and f space"""
    # Fisher, all modes, s marginalized
    FisherPP = FisherProjection(XP, C_matrix3PP)
    FisherXi = FisherProjection(XP, inv(Fisher_bandpower_Xi))
    FisherC_nooff = FisherProjection(XP, inv(Fisher_bandpower_nooff))
    FisherC = FisherProjection(XP, inv(Fisher_bandpower))
    
    Cov_PP = inv(FisherPP)
    Cov_Xi = inv(FisherXi)
    Cov_C_nooff = inv(FisherC_nooff)
    Cov_C = inv(FisherC)


    makedirectory('plots/RSD/ellipse')
    title = 'Confidence Ellipse for b and f, l = 0,2,4 \n rmin={:>3.5f} rmax={:>3.2f} kmin={:>3.5f} kmax={:>3.2f}'.format(RSDPower.rmin[rcut_max],RSDPower.rmax[rcut_min],RSDPower.kmin[kcut_min],RSDPower.kmax[kcut_max] )
    pdfname = 'plots/RSD/ellipse/ConfidenceEllipseTotal_r{:>3.2f}_{:>3.2f}_k{:>3.2f}_{:>3.2f}_rN{}kN{}.pdf'.format(RSDPower.rmin[rcut_max],RSDPower.rmax[rcut_min],RSDPower.kmin[kcut_min],RSDPower.kmax[kcut_max], RSDPower.n2, RSDPower.n )
    labellist = ['$C_{P}$','$C_{Xi}$','$C_{nooff}$','$C_{total}$']
    confidence_ellipse(RSDPower.b, RSDPower.f, labellist, Cov_PP[0:2,0:2], Cov_Xi[0:2,0:2], Cov_C_nooff[0:2,0:2], Cov_C[0:2,0:2], title = title, pdfname = pdfname)

    filename = 'plots/RSD/ellipse/ConfidenceEllipseCov_r{:>3.2f}_{:>3.2f}_k{:>3.2f}_{:>3.2f}.txt'.format(RSDPower.rmin[rcut_max],RSDPower.rmax[rcut_min],RSDPower.kmin[kcut_min],RSDPower.kmax[kcut_max] )
    
    Cov_PP = np.reshape( Cov_PP[0:2,0:2], (1,4))
    Cov_Xi = np.reshape( Cov_Xi[0:2,0:2], (1,4))
    Cov_C_nooff = np.reshape( Cov_C_nooff[0:2,0:2], (1,4))
    Cov_C = np.reshape( Cov_C[0:2,0:2], (1,4))
    
    DAT = np.column_stack(( Cov_PP, Cov_Xi, Cov_C_nooff, Cov_C ))
    np.savetxt(filename, DAT, delimiter=" ", fmt="%s")



def Compare_Derivatives():

    import numpy as np
    #TwoStep
    TwoStep_dxib0 = np.dot(RSDPower.dPb0, RSDPower.dxip0)
    TwoStep_dxib2 = np.dot(RSDPower.dPb2, RSDPower.dxip2)
    TwoStep_dxib4 = np.dot(RSDPower.dPb4, RSDPower.dxip4)
    
    TwoStep_dxif0 = np.dot(RSDPower.dPf0, RSDPower.dxip0)
    TwoStep_dxif2 = np.dot(RSDPower.dPf2, RSDPower.dxip2)
    TwoStep_dxif4 = np.dot(RSDPower.dPf4, RSDPower.dxip4)
    
    TwoStep_dxis0 = np.dot(RSDPower.dPs0, RSDPower.dxip0)
    TwoStep_dxis2 = np.dot(RSDPower.dPs2, RSDPower.dxip2)
    TwoStep_dxis4 = np.dot(RSDPower.dPs4, RSDPower.dxip4)
    
    import matplotlib.pyplot as plt
    
    
    #plt.subplot(3, 1, 1)
    #fig, ax1, ax2 = plt.figure()
    fig, (ax1, ax2) = plt.subplots( nrows = 1, ncols = 2, figsize = (14,7))
    
    ax1.loglog(RSDPower.rcenter,TwoStep_dxib0, 'b-', label='Two, 0' )
    ax1.loglog(RSDPower.rcenter,RSDPower.dxib0, 'b--',label='One, 0' )
    ax1.loglog(RSDPower.rcenter,TwoStep_dxib2, 'r-',label='Two, 2' )
    ax1.loglog(RSDPower.rcenter,RSDPower.dxib2, 'r--' ,label='One, 2')
    ax1.loglog(RSDPower.rcenter,TwoStep_dxib4, 'g-',label='Two, 4' )
    ax1.loglog(RSDPower.rcenter,RSDPower.dxib4, 'g--',label='One, 4' )
    ax1.legend(loc=3)
    ax1.set_xlabel('r')
    ax1.set_ylabel('dxi/db')
    
    #ax2.semilogx(RSDPower.rcenter,TwoStep_dxib0, 'b-', label='Two, 0' )
    ax2.semilogx(RSDPower.rcenter,RSDPower.dxib0-TwoStep_dxib0, 'b.',label='0' )
    #ax2.semilogx(RSDPower.rcenter,TwoStep_dxib2, 'r-',label='Two, 2' )
    ax2.semilogx(RSDPower.rcenter,RSDPower.dxib2-TwoStep_dxib2, 'r.' ,label='2')
    #ax2.semilogx(RSDPower.rcenter,TwoStep_dxib4, 'g-',label='Two, 4' )
    ax2.semilogx(RSDPower.rcenter,RSDPower.dxib4-TwoStep_dxib4, 'g.',label='4' )
    ax2.set_xlabel('r')
    ax2.set_ylabel('One Step - Two Step')
    ax2.legend(loc=1)
    
    """
    plt.ylabel('dxib')
    plt.title('Derivatives  kN = {}, rN = {}'.format(RSDPower.n, RSDPower.n2))
    plt.legend(loc=4,prop={'size':10})
    """
    """
    plt.subplot(3, 1, 2)
    plt.loglog(RSDPower.rcenter,TwoStep_dxif0, 'b-', label='Two, 0'  )
    plt.loglog(RSDPower.rcenter,RSDPower.dxif0, 'b^',label='One, 0' )
    plt.loglog(RSDPower.rcenter,TwoStep_dxif2, 'r-', label='Two, 2'  )
    plt.loglog(RSDPower.rcenter,RSDPower.dxif2, 'r^',label='One, 2' )
    plt.loglog(RSDPower.rcenter,TwoStep_dxif4, 'g-', label='Two, 4'  )
    plt.loglog(RSDPower.rcenter,RSDPower.dxif4, 'g^',label='One, 4' )
    plt.ylabel('dxif')
    
    plt.subplot(3, 1, 3)
    plt.loglog(RSDPower.rcenter,TwoStep_dxis0, 'b-', label='Two, 0'  )
    plt.loglog(RSDPower.rcenter,RSDPower.dxis0, 'b^',label='One, 0' )
    plt.loglog(RSDPower.rcenter,TwoStep_dxis2, 'r-', label='Two, 2'  )
    plt.loglog(RSDPower.rcenter,RSDPower.dxis2, 'r^',label='One, 2' )
    plt.loglog(RSDPower.rcenter,TwoStep_dxis4, 'g-', label='Two, 4'  )
    plt.loglog(RSDPower.rcenter,RSDPower.dxis4, 'g^',label='One, 4' )
    plt.ylabel('dxis')
    """
    
    #plt.show()

    from matplotlib.backends.backend_pdf import PdfPages
    pdfname = 'plots/compare/Compare_Derivatives_r{:>1.3f}_k{:>1.3f}_sk{:>1.4f}.pdf'.format(RSDPower.dlnr, RSDPower.dlnk, RSDPower.sdlnk[2])
    pdf=PdfPages( pdfname )
    pdf.savefig(fig)
    pdf.close()
    #plt.clf()
    print " pdf file saved : ", pdfname



def Reid_step2(l):
    
    
    matricesXi = [RSDPower.covariance00, RSDPower.covariance02, RSDPower.covariance04, np.transpose(RSDPower.covariance02), RSDPower.covariance22, RSDPower.covariance24,np.transpose(RSDPower.covariance04), np.transpose(RSDPower.covariance24), RSDPower.covariance44]
    
    #matrices2P = [dPb0, dPb2, dPb4, dPf0, dPf2, dPf4, dPs0, dPs2, dPs4]
    
    matrices2P = [RSDPower.dPb0[kcut_min:kcut_max+1], RSDPower.dPb2[kcut_min:kcut_max+1],\
                  RSDPower.dPb4[kcut_min:kcut_max+1], RSDPower.dPf0[kcut_min:kcut_max+1],\
                  RSDPower.dPf2[kcut_min:kcut_max+1], RSDPower.dPf4[kcut_min:kcut_max+1],\
                  RSDPower.dPs0[kcut_min:kcut_max+1], RSDPower.dPs2[kcut_min:kcut_max+1],\
                  RSDPower.dPs4[kcut_min:kcut_max+1]]
                  
    
    Xizeros = np.zeros((len(RSDPower.kcenter),len(RSDPower.rcenter)))
    matrices2Xi = [RSDPower.dxip0, Xizeros,Xizeros,Xizeros,RSDPower.dxip2,Xizeros,Xizeros,Xizeros,RSDPower.dxip4]
    
    """ GET full size C_Xi"""
    l_r = len(RSDPower.rcenter)
    C_matrix3 = CombineCovariance3(l_r, matricesXi)
    C_matrix2 = CombineCovariance2(l_r, matricesXi)
    Xi, Xi2 = CombineDevXi3(l_r, matrices2Xi)
    
    
    """ F_bandpower """
    Fisher_bandpower_Xi = FisherProjection(Xi, C_matrix3)
    cut = len(RSDPower.kcenter)
    
    
    part00 = Fisher_bandpower_Xi[0:cut, 0:cut]
    part02 = Fisher_bandpower_Xi[0:cut, cut:2*cut]
    part04 = Fisher_bandpower_Xi[0:cut, 2*cut:3*cut+1]
    part22 = Fisher_bandpower_Xi[cut:2*cut, cut:2*cut]
    part24 = Fisher_bandpower_Xi[cut:2*cut, 2*cut:3*cut+1]
    part44 = Fisher_bandpower_Xi[2*cut:3*cut+1, 2*cut:3*cut+1]
    
    part_list = [ part00[kcut_min:kcut_max+1,kcut_min:kcut_max+1], part02[kcut_min:kcut_max+1,kcut_min:kcut_max+1], part04[kcut_min:kcut_max+1,kcut_min:kcut_max+1], np.transpose(part02[kcut_min:kcut_max+1,kcut_min:kcut_max+1]), part22[kcut_min:kcut_max+1,kcut_min:kcut_max+1], part24[kcut_min:kcut_max+1,kcut_min:kcut_max+1], np.transpose(part04[kcut_min:kcut_max+1,kcut_min:kcut_max+1]), np.transpose(part24[kcut_min:kcut_max+1,kcut_min:kcut_max+1]), part44[kcut_min:kcut_max+1,kcut_min:kcut_max+1]]

    part_Fisher_bandpower_Xi = CombineCovariance3(l, part_list)
    part_Fisher_bandpower_Xi_two = CombineCovariance2(l, part_list)
    
    """ dP/dq"""
    XP, XP2 = CombineDevXi(l, matrices2P)
    FisherXi = FisherProjection(XP, inv(part_Fisher_bandpower_Xi))
    FisherXi_two = FisherProjection(XP2, inv(part_Fisher_bandpower_Xi_two))
    
    """
    Cross = CrossCoeff( inv(Fisher_bandpower_Xi) )
    
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    plt.imshow( Cross )
    
    #plt.imshow( inv(FisherXi) )
    plt.show()
    """
    
    error_b_marginal, error_f_marginal = FractionalError( RSDPower.b, RSDPower.f, inv(FisherXi))
    error_b_determin, error_f_determin = FractionalError( RSDPower.b, RSDPower.f, inv(FisherXi[0:2,0:2]))
    error_b_marginal2, error_f_marginal2 = FractionalError( RSDPower.b, RSDPower.f, inv(FisherXi_two))
    error_b_determin2, error_f_determin2 = FractionalError( RSDPower.b, RSDPower.f, inv(FisherXi_two[0:2,0:2]))
    
    rr = 1.15 * np.pi/RSDPower.kcenter[l]
    return rr, error_b_determin, error_f_determin, error_b_marginal, error_f_marginal\
    ,error_b_determin2, error_f_determin2, error_b_marginal2, error_f_marginal2

def Reid_step1(l):

    
    matricesXi = [RSDPower.covariance00, RSDPower.covariance02, RSDPower.covariance04, np.transpose(RSDPower.covariance02), RSDPower.covariance22, RSDPower.covariance24,np.transpose(RSDPower.covariance04), np.transpose(RSDPower.covariance24), RSDPower.covariance44]
    
    matrices2Xi = [RSDPower.dxib0, RSDPower.dxib2, RSDPower.dxib4, RSDPower.dxif0, RSDPower.dxif2, RSDPower.dxif4, RSDPower.dxis0, RSDPower.dxis2, RSDPower.dxis4]
    Xi, Xi2 = CombineDevXi(l, matrices2Xi)
    
    C_matrix3 = CombineCovariance3(l, matricesXi)
    C_matrix2 = CombineCovariance2(l, matricesXi)

    """ Fisher """
    Fisher_marginal = FisherProjection(Xi, C_matrix3)
    Fisher_determin = Fisher_marginal[0:2,0:2]
    Fisher_marginal2 = FisherProjection(Xi2, C_matrix2)
    Fisher_determin2 = Fisher_marginal2[0:2,0:2]
    error_b_marginal, error_f_marginal = FractionalError( RSDPower.b, RSDPower.f, inv(Fisher_marginal))
    error_b_determin, error_f_determin = FractionalError( RSDPower.b, RSDPower.f, inv(Fisher_determin))
    error_b_marginal2, error_f_marginal2 = FractionalError( RSDPower.b, RSDPower.f, inv(Fisher_marginal2))
    error_b_determin2, error_f_determin2 = FractionalError( RSDPower.b, RSDPower.f, inv(Fisher_determin2))

    return RSDPower.rcenter[l], error_b_determin, error_f_determin, error_b_marginal, error_f_marginal\
    ,error_b_determin2, error_f_determin2, error_b_marginal2, error_f_marginal2


Reid_step2 = np.vectorize(Reid_step2)
Reid_step1 = np.vectorize(Reid_step1)
error_only_CP = np.vectorize(error_only_CP)

if __name__=='__main__':
    main()


stop






















#####========================
numberlist = np.arange(1,len(RSDPower.rcenter))

rr, error_b_determin, error_f_determin, error_b_marginal, error_f_marginal\
    ,error_b_determin2, error_f_determin2, error_b_marginal2, error_f_marginal2 = error_only_Cxi(numberlist)


DAT = np.column_stack((rr, error_b_determin, error_f_determin, error_b_marginal, error_f_marginal))
DAT2 = np.column_stack((rr, error_b_determin2, error_f_determin2, error_b_marginal2, error_f_marginal2))

filename ='RSDXiinversefisher.txt'
filename2 = 'RSDXiinversefisher2.txt'
np.savetxt(filename,DAT,delimiter=" ", fmt="%s")
np.savetxt(filename2,DAT2,delimiter=" ", fmt="%s")

output_pdf = 'RSDFractionalError_Xi_dlnr{:>1.2f}_dlnk{:>1.2f}.pdf'.format(RSDPower.dlnr, RSDPower.dlnk)
plot_title = 'RSDFractional Error on b and f (RSD)\n z=0.55, dlnr = {:>1.3f}, dlnk = {:>1.3f}, rN = {}, kN = {}'.format(RSDPower.dlnr, RSDPower.dlnk, RSDPower.n, RSDPower.n2)

#output_pdf = 'RSDFractionalError_Xi_noFOG_dlnr{:>1.2f}_dlnk{:>1.2f}.pdf'.format(RSDPower.dlnr, RSDPower.dlnk)
#plot_title = 'RSDFractional Error on b and f (w/o FoG)\n z=0.55, dlnr = {:>1.3f}, dlnk = {:>1.3f}, rN = {}, kN = {}'.format(RSDPower.dlnr, RSDPower.dlnk, RSDPower.n, RSDPower.n2)


file = open(filename)
file2 = open(filename2)
plotting(file, file2, output_pdf, plot_title)


print "\n--------------------------------------------"
print " Data stored in ", filename, filename2
print " Plot saved : ", output_pdf
print "--------------------------------------------"


##################################################


