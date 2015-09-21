import numpy as np
from numpy import zeros, sqrt, pi
from numpy.linalg import inv
from numpy import vectorize
from scipy.interpolate import interp1d
from covariance_class2 import *
import pp, sys, time, datetime
from multiprocessing import Process, Queue

#   Initialization

print '\nestimator 2'
print datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

KMIN = 0.0001
KMAX = 502.32
    
# r scale
RMIN = 0.1# 6. #29. # 24. #0.1 for Reid   #1.15 * np.pi / self.kmax
RMAX = 180. #200.  #628.32 # for Reid  #1.15 * np.pi / self.kmin
    
# REID (0.01~ 180) corresponding k value :(0.02 ~ 361.28)
# REID convergence condition : kN = 61, rN = 151, subN = 101
# REID convergence condition : kN = 101, rN = 101, subN = 121
kN = 101  #converge perfectly at 151, 2by2 components converge at 121 # the number of k bins. the sample should be odd
rN = 501  #101 for Reid # number of r bins
subN = 201 #101 for Reid

RSDPower = RSD_covariance(KMIN, KMAX, RMIN, RMAX, kN, rN, subN)
#RSDPower.compile_fortran_modules()
## run only for first time running

file = open('matterpower_z_0.55.dat')
#file = open('matterpower.dat')
RSDPower.MatterPower(file)
RSDPower.Shell_avg_band()

    
rcut_max = len(RSDPower.rcenter)-1
rcut_min = 0 #25#len(RSDPower.rcenter)-1
kcut_min = 50 #12 # 45 #150 #90#104
kcut_max = 82#  len(RSDPower.kmax)-1 #24  #90
    
# 21,31 (61) 45, 73 (151) for BAO only (k : 0.02 ~ 0.3)
# 18, 29(61) for BAO+RSD (k: 0.01 ~ 0.2))
# 12, 24(41) for big scale and small scale
# 21, __(61) for Reid range # REID (0.01~ 180) corresponding k value :(0.02 ~ 361.28)
# 50, (101) for Reid range # REID (0.01~ 180) corresponding k value :(0.02 ~ 361.28)

print "kcut_min :", RSDPower.kmin[kcut_min], "  kcut_max :", RSDPower.kmax[kcut_max]
print "rcut_min :", RSDPower.rmin[rcut_max], "  rcut_max :", RSDPower.rmax[rcut_min], "\n"


def main():


    #multipole band ================================================================
    # shall averaged in each k bin
    
    RSDPower.multipole_bandpower_all()
    
    # multipole derivatives ========================================================
    
    ## devP
    RSDPower.RSDband_derivative_P_All()
    # dPb0, dPf0, dPs0, dPb2, dPf2, dPs2, dPb4, dPf4, dPs4 = RSDPower.RSDband_derivative_P_All()
    
    """
    DAT = np.column_stack(( RSDPower.kcenter, RSDPower.dPb0))
    filename1 = 'plots/compare/dPb0_sk{:>1.3f}_k{:>1.3f}.txt'.format(RSDPower.sdlnk[2], RSDPower.dlnk)
    np.savetxt(filename1, DAT, delimiter=" ", fmt="%s")
    print "txt saved : ", filename1
    stop
    """
    """
    file1 = 'plots/compare/dPb0_sk0.000_k0.038.txt' # kN401, subN251
    file2 = 'plots/compare/dPb0_sk0.000_k0.019.txt' # kN801, subN125
    
    Plot_dPb(file2)
    stop
    """
    #Plot_derivatives()
    #stop
    
    RSDPower.derivative_Xi_band_All()
    #dxip0, dxip2, dxip4 = RSDPower.derivative_Xi_band_All()
    
    RSDPower.derivative_Xi_All()
    #RSDPower.RSDband_derivative_xi_All()

    Compare_Derivatives()
    

    # test ======================================================================
    
    
    #covariance matrix============================================================

    ##calling function for covariance matrix components Cp_ll'
    RSDPower.RSDband_covariance_PP_All()

    RSDPower.RSDband_covariance_PXi_All()
    ## parallel python applied.
    ## check! it could not be retrieved by order
    
    
    start_time = time.time()
    RSDPower.RSD_covariance_Allmodes()
    print "elapsed time :", time.time()-start_time
    
    matricesXi = [RSDPower.covariance00, RSDPower.covariance02, RSDPower.covariance04, np.transpose(RSDPower.covariance02), RSDPower.covariance22, RSDPower.covariance24,np.transpose(RSDPower.covariance04), np.transpose(RSDPower.covariance24), RSDPower.covariance44]
    print np.sum(matricesXi)
    
    
    error()
    break
    
    #-------------------------------------------------------
    # loop initial setting
    
    step1_process = 12 # for r loop
    step2_process = 12 # for k loop
    
    print 'multi_processing for r loop : ', step1_process, ' workers'
    print 'multi_processing for k loop : ', step2_process, ' workers'
    
    numberlist_k = np.arange(1, len(RSDPower.kcenter)-1,10)
    numberlist_k_split = np.array_split(numberlist_k, step2_process)
  
    numberlist_r = np.arange(1,len(RSDPower.rcenter)-1,2)
    numberlist_r_split = np.array_split(numberlist_r, step1_process)
    
    #-------------------------------------------------------
    # step 2 r-cut loop

    """
    def Reid_step2_loop(q, order, input):
    
        rr, error_b_determin, error_f_determin, error_b_marginal, error_f_marginal, error_b_determin2, error_f_determin2, error_b_marginal2, error_f_marginal2 = Reid_step2(input)
        DAT = np.array(np.concatenate((rr, error_b_determin, error_f_determin, error_b_marginal, error_f_marginal, error_b_determin2, error_f_determin2, error_b_marginal2, error_f_marginal2), axis = 0)).reshape(9,len(input))
        q.put(( order, DAT ))

    
    loop_queue = Queue()
    loop_processes = [Process(target=Reid_step2_loop, args=(loop_queue, z[0], z[1])) for z in zip(range(step1_process+1), numberlist_r_split)]
    for p in loop_processes:
        p.start()
        #for p in loop_processes:
        #p.join()
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


    print 'plotting'
    output_b_pdf = 'plots/Reid/RTwoFractionalError_b_dlnr{:>1.2f}_dlnk{:>1.2f}.pdf'.format(RSDPower.dlnr, RSDPower.dlnk)
    output_f_pdf = 'plots/Reid/RTwoFractionalError_f_dlnr{:>1.2f}_dlnk{:>1.2f}.pdf'.format(RSDPower.dlnr, RSDPower.dlnk)
    plot_title = 'RSDFractional Error on b and f (Two Step)\n z=0.55, dlnr = {:>1.3f}, dlnk = {:>1.3f}, kN = {}, rN = {}'.format(RSDPower.dlnr, RSDPower.dlnk, RSDPower.n, RSDPower.n2)

    Linear_plot( rr, ['det', 'marg', 'det2', 'marg2'], error_b_determin, error_b_marginal, error_b_determin2, error_b_marginal2, pdfname = output_b_pdf, title = plot_title, xmin=0.0, xmax=60.0, ymin=0.0005, ymax=5*10**(-2), scale='semilogy', basename='r_min'  )
    Linear_plot( rr, ['det', 'marg', 'det2', 'marg2'], error_f_determin, error_f_marginal, error_f_determin2, error_f_marginal2, pdfname = output_f_pdf, title = plot_title, xmin=0.0, xmax=60.0, ymin=0.0, ymax=0.07, basename='r_min'  )#xmax=180.0, ymin=0.0, ymax=0.30, basename='r_min'  )

    """
    #-------------------------------------------------------
    # step 2 k-cut loop
    """
    def Reid_step2_loop(q, order, input):
        
        rr, error_b_determin, error_f_determin, error_b_marginal, error_f_marginal, error_b_determin2, error_f_determin2, error_b_marginal2, error_f_marginal2 = Reid_step2_kcut(input)
        DAT = np.array(np.concatenate((rr, error_b_determin, error_f_determin, error_b_marginal, error_f_marginal, error_b_determin2, error_f_determin2, error_b_marginal2, error_f_marginal2), axis = 0)).reshape(9,len(input))
        q.put(( order, DAT ))

    loop_queue = Queue()
    loop_processes = [Process(target=Reid_step2_loop, args=(loop_queue, z[0], z[1])) for z in zip(range(step2_process+1), numberlist_k_split)]
    for p in loop_processes: p.start()

    loop_result = [loop_queue.get() for p in loop_processes]
    loop_result.sort()
    loop_result_list = [ loop[1] for loop in loop_result ]
    loops = loop_result_list[0]
    for i in range(1, step2_process): loops = np.concatenate((loops, loop_result_list[i]), axis =1 )

    rr = loops[0]
    error_b_determin = loops[1]
    error_f_determin = loops[2]
    error_b_marginal = loops[3]
    error_f_marginal = loops[4]
    error_b_determin2 = loops[5]
    error_f_determin2 = loops[6]
    error_b_marginal2 = loops[7]
    error_f_marginal2 = loops[8]


    output_b_pdf = 'plots/Reid/KTwoFractionalError_b_dlnr{:>1.2f}_dlnk{:>1.2f}.pdf'.format(RSDPower.dlnr, RSDPower.dlnk)
    output_f_pdf = 'plots/Reid/KTwoFractionalError_f_dlnr{:>1.2f}_dlnk{:>1.2f}.pdf'.format(RSDPower.dlnr, RSDPower.dlnk)
    plot_title = 'RSDFractional Error on b and f (Two Step, kcut)\n z=0.55, dlnr = {:>1.3f}, dlnk = {:>1.3f}, kN = {}, rN = {}'.format(RSDPower.dlnr, RSDPower.dlnk, RSDPower.n, RSDPower.n2)

    Linear_plot( rr, ['det', 'marg', 'det2', 'marg2'], error_b_determin, error_b_marginal, error_b_determin2, error_b_marginal2, pdfname = output_b_pdf, title = plot_title, xmin=0.0, xmax=60.0, ymin=0.0005, ymax=5*10**(-2), scale='semilogy', basename='r_min' )
    Linear_plot( rr, ['det', 'marg', 'det2', 'marg2'], error_f_determin, error_f_marginal, error_f_determin2, error_f_marginal2, pdfname = output_f_pdf, title = plot_title, xmin=0.0, xmax=60.0, ymin=0.0, ymax=0.07, basename='r_min'  )
    """
    
    #-------------------------------------------------------
    # step 1 r loop

    """
 
    def Reid_step1_loop(q, order, input):
    
        rr, error_b_determin, error_f_determin, error_b_marginal, error_f_marginal, error_b_determin2, error_f_determin2, error_b_marginal2, error_f_marginal2 = Reid_step1(input)
        DAT = np.array(np.concatenate((rr, error_b_determin, error_f_determin, error_b_marginal, error_f_marginal, error_b_determin2, error_f_determin2, error_b_marginal2, error_f_marginal2), axis = 0)).reshape(9,len(input))
        q.put(( order, DAT ))
    
    loop_queue = Queue()
    loop_processes = [Process(target=Reid_step1_loop, args=(loop_queue, z[0], z[1])) for z in zip(range(step1_process), numberlist_r_split)]
    for p in loop_processes:
        p.start()
        #for p in loop_processes:
        #p.join()
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
    
    
    output_b_pdf2 = 'plots/Reid/ROneFractionalError_b_dlnr{:>1.2f}_dlnk{:>1.2f}.pdf'.format(RSDPower.dlnr, RSDPower.dlnk)
    output_f_pdf2 = 'plots/Reid/ROneFractionalError_f_dlnr{:>1.2f}_dlnk{:>1.2f}.pdf'.format(RSDPower.dlnr, RSDPower.dlnk)
    plot_title2 = 'RSDFractional Error on b and f (One Step, Xi)\n z=0.55, dlnr = {:>1.3f}, dlnk = {:>1.3f}, kN = {}, rN = {}'.format(RSDPower.dlnr, RSDPower.dlnk, RSDPower.n, RSDPower.n2)
    
    Linear_plot( rr, ['det', 'marg', 'det2', 'marg2'], error_b_determin, error_b_marginal, error_b_determin2, error_b_marginal2, pdfname = output_b_pdf2, title = plot_title2, xmin=0.0, xmax=60.0, ymin=0.0005, ymax=5*10**(-2), scale='semilogy',basename='r_min'  )
    Linear_plot( rr, ['det', 'marg', 'det2', 'marg2'], error_f_determin, error_f_marginal, error_f_determin2, error_f_marginal2, pdfname = output_f_pdf2, title = plot_title2, xmin=0.0, xmax=60.0, ymin=0.0, ymax=0.07, basename='r_min'  )

    """

    #-------------------------------------------------------
    # step 1 k loop
    """
    rr, error_b_determin, error_f_determin, error_b_marginal, error_f_marginal, error_b_determin2, error_f_determin2, error_b_marginal2, error_f_marginal2 = error_only_CP(numberlist_k)

    output_b_pdf = 'plots/Reid/KOneFractionalError_b_dlnr{:>1.2f}_dlnk{:>1.2f}.pdf'.format(RSDPower.dlnr, RSDPower.dlnk)
    output_f_pdf = 'plots/Reid/KOneFractionalError_f_dlnr{:>1.2f}_dlnk{:>1.2f}.pdf'.format(RSDPower.dlnr, RSDPower.dlnk)
    plot_title = 'RSDFractional Error on b and f (One Step, P)\n z=0.55, dlnr = {:>1.3f}, dlnk = {:>1.3f}, kN = {}, rN = {}'.format(RSDPower.dlnr, RSDPower.dlnk, RSDPower.n, RSDPower.n2)

    Linear_plot( rr, ['det', 'marg', 'det2', 'marg2'], error_b_determin, error_b_marginal, error_b_determin2, error_b_marginal2, pdfname = output_b_pdf, title = plot_title, xmin=0.0, xmax=60.0, ymin=0.0005, ymax=5*10**(-2), scale='semilogy', basename='r_min' )
    Linear_plot( rr, ['det', 'marg', 'det2', 'marg2'], error_f_determin, error_f_marginal, error_f_determin2, error_f_marginal2, pdfname = output_f_pdf, title = plot_title, xmin=0.0, xmax=60.0, ymin=0.0, ymax=0.07, basename='r_min'  )
    """

def error():
    
    l1 = len(RSDPower.kcenter)-1
    l2 = len(RSDPower.rcenter)-1
    
    matricesPP_total = [RSDPower.covariance_PP00, RSDPower.covariance_PP02, RSDPower.covariance_PP04,RSDPower.covariance_PP02, RSDPower.covariance_PP22, RSDPower.covariance_PP24,RSDPower.covariance_PP04, RSDPower.covariance_PP24, RSDPower.covariance_PP44]
    """
    matricesPXi = [covariance_PXi00, covariance_PXi02, covariance_PXi04, covariance_PXi20, covariance_PXi22, covariance_PXi24,covariance_PXi40, covariance_PXi42, covariance_PXi44]
    matricesXi = [covariance00, covariance02, covariance04, np.transpose(covariance02), covariance22, covariance24,np.transpose(covariance04), np.transpose(covariance24), covariance44]
    """
    
    
    matricesPP = [RSDPower.covariance_PP00[kcut_min:kcut_max+1,kcut_min:kcut_max+1],\
                  RSDPower.covariance_PP02[kcut_min:kcut_max+1,kcut_min:kcut_max+1], \
                  RSDPower.covariance_PP04[kcut_min:kcut_max+1,kcut_min:kcut_max+1], \
                  RSDPower.covariance_PP02[kcut_min:kcut_max+1,kcut_min:kcut_max+1],\
                  RSDPower.covariance_PP22[kcut_min:kcut_max+1,kcut_min:kcut_max+1], \
                  RSDPower.covariance_PP24[kcut_min:kcut_max+1,kcut_min:kcut_max+1],\
                  RSDPower.covariance_PP04[kcut_min:kcut_max+1,kcut_min:kcut_max+1], \
                  RSDPower.covariance_PP24[kcut_min:kcut_max+1,kcut_min:kcut_max+1], \
                  RSDPower.covariance_PP44[kcut_min:kcut_max+1,kcut_min:kcut_max+1]]
    matricesPXi = [RSDPower.covariance_PXi00[kcut_min:kcut_max+1,rcut_min:rcut_max+1], \
                   RSDPower.covariance_PXi02[kcut_min:kcut_max+1,rcut_min:rcut_max+1],\
                   RSDPower.covariance_PXi04[kcut_min:kcut_max+1,rcut_min:rcut_max+1],\
                   RSDPower.covariance_PXi20[kcut_min:kcut_max+1,rcut_min:rcut_max+1], \
                   RSDPower.covariance_PXi22[kcut_min:kcut_max+1,rcut_min:rcut_max+1], \
                   RSDPower.covariance_PXi24[kcut_min:kcut_max+1,rcut_min:rcut_max+1],\
                   RSDPower.covariance_PXi40[kcut_min:kcut_max+1,rcut_min:rcut_max+1],\
                   RSDPower.covariance_PXi42[kcut_min:kcut_max+1,rcut_min:rcut_max+1],\
                   RSDPower.covariance_PXi44[kcut_min:kcut_max+1,rcut_min:rcut_max+1]]
    matricesXi = [RSDPower.covariance00[rcut_min:rcut_max+1,rcut_min:rcut_max+1],\
                  RSDPower.covariance02[rcut_min:rcut_max+1,rcut_min:rcut_max+1], \
                  RSDPower.covariance04[rcut_min:rcut_max+1,rcut_min:rcut_max+1],\
                  np.transpose(RSDPower.covariance02[rcut_min:rcut_max+1,rcut_min:rcut_max+1]), \
                  RSDPower.covariance22[rcut_min:rcut_max+1,rcut_min:rcut_max+1],\
                  RSDPower.covariance24[rcut_min:rcut_max+1,rcut_min:rcut_max+1],\
                  np.transpose(RSDPower.covariance04[rcut_min:rcut_max+1,rcut_min:rcut_max+1]),\
                  np.transpose(RSDPower.covariance24[rcut_min:rcut_max+1,rcut_min:rcut_max+1]),\
                  RSDPower.covariance44[rcut_min:rcut_max+1,rcut_min:rcut_max+1]]
    
    matrices2P_cut = [RSDPower.dPb0[kcut_min:kcut_max+1], RSDPower.dPb2[kcut_min:kcut_max+1],\
                      RSDPower.dPb4[kcut_min:kcut_max+1], RSDPower.dPf0[kcut_min:kcut_max+1], \
                      RSDPower.dPf2[kcut_min:kcut_max+1], RSDPower.dPf4[kcut_min:kcut_max+1], \
                      RSDPower.dPs0[kcut_min:kcut_max+1], RSDPower.dPs2[kcut_min:kcut_max+1],\
                      RSDPower.dPs4[kcut_min:kcut_max+1]]
                  
    matrices2P = [RSDPower.dPb0, RSDPower.dPb2, RSDPower.dPb4, RSDPower.dPf0, RSDPower.dPf2, RSDPower.dPf4, RSDPower.dPs0, RSDPower.dPs2, RSDPower.dPs4]
    
    
    
    #   derivative matrices
    derivative_P0 = np.identity(len(RSDPower.kcenter))[:,kcut_min:kcut_max+1]
    Pzeros = np.zeros((len(RSDPower.kcenter),kcut_max-kcut_min+1))
    derivative_P = np.concatenate((np.concatenate((derivative_P0, Pzeros, Pzeros),axis=1 ),\
                                   np.concatenate((Pzeros, derivative_P0,Pzeros),axis=1 ),\
                                   np.concatenate((Pzeros, Pzeros,derivative_P0),axis=1 )), axis=0)
    Xizeros = np.zeros((len(RSDPower.kcenter),rcut_max - rcut_min + 1))

    derivative_correl_avg = np.concatenate(( np.concatenate((RSDPower.dxip0,Xizeros,Xizeros), axis=1),\
                                            np.concatenate((Xizeros,RSDPower.dxip2,Xizeros), axis=1),\
                                            np.concatenate((Xizeros,Xizeros,RSDPower.dxip4), axis=1)),axis=0 )
                                            
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
    derivative_correl_avg2 = np.concatenate(( np.concatenate((RSDPower.dxip0,Xizeros), axis=1),\
                                            np.concatenate((RSDPower.dxip2,Xizeros), axis=1)),axis=0 )
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
    
    bandpower_base = RSDPower.multipole_bandpower #[0:len(RSDPower.kcenter)+1]
    
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
    
    
    
    
    #------------------------------------------------------------
    """ projection to b and f space"""

    XP, XP2 = CombineDevXi(l1, matrices2P)
    XP_cut, XP2_cut = CombineDevXi(l1, matrices2P_cut)
    
    
    # Fisher, all modes, s marginalized
    FisherPP = FisherProjection(XP_cut, C_matrix3PP)
    FisherXi = FisherProjection_Fishergiven(XP, Fisher_bandpower_Xi)
    FisherC_nf = FisherProjection_Fishergiven(XP, Fisher_bandpower_nf)
    FisherC = FisherProjection_Fishergiven(XP, Fisher_bandpower)
    
    # Fisher, two modes, s marginalized
    FisherPP2 = FisherProjection(XP2_cut, C_matrix2PP)
    FisherXi2 = FisherProjection_Fishergiven(XP2, Fisher_bandpower_Xi2)
    FisherC2_nf = FisherProjection_Fishergiven(XP2, Fisher_bandpower2_nf)
    FisherC2 = FisherProjection_Fishergiven(XP2, Fisher_bandpower2)
    
    # Fisher, all modes, s determined
    FisherPP_d = FisherProjection(XP_cut[0:2,:], C_matrix3PP)
    FisherXi_d = FisherProjection_Fishergiven(XP[0:2,:], Fisher_bandpower_Xi)
    FisherC_nf_d = FisherProjection_Fishergiven(XP[0:2,:], Fisher_bandpower_nf)
    FisherC_d = FisherProjection_Fishergiven(XP[0:2,:], Fisher_bandpower)

    
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
    Fisher_determin = Fisher_marginal[0:2,0:2]
    Fisher_marginal2 = FisherProjection(Xi2, C_matrix2)
    Fisher_determin2 = Fisher_marginal2[0:2,0:2]
    
    """
    Cross = CrossCoeff( C_matrix3 )
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    plt.imshow( Cross )
    plt.show()
    """


    error_b_marginal, error_f_marginal = FractionalError( RSDPower.b, RSDPower.f, inv(Fisher_marginal))
    error_b_determin, error_f_determin = FractionalError( RSDPower.b, RSDPower.f, inv(Fisher_determin))
    error_b_marginal2, error_f_marginal2 = FractionalError( RSDPower.b, RSDPower.f, inv(Fisher_marginal2))
    error_b_determin2, error_f_determin2 = FractionalError( RSDPower.b, RSDPower.f, inv(Fisher_determin2))
    
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
    
    
    matricesPP = [RSDPower.covariance_PP00[kcut_min:kcut_max+1,kcut_min:kcut_max+1],\
                  RSDPower.covariance_PP02[kcut_min:kcut_max+1,kcut_min:kcut_max+1], \
                  RSDPower.covariance_PP04[kcut_min:kcut_max+1,kcut_min:kcut_max+1], \
                  RSDPower.covariance_PP02[kcut_min:kcut_max+1,kcut_min:kcut_max+1],\
                  RSDPower.covariance_PP22[kcut_min:kcut_max+1,kcut_min:kcut_max+1], \
                  RSDPower.covariance_PP24[kcut_min:kcut_max+1,kcut_min:kcut_max+1],\
                  RSDPower.covariance_PP04[kcut_min:kcut_max+1,kcut_min:kcut_max+1], \
                  RSDPower.covariance_PP24[kcut_min:kcut_max+1,kcut_min:kcut_max+1], \
                  RSDPower.covariance_PP44[kcut_min:kcut_max+1,kcut_min:kcut_max+1]]
    matricesPXi = [RSDPower.covariance_PXi00[kcut_min:kcut_max+1,rcut_min:rcut_max+1], \
                   RSDPower.covariance_PXi02[kcut_min:kcut_max+1,rcut_min:rcut_max+1],\
                   RSDPower.covariance_PXi04[kcut_min:kcut_max+1,rcut_min:rcut_max+1],\
                   RSDPower.covariance_PXi20[kcut_min:kcut_max+1,rcut_min:rcut_max+1], \
                   RSDPower.covariance_PXi22[kcut_min:kcut_max+1,rcut_min:rcut_max+1], \
                   RSDPower.covariance_PXi24[kcut_min:kcut_max+1,rcut_min:rcut_max+1],\
                   RSDPower.covariance_PXi40[kcut_min:kcut_max+1,rcut_min:rcut_max+1],\
                   RSDPower.covariance_PXi42[kcut_min:kcut_max+1,rcut_min:rcut_max+1],\
                   RSDPower.covariance_PXi44[kcut_min:kcut_max+1,rcut_min:rcut_max+1]]
                   
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
    FisherXi = FisherProjection_Fishergiven(XP, Fisher_bandpower_Xi)
    FisherC_nooff = FisherProjection_Fishergiven(XP, Fisher_bandpower_nooff)
    FisherC = FisherProjection_Fishergiven(XP, Fisher_bandpower)
    
    Cov_PP = inv(FisherPP)
    Cov_Xi = inv(FisherXi)
    Cov_C_nooff = inv(FisherC_nooff)
    Cov_C = inv(FisherC)

    makedirectory('plots/RSD/ellipse')
    title = 'Confidence Ellipse for b and f, l = 0,2,4 \n rmin={:>3.5f} rmax={:>3.2f} kmin={:>3.5f} kmax={:>3.2f}'.format(RSDPower.rmin[rcut_max],RSDPower.rmax[rcut_min],RSDPower.kmin[kcut_min],RSDPower.kmax[kcut_max] )
    pdfname = 'plots/RSD/ellipse/ConfidenceEllipseTotal_r{:>3.2f}_{:>3.2f}_k{:>3.2f}_{:>3.2f}_rN{}kN{}.pdf'.format(RSDPower.rmin[rcut_max],RSDPower.rmax[rcut_min],RSDPower.kmin[kcut_min],RSDPower.kmax[kcut_max], RSDPower.n2, RSDPower.n )
    labellist = ['$C_{P}$','$C_{Xi}$','$C_{nooff}$','$C_{total}$']
    confidence_ellipse(RSDPower.b, RSDPower.f, labellist, Cov_PP[0:2,0:2], Cov_Xi[0:2,0:2], Cov_C_nooff[0:2,0:2], Cov_C[0:2,0:2], title = title, pdfname = pdfname)
    
    """
    filename = 'plots/RSD/ellipse/ConfidenceEllipseCov_r{:>3.2f}_{:>3.2f}_k{:>3.2f}_{:>3.2f}.txt'.format(RSDPower.rmin[rcut_max],RSDPower.rmax[rcut_min],RSDPower.kmin[kcut_min],RSDPower.kmax[kcut_max] )
    
    Cov_PP = np.reshape( Cov_PP[0:2,0:2], (1,4))
    Cov_Xi = np.reshape( Cov_Xi[0:2,0:2], (1,4))
    Cov_C_nooff = np.reshape( Cov_C_nooff[0:2,0:2], (1,4))
    Cov_C = np.reshape( Cov_C[0:2,0:2], (1,4))
    
    DAT = np.column_stack(( Cov_PP, Cov_Xi, Cov_C_nooff, Cov_C ))
    np.savetxt(filename, DAT, delimiter=" ", fmt="%s")
    """

def Plot_dPb(file1_name):

    import numpy as np
    import matplotlib.pyplot as plt

    file1 = open(file1_name)
    dPb1=np.array(np.loadtxt(file1))
    k = np.array(dPb1[:,0])
    dPb_data=np.array(dPb1[:,1])
    dPb_int = interp1d(k, dPb_data ,kind= "cubic")
    dPb_interpolated = np.array([dPb_int(RSDPower.kcenter[i]) for i in range(len(RSDPower.kcenter))])
    
    fig, (ax1, ax2) = plt.subplots( nrows = 1, ncols = 2, figsize = (14,7))

    ax1.loglog(RSDPower.kcenter, dPb_interpolated, 'b-', label='data : dlnk = 0.019, sdlnk = 0.0001')
    ax1.loglog(RSDPower.kcenter, RSDPower.dPb0, 'r--',label='current : dlnk = {:>1.3f}, sdlnk = {:>1.4f}'.format(RSDPower.dlnk, RSDPower.sdlnk[2]) )
    ax1.legend(loc=3, prop={'size':12})
    ax1.set_xlabel('k')
    ax1.set_ylabel('dP/db, zero')

    ax2.semilogx(RSDPower.kcenter,RSDPower.dPb0 - dPb_interpolated, 'b-', label='current - data' )
    ax2.legend(loc=3, prop={'size':12})
    ax2.set_xlabel('k')
    ax2.set_ylabel('current - data')

    from matplotlib.backends.backend_pdf import PdfPages
    pdfname = 'plots/compare/Compare_dPb0_r{:>1.3f}_k{:>1.3f}_sk{:>1.4f}.pdf'.format(RSDPower.dlnr, RSDPower.dlnk, RSDPower.sdlnk[2])
    pdf=PdfPages( pdfname )
    pdf.savefig(fig)
    pdf.close()
    print " pdf file saved : ", pdfname

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
    

    fig, (ax1, ax2, ax3) = plt.subplots( nrows = 1, ncols = 3, figsize = (20,7))
    plot_title ='dlnk = {}, dlnr = {}, sdlnk = {}, kN = {}, rN = {}, subN = {}'.format(RSDPower.dlnk, RSDPower.dlnr, RSDPower.sdlnk[2], RSDPower.n, RSDPower.n2, RSDPower.subN)

    fig.suptitle(plot_title)
    
    ax1.loglog(RSDPower.rcenter,TwoStep_dxib0, 'b-', label='Two, 0' )
    ax1.loglog(RSDPower.rcenter,RSDPower.dxib0, 'b--',label='One, 0' )
    ax1.loglog(RSDPower.rcenter,TwoStep_dxib2, 'r-',label='Two, 2' )
    ax1.loglog(RSDPower.rcenter,RSDPower.dxib2, 'r--' ,label='One, 2')
    ax1.loglog(RSDPower.rcenter,TwoStep_dxib4, 'g-',label='Two, 4' )
    ax1.loglog(RSDPower.rcenter,RSDPower.dxib4, 'g--',label='One, 4' )
    ax1.legend(loc=3)
    ax1.set_xlabel('r')
    ax1.set_ylabel('dxi/db')
    
    ax2.semilogx(RSDPower.rcenter,RSDPower.dxib0-TwoStep_dxib0, 'b-',label='0' )
    ax2.semilogx(RSDPower.rcenter,RSDPower.dxib2-TwoStep_dxib2, 'r-',label='2')
    ax2.semilogx(RSDPower.rcenter,RSDPower.dxib4-TwoStep_dxib4, 'g-',label='4' )
    ax2.set_xlabel('r')
    ax2.set_ylabel('One Step - Two Step')
    ax2.legend(loc=1)
    
    ax3.loglog(RSDPower.rcenter,np.fabs((RSDPower.dxib0-TwoStep_dxib0)/RSDPower.dxib0), 'b-',label='0' )
    ax3.loglog(RSDPower.rcenter,np.fabs((RSDPower.dxib2-TwoStep_dxib2)/RSDPower.dxib2), 'r-',label='2')
    ax3.loglog(RSDPower.rcenter,np.fabs((RSDPower.dxib4-TwoStep_dxib4)/RSDPower.dxib4), 'g-',label='4' )
    ax3.set_xlabel('r')
    ax3.set_ylabel('absolute((OneStep - TwoStep)/OneStep)')
    ax3.legend(loc=1)
    
    
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

def Plot_derivatives():
    
    import numpy as np
    import matplotlib.pyplot as plt


    analytic_data = RSDPower.derivative_P_analytic(0.0)
    print 'data obtained'
    file = open('plots/compare/dPd0_analytic.txt')
    Pkl=np.array(np.loadtxt(file))
    Power=np.array(Pkl[:,1])
    print "file load"
    Pm = interp1d(RSDPower.skcenter, Power ,kind= "cubic")
    print "interpolation over"
    analytic = np.array([Pm(RSDPower.kcenter[i]) for i in range(len(RSDPower.kcenter))])
    
    fig, (ax1, ax2) = plt.subplots( nrows = 1, ncols = 2, figsize = (14,7))
    
    ax1.loglog(RSDPower.kcenter,analytic, 'b-', label='Analytic' )
    ax1.loglog(RSDPower.kcenter,RSDPower.dPb0, 'b--',label='Numeric' )
    ax1.legend(loc=3)
    ax1.set_xlabel('r')
    ax1.set_ylabel('dP/db, zero')
    
    ax2.loglog(RSDPower.kcenter,np.fabs(RSDPower.dPb0 - analytic)/analytic, 'b-', label='Two, 0' )
    ax2.legend(loc=3)
    ax2.set_xlabel('r')
    ax2.set_ylabel('Analytic - Numeric')
    
    from matplotlib.backends.backend_pdf import PdfPages
    pdfname = 'plots/compare/Compare_Derivatives_r{:>1.3f}_k{:>1.3f}_sk{:>1.4f}.pdf'.format(RSDPower.dlnr, RSDPower.dlnk, RSDPower.sdlnk[2])
    pdf=PdfPages( pdfname )
    pdf.savefig(fig)
    pdf.close()
    print " pdf file saved : ", pdfname

def Reid_step2_kcut(l):
    
    
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
    l_r = len(RSDPower.rcenter)-1
    C_matrix3 = CombineCovariance3(l_r, matricesXi)
    C_matrix2 = CombineCovariance2(l_r, matricesXi)
    Xi, Xi2 = CombineDevXi3(l_r, matrices2Xi)
    
    
    """ F_bandpower """
    Fisher_bandpower_Xi = FisherProjection(Xi, C_matrix3)
    Fisher_bandpower_Xi_two = FisherProjection(Xi2, C_matrix2)
    cut = len(RSDPower.kcenter)
    
    part00 = Fisher_bandpower_Xi[0:cut, 0:cut]
    part02 = Fisher_bandpower_Xi[0:cut, cut:2*cut]
    part04 = Fisher_bandpower_Xi[0:cut, 2*cut:3*cut+1]
    part22 = Fisher_bandpower_Xi[cut:2*cut, cut:2*cut]
    part24 = Fisher_bandpower_Xi[cut:2*cut, 2*cut:3*cut+1]
    part44 = Fisher_bandpower_Xi[2*cut:3*cut+1, 2*cut:3*cut+1]
    
    part00_2 = Fisher_bandpower_Xi_two[0:cut, 0:cut]
    part02_2 = Fisher_bandpower_Xi_two[0:cut, cut:2*cut]
    part04_2 = Fisher_bandpower_Xi_two[0:cut, 2*cut:3*cut+1]
    part22_2 = Fisher_bandpower_Xi_two[cut:2*cut, cut:2*cut]
    
    part_list = [ part00[kcut_min:kcut_max+1,kcut_min:kcut_max+1], part02[kcut_min:kcut_max+1,kcut_min:kcut_max+1], part04[kcut_min:kcut_max+1,kcut_min:kcut_max+1], np.transpose(part02[kcut_min:kcut_max+1,kcut_min:kcut_max+1]), part22[kcut_min:kcut_max+1,kcut_min:kcut_max+1], part24[kcut_min:kcut_max+1,kcut_min:kcut_max+1], np.transpose(part04[kcut_min:kcut_max+1,kcut_min:kcut_max+1]), np.transpose(part24[kcut_min:kcut_max+1,kcut_min:kcut_max+1]), part44[kcut_min:kcut_max+1,kcut_min:kcut_max+1]]
    

    part_list2 = [part00_2[kcut_min:kcut_max+1,kcut_min:kcut_max+1], part02_2[kcut_min:kcut_max+1,kcut_min:kcut_max+1], part00_2[kcut_min:kcut_max+1,kcut_min:kcut_max+1], np.transpose(part02_2[kcut_min:kcut_max+1,kcut_min:kcut_max+1]), part22_2[kcut_min:kcut_max+1,kcut_min:kcut_max+1]]
    
    part_Fisher_bandpower_Xi = CombineCovariance3(l, part_list)
    part_Fisher_bandpower_Xi_two = CombineCovariance2(l, part_list2)
    
    """ dP/dq"""
    XP, XP2 = CombineDevXi(l, matrices2P)
    FisherXi = FisherProjection_Fishergiven(XP, part_Fisher_bandpower_Xi)
    FisherXi_two = FisherProjection_Fishergiven(XP2, part_Fisher_bandpower_Xi_two)
    
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

    #con_num = np.linalg.cond(FisherXi_two[0:2, 0:2])
    #print con_num
    
    rr = 1.15 * np.pi/RSDPower.kcenter[l]
    return rr, error_b_determin, error_f_determin, error_b_marginal, error_f_marginal\
    ,error_b_determin2, error_f_determin2, error_b_marginal2, error_f_marginal2

def Reid_step2(l):
    
    
    matricesXi = [RSDPower.covariance00, RSDPower.covariance02, RSDPower.covariance04, np.transpose(RSDPower.covariance02), RSDPower.covariance22, RSDPower.covariance24,np.transpose(RSDPower.covariance04), np.transpose(RSDPower.covariance24), RSDPower.covariance44]
    
    #matrices2P = [RSDPower.dPb0, RSDPower.dPb2, RSDPower.dPb4, RSDPower.dPf0, RSDPower.dPf2, RSDPower.dPf4, RSDPower.dPs0, RSDPower.dPs2, RSDPower.dPs4]
    
    matrices2P = [RSDPower.dPb0[kcut_min:kcut_max+1], RSDPower.dPb2[kcut_min:kcut_max+1],\
                  RSDPower.dPb4[kcut_min:kcut_max+1], RSDPower.dPf0[kcut_min:kcut_max+1],\
                  RSDPower.dPf2[kcut_min:kcut_max+1], RSDPower.dPf4[kcut_min:kcut_max+1],\
                  RSDPower.dPs0[kcut_min:kcut_max+1], RSDPower.dPs2[kcut_min:kcut_max+1],\
                  RSDPower.dPs4[kcut_min:kcut_max+1]]
                  
    Xizeros = np.zeros((len(RSDPower.kcenter),len(RSDPower.rcenter)))
    matrices2Xi = [RSDPower.dxip0, Xizeros,Xizeros,Xizeros,RSDPower.dxip2,Xizeros,Xizeros,Xizeros,RSDPower.dxip4]

    """ correlation covariance """
    C_matrix3 = CombineCovariance3(l, matricesXi)
    C_matrix2 = CombineCovariance2(l, matricesXi)

    Xi, Xi2 = CombineDevXi3(l, matrices2Xi)

    """ F_bandpower """
    part_Fisher_bandpower_Xi = FisherProjection(Xi, C_matrix3)
    part_Fisher_bandpower_Xi_two = FisherProjection(Xi2, C_matrix2)
    
    """ dP/dq"""
    l_k = len(RSDPower.kcenter)-1
    XP, XP2 = CombineDevXi(l_k, matrices2P)
    
    FisherXi = FisherProjection_Fishergiven(XP, part_Fisher_bandpower_Xi)
    FisherXi_two = FisherProjection_Fishergiven(XP2, part_Fisher_bandpower_Xi_two)
    
    error_b_marginal, error_f_marginal = FractionalError( RSDPower.b, RSDPower.f, inv(FisherXi))
    error_b_determin, error_f_determin = FractionalError( RSDPower.b, RSDPower.f, inv(FisherXi[0:2,0:2]))
    error_b_marginal2, error_f_marginal2 = FractionalError( RSDPower.b, RSDPower.f, inv(FisherXi_two))
    error_b_determin2, error_f_determin2 = FractionalError( RSDPower.b, RSDPower.f, inv(FisherXi_two[0:2,0:2]))

    rr = RSDPower.rcenter[l]
    return rr, error_b_determin, error_f_determin, error_b_marginal, error_f_marginal,error_b_determin2, error_f_determin2, error_b_marginal2, error_f_marginal2

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


Reid_step2_kcut = np.vectorize(Reid_step2_kcut)
Reid_step2 = np.vectorize(Reid_step2)
Reid_step1 = np.vectorize(Reid_step1)
error_only_CP = np.vectorize(error_only_CP)

if __name__=='__main__':
    main()



