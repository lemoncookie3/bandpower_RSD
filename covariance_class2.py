import numpy as np
from numpy import zeros, sqrt, pi, sin, cos, exp
from numpy.linalg import pinv, inv
from numpy import vectorize
from scipy.interpolate import interp1d
from scipy.integrate import simps, quad


class error_analysis():
    
    """ No redshift, Not completed yet"""

    def __init__(self, KMIN, KMAX, RMIN, RMAX, n, n2, subN):
        
        #parameter
        self.h=1.0
        self.Vs= 5.0*10**9 # survey volume
        self.nn= 3.0 * 10**(-4) # shot noise : \bar{n}
        
        # k scale range
        self.KMIN = KMIN
        self.KMAX = KMAX
        
        # r scale
        self.RMIN = RMIN #30.#0.1 #0.1 for Reid   #1.15 * np.pi / self.kmax
        self.RMAX = RMAX #180. #180. for Reid  #1.15 * np.pi / self.kmin
        
        self.n = n # converge # 201 for Reid # 201 # the number of k bins. the sample should be odd
        self.n2 = n2 #101 for Reid # number of r bins
        self.subN = subN
        #self.klist = np.logspace(np.log(self.KMIN),np.log(self.KMAX),self.n, base = np.e)
        
        
        self.rlist = np.logspace(np.log(self.RMAX),np.log(self.RMIN),self.n2, base = np.e)
        rlist = self.rlist
        self.rmax = np.delete(rlist,-1)
        self.rmin = np.delete(rlist,0)
        self.rcenter = np.array([ (rlist[i] + rlist[i+1])/2. for i in range(len(rlist)-1) ])
        
        self.dr = np.abs(self.rmax - self.rmin)
        self.dlnr = np.log(self.rmax/self.rmin)[3]
   
   
        self.skbin = np.logspace(np.log10(self.KMIN),np.log10(self.KMAX), subN * self.n + 1, base=10) #For Reid, delete '+1', not relevant anymore
        self.skcenter = np.array([(self.skbin[i] + self.skbin[i+1])/2. for i in range(len(self.skbin)-1)])
        self.skmin = np.delete(self.skbin,-1)
        self.skmax = np.delete(self.skbin,0)
        self.sdlnk = np.log(self.skmax/self.skmin)
        
        self.klist = np.array([self.skbin[i*subN] for i in range(len(self.skbin)/subN + 1)]) #For Reid, delete '+1', not relevant anymore
        
        
        self.kcenter = np.array([(self.klist[i] + self.klist[i+1])/2. for i in range(len(self.klist)-1)])
        self.kmin = np.delete(self.klist,-1)
        self.kmax = np.delete(self.klist,0)
        self.dk = self.kmax - self.kmin
        self.dlnk = np.log(self.kmax/self.kmin)[3]



    def compile_fortran_modules(self):
        
        import numpy.f2py.f2py2e as f2py2e
        import sys
        sys.argv +=  "-c -m sici sici.f90".split()
        f2py2e.main()
        sys.argv = [sys.argv[0]]
        sys.argv +=  "-c -m sbess sbess.f90".split()
        f2py2e.main()
        sys.argv = [sys.argv[0]]
        sys.argv +=  "-c -m legen legen.f90".split()
        f2py2e.main()
        sys.argv = [sys.argv[0]]


    def MatterPower(self, file):
        
        #Pkl=np.array(np.loadtxt('matterpower_z_0.55.dat')) # z=0.55
        Pkl=np.array(np.loadtxt(file))
        k=np.array(Pkl[:,0])
        P=np.array(Pkl[:,1])
        
        #power spectrum interpolation
        Pm = interp1d(k, P ,kind= "cubic") #matterpower
        #self.Pmlist = Pm(self.kcenter)
        
        #REAL POWERSPECTRUM DATA
        self.RealPowerBand = np.array([Pm(self.skcenter[i]) for i in range(len(self.skcenter))])


    def Shell_avg_band( self ):
        #
        #   Shell_avg_band
        #
        from scipy.integrate import simps
        powerspectrum = self.RealPowerBand
        skbin = self.skbin
        skcenter = self.skcenter
        kcenter = self.kcenter
        dk = self.dk
        
        Vi = 4 * pi * kcenter**2 * dk + 1./3 * pi * (dk)**3
        
        resultlist=[]
        for i in range(len(kcenter)):
            k = skcenter[i*self.subN:i*self.subN+self.subN]
            data = powerspectrum[i*self.subN:i*self.subN+self.subN]
            
            result = simps(4* np.pi * k**2 * data, k )
            resultlist.append(result)
        self.Pmlist = resultlist/Vi
        return self.Pmlist


    def PowerCovMatrix( self ):
        # NO RSD
        # func for constructing Power Spectrum covariance matrix
        # C(k,k')
        """ note: survey volume definition """
        kcenter, dlnk, Vs, nn = self.kcenter, self.dlnk, self.Vs, self.nn
        dk = self.dk
        Shell_avg_bandpower = self.Shell_avg_band()
        
        V = 4./3 * np.pi * self.RMAX**3
        PowerCovariance = np.zeros((len(kcenter),len(kcenter)))
        Vi = 4 * pi * kcenter**2 * dk + 1./3 * pi * (dk)**3
        PowerCov = 2./V * (Shell_avg_bandpower + 1./nn)**2 /Vi * (2 * pi)**3
        #PowerCov = (Shell_avg_bandpower + 1./nn)**2 / Vi
        for i in range(len(kcenter)):
            PowerCovariance[i,i] = PowerCov[i]
        
        self.PowerCovariance = PowerCovariance
        return self.PowerCovariance






class RSD_covariance(error_analysis):

    def __init__(self, KMIN, KMAX, RMIN, RMAX, n, n2, subN):
        error_analysis.__init__(self, KMIN, KMAX, RMIN, RMAX, n, n2, subN)
        
        # RSD parameter

        self.b=2.0
        self.f=0.74
        self.s= 3.5  # sigma in Power spectrum

        self.n3 = 31 # 101 for Reid number of mu bins
        self.mulist = np.linspace(-1.,1.,self.n3)
        
        
        print '\n-------------------------------------------------------------------\
        \n < RSD : Fractional Error for parameter b, f > \
        \nz = 0.0\
        \nnumber of k bins n ={}, subN = {}\
        \nnumber of r bins n2 = {} \
        \ndlnr = {}, dlnk={}, sdlnk={}'.format(self.n, self.subN, self.n2, np.log(self.rlist[1]/self.rlist[2]),self.dlnk, self.sdlnk[2] )



    """
    def RSDPowerFisher(self):
    
        skbin = self.skbin
        kcenter = self.kcenter
        dk = self.dk
        powerspectrum = self.RealPowerBand
        Pmlist = self.Pmlist
        
        matrix1, matrix2 = np.mgrid[0:len(self.mulist), 0:self.subN ]
        mumatrix = self.mulist[matrix1]
        
        resultlist=[]
        for i in range(len(kcenter)):
            k = skbin[i*self.subN:i*self.subN+self.subN]
            kmatrix = k[matrix2]
            Dmatrix = exp( - kmatrix**2 * mumatrix**2 * self.s**2) #FOG matrix
            R = (self.b + self.f * mumatrix**2)**2 * Dmatrix
            muint = simps( kmatrix**2 * R**2 * (Pmlist[i] * R + 1./self.nn)**(-2), mumatrix, axis=0 )
            result = simps( muint, k )
            resultlist.append(result)
        self.RSDPowerFishermatrix = np.zeros((len(kcenter), len(kcenter)))
        np.fill_diagonal( self.RSDPowerFishermatrix, self.Vs/(8 * pi**2) * np.array(resultlist) )
        
        return self.RSDPowerFishermatrix
    """
    




    def multipole_P(self,l):
    
        b = self.b
        f = self.f
        s = self.s

        klist = self.klist
        kcenter = self.kcenter
        skcenter = self.skcenter
        skbin = self.skbin
        dk = self.dk
        mulist = self.mulist
        dlnk = self.dlnk
        Pmlist = 1. #self.Shell_avg_band()
        matterpower = self.RealPowerBand
        
        matrix1, matrix2 = np.mgrid[0:len(mulist),0:self.subN]
        mumatrix = self.mulist[matrix1]
        Le_matrix = Ll(l,mumatrix)

        Vi = 4 * pi * kcenter**2 * dk + 1./3 * pi * (dk)**3
                
        resultlist=[]
        for i in range(len(kcenter)):
            k = skcenter[i*self.subN : i*self.subN+self.subN]
            Pm = matterpower[i*self.subN : i*self.subN+self.subN]
            kmatrix = k[matrix2]
            Pmmatrix = Pm[matrix2]
            Dmatrix = np.exp(- kmatrix**2 * mumatrix**2 * self.s**2) #FOG matrix
            R = (b + f * mumatrix**2)**2 * Dmatrix * Le_matrix
            muint = (2 * l + 1.)/2 * simps( 4 * pi * kmatrix**2 * Pmmatrix * R, mumatrix, axis=0 )
            result = simps( muint, k )
            resultlist.append(result)
        return resultlist/Vi
    
    def multipole_bandpower_all(self):
    
        self.multipole_bandpower0 = self.multipole_P(0.0)
        self.multipole_bandpower2 = self.multipole_P(2.0)
        self.multipole_bandpower4 = self.multipole_P(4.0)
        self.multipole_bandpower = np.concatenate((self.multipole_bandpower0, self.multipole_bandpower2,self.multipole_bandpower4), axis=0)

    
    def RSDband_derivative_P(self,l):
        
        """ dP_l/dq """
  
        b = self.b
        f = self.f
        s = self.s
        
        klist = self.klist
        kcenter = self.kcenter
        skbin = self.skbin
        skcenter = self.skcenter
        dk = self.dk
        mulist = self.mulist
        dlnk = self.dlnk
        Pmlist = self.Pmlist
        matterpower = self.RealPowerBand
        
        matrix1, matrix2 = np.mgrid[0:len(mulist),0:self.subN]
        mumatrix = self.mulist[matrix1]
        Le_matrix = Ll(l,mumatrix)
        
        Vi = 4 * pi * kcenter**2 * dk + 1./3 * pi * (dk)**3
        
        resultlistb=[]
        resultlistf=[]
        resultlists=[]
        for i in range(len(kcenter)):
            k = skcenter[i*self.subN : i*self.subN+self.subN]
            Pm = matterpower[i*self.subN : i*self.subN+self.subN]
            kmatrix = k[matrix2]
            Pmmatrix = Pm[matrix2]
            Dmatrix = np.exp(- kmatrix**2 * mumatrix**2 * self.s**2) #FOG matrix
            Rb = 2 *(self.b + self.f * mumatrix**2) * Dmatrix #* Le_matrix
            Rf = 2 * mumatrix**2 *(self.b + self.f * mumatrix**2) * Dmatrix #* Le_matrix
            Rs = (- kmatrix**2 * mumatrix**2)*(self.b + self.f * mumatrix**2)**2 * Dmatrix# * Le_matrix
            Rintb = simps( Rb * Le_matrix, mumatrix, axis=0 )
            Rintf = simps( Rf * Le_matrix, mumatrix, axis=0 )
            Rints = simps( Rs * Le_matrix, mumatrix, axis=0 )
            resultb = (2 * l + 1.)/2 * simps( 4 * pi * k**2 * Pm * Rintb, k )
            resultf = (2 * l + 1.)/2 * simps( 4 * pi * k**2 * Pm * Rintf, k )
            results = (2 * l + 1.)/2 * simps( 4 * pi * k**2 * Pm * Rints, k )

            resultlistb.append(resultb)
            resultlistf.append(resultf)
            resultlists.append(results)
        return resultlistb/Vi, resultlistf/Vi, resultlists/Vi



    def RSDband_derivative_P_All(self):
    
        self.dPb0, self.dPf0, self.dPs0 = self.RSDband_derivative_P(0.0)
        self.dPb2, self.dPf2, self.dPs2 = self.RSDband_derivative_P(2.0)
        self.dPb4, self.dPf4, self.dPs4 = self.RSDband_derivative_P(4.0)
        print "RSDband_derivative_P_All (dP/db, dP/df, dP/ds)"



    def derivative_P_analytic(self, l):
    
        from scipy.integrate import quad
        self.kcenter
        self.skcenter
        Pm = self.RealPowerBand
        
        b = self.b
        f = self.f
        s = self.s
        
        D = lambda k,mu : np.exp(- k*k*mu*mu*s*s)
        R = lambda k, mu :  0.5 * 2 * (b + f * mu**2) * D(k,mu)
    
        const = (2*l + 1)/2.
        multipole = []
        for i in range(len(self.skcenter)):
            k = self.skcenter
            re, err = quad( lambda mu : R(k[i], mu) * D( k[i], mu ), -1, 1 )
            result = Pm[i] * re
            multipole.append(result)
        
        DAT = np.column_stack(( self.skcenter, multipole))
        np.savetxt('plots/compare/dPd0_analytic.txt', DAT, delimiter=" ", fmt="%s")
        return multipole
    
    
    
    
    
    
    def _(self):
    #
    #   dP_l/dq
    #
    #
    
        b = self.b
        f = self.f
        s = self.s
    
        klist = self.klist
        kcenter = self.kcenter
        skbin = self.skbin
        skcenter = self.skcenter
        dk = self.dk
        mulist = self.mulist
        dlnk = self.dlnk
        Pmlist = self.Pmlist
        matterpower = self.RealPowerBand
    
        matrix1, matrix2 = np.mgrid[0:len(mulist),0:self.subN]
        mumatrix = self.mulist[matrix1]
        Le_matrix0 = Ll(0.0, mumatrix)
        Le_matrix2 = Ll(2.0, mumatrix)
        Le_matrix4 = Ll(4.0, mumatrix)
    
        Vi = 4 * pi * kcenter**2 * dk + 1./3 * pi * (dk)**3
    
        resultlistb0=[]
        resultlistf0=[]
        resultlists0=[]
        resultlistb2=[]
        resultlistf2=[]
        resultlists2=[]
        resultlistb4=[]
        resultlistf4=[]
        resultlists4=[]
        
        for i in range(len(kcenter)):
            k = skcenter[i*self.subN : i*self.subN+self.subN]
            Pm = matterpower[i*self.subN : i*self.subN+self.subN]
            kmatrix = k[matrix2]
            Pmmatrix = Pm[matrix2]
            Dmatrix = np.exp(- kmatrix**2 * mumatrix**2 * self.s**2) #FOG matrix
            Rb = 2 *(self.b + self.f * mumatrix**2) * Dmatrix #* Le_matrix
            Rf = 2 * mumatrix**2 *(self.b + self.f * mumatrix**2) * Dmatrix #* Le_matrix
            Rs = (- kmatrix**2 * mumatrix**2)*(self.b + self.f * mumatrix**2)**2 * Dmatrix# * Le_matrix
            Rintb0 = simps( Rb * Le_matrix0, mumatrix, axis=0 )
            Rintf0 = simps( Rf * Le_matrix0, mumatrix, axis=0 )
            Rints0 = simps( Rs * Le_matrix0, mumatrix, axis=0 )
            Rintb2 = simps( Rb * Le_matrix2, mumatrix, axis=0 )
            Rintf2 = simps( Rf * Le_matrix2, mumatrix, axis=0 )
            Rints2 = simps( Rs * Le_matrix2, mumatrix, axis=0 )
            Rintb4 = simps( Rb * Le_matrix4, mumatrix, axis=0 )
            Rintf4 = simps( Rf * Le_matrix4, mumatrix, axis=0 )
            Rints4 = simps( Rs * Le_matrix4, mumatrix, axis=0 )
            
            resultb0 = (1.)/2 * simps( 4 * pi * k**2 * Pm * Rintb0, k )
            resultf0 = (1.)/2 * simps( 4 * pi * k**2 * Pm * Rintf0, k )
            results0 = (1.)/2 * simps( 4 * pi * k**2 * Pm * Rints0, k )
            resultb2 = (5.)/2 * simps( 4 * pi * k**2 * Pm * Rintb2, k )
            resultf2 = (5.)/2 * simps( 4 * pi * k**2 * Pm * Rintf2, k )
            results2 = (5.)/2 * simps( 4 * pi * k**2 * Pm * Rints2, k )
            resultb4 = (9.)/2 * simps( 4 * pi * k**2 * Pm * Rintb4, k )
            resultf4 = (9.)/2 * simps( 4 * pi * k**2 * Pm * Rintf4, k )
            results4 = (9.)/2 * simps( 4 * pi * k**2 * Pm * Rints4, k )
        
            """
            Rintb = (2 * l + 1.)/2 * simps( 4 * pi * kmatrix**2 * Rb * Pmmatrix, mumatrix, axis=0 )
            Rintf = (2 * l + 1.)/2 * simps( 4 * pi * kmatrix**2 * Rf * Pmmatrix, mumatrix, axis=0 )
            Rints = (2 * l + 1.)/2 * simps( 4 * pi * kmatrix**2 * Rs * Pmmatrix, mumatrix, axis=0 )
            resultb = simps( muintb, k )
            resultf = simps( muintf, k )
            results = simps( muints, k )
            """
            resultlistb0.append(resultb0)
            resultlistf0.append(resultf0)
            resultlists0.append(results0)
            resultlistb2.append(resultb2)
            resultlistf2.append(resultf2)
            resultlists2.append(results2)
            resultlistb4.append(resultb4)
            resultlistf4.append(resultf4)
            resultlists4.append(results4)
        
        self.dPb0 = resultlistb0/Vi
        self.dPf0 = resultlistf0/Vi
        self.dPs0 = resultlists0/Vi
        self.dPb2 = resultlistb2/Vi
        self.dPf2 = resultlistf2/Vi
        self.dPs2 = resultlists2/Vi
        self.dPb4 = resultlistb4/Vi
        self.dPf4 = resultlistf4/Vi
        self.dPs4 = resultlists4/Vi
                
        return self.dPb0, self.dPf0, self.dPs0, self.dPb2, self.dPf2, self.dPs2, self.dPb4, self.dPf4, self.dPs4


    def derivative_P_band(self, l):
        #
        #   Shell averaged
        #
    
        b = self.b
        f = self.f
        s = self.s
    
        klist = self.klist
        kcenter = self.kcenter
        skcenter = self.skcenter
        skbin = self.skbin
        dk = self.dk
        mulist = self.mulist
        dlnk = self.dlnk
        Pmlist = 1. #self.Shell_avg_band()
        matterpower = self.RealPowerBand
    
        matrix1, matrix2 = np.mgrid[0:len(mulist),0: self.subN ]
        mumatrix = self.mulist[matrix1]
        Le_matrix0 = Ll(0.0,mumatrix)
        Le_matrix2 = Ll(2.0,mumatrix)
        Le_matrix4 = Ll(4.0,mumatrix)
        
        
        Vi = 4 * pi * kcenter**2 * dk + 1./3 * pi * (dk)**3
    
        resultlist=[]
        for i in range(len(kcenter)):
            k = skcenter[i*self.subN : i*self.subN+self.subN]
            Pm = matterpower[i*self.subN : i*self.subN+self.subN]
            kmatrix = k[matrix2]
            Dmatrix = np.exp(- kmatrix**2 * mumatrix**2 * self.s**2) #FOG matrix
            R = (b + f*mumatrix**2)**2 * Dmatrix

            Rintegral = (2*l + 1.)/2 * simps( R * Le_matrix, mumatrix, axis = 0 )
            result = simps( 4 * pi * k**2 * Pm * Rintegral , k )

            resultlist.append(result)

        derivative_P_bandpower = np.zeros((len(kcenter), len(kcenter)))
        np.fill_diagonal(derivative_P_bandpower, Pmlist * np.array(resultlist)/Vi)
        
        return derivative_P_bandpower
        

    def derivative_Xi_band(self, l):
        #
        #   Shell averaged
        #   dxi_l / dp_li = i^l int(k^2 ShellavgBessel(kr) / 2pi^2, kmin, kmax)
        #
        import cmath
        I = cmath.sqrt(-1)

        klist = self.klist
        kcenter = self.kcenter
        skbin = self.skbin
        dk = self.dk
        mulist = self.mulist
        dlnk = self.dlnk
        rcenter = self.rcenter
        rmin = self.rmin
        rmax = self.rmax
        dr = self.dr
        
        matrix1, matrix2 = np.mgrid[ 0:len(mulist), 0: self.subN ]
        matrix3, matrix4 = np.mgrid[ 0:len(rcenter), 0: self.subN ]
        mumatrix = self.mulist[matrix1]
        rminmatrix = rmin[matrix3]
        rmaxmatrix = rmax[matrix3]
        Le_matrix = Ll(l,mumatrix)
        rmatrix = rcenter[matrix3]
        drmatrix = dr[matrix3]

        Vir = 4 * pi * rmatrix**2 * drmatrix + 1./3 * pi * (drmatrix)**3
        
        resultlist=[]
        for i in range(len(kcenter)):
            k = skbin[i*self.subN : i*self.subN+self.subN]
            kmatrix = k[matrix2]
            Dmatrix = np.exp( - kmatrix**2 * mumatrix**2 * self.s**2) #FOG matrix
            kmatrix2 = k[matrix4]
            AvgBesselmatrix = avgBessel(l, kmatrix2 ,rminmatrix,rmaxmatrix)/Vir
            
            result = np.real(I**l) * simps( kmatrix2**2 * AvgBesselmatrix /(2*pi**2) , k, axis = 1 )
            resultlist.append(result)
        
        derivative_Xi_bandpower = np.array(resultlist).reshape((len(kcenter),len(rcenter)))
        return derivative_Xi_bandpower
            
            
    def derivative_Xi_band_All(self):
    
        self.dxip0 = self.derivative_Xi_band( 0.0 )
        self.dxip2 = self.derivative_Xi_band( 2.0 )
        self.dxip4 = self.derivative_Xi_band( 4.0 )
    
        print "derivative_Xi_band_All (dxi/dp)"
    


    def covariance_PP(self, l1, l2):

        from scipy.integrate import quad,simps
        from numpy import zeros, sqrt, pi, exp
        import cmath
        I = cmath.sqrt(-1)

        klist = self.klist
        kcenter = self.kcenter

        skbin = self.skbin
        mulist = self.mulist
        dk = self.dk
        dlnk = self.dlnk
        Pmlist = self.Pmlist
        matterpower = self.RealPowerBand
        s = self.s
        b = self.b
        f = self.f
        Vs = self.Vs
        nn = self.nn


        Vi = 4 * pi * kcenter**2 * dk + 1./3 * pi * (dk)**3
        
        # FirstTerm + SecondTerm
        matrix1, matrix2 = np.mgrid[0:len(mulist),0:len(kcenter)]
        mumatrix = self.mulist[matrix1]
        kmatrix = kcenter[matrix2]
        Pmlistmatrix = Pmlist[matrix2]

        Le_matrix1 = Ll(l1,mumatrix)
        Le_matrix2 = Ll(l2,mumatrix)
        
        Dmatrix = np.exp(- kmatrix**2 * mumatrix**2 * self.s**2)
        Const_alpha = (2*l1 + 1.) * (2*l2 + 1.) /Vs * (2*pi)**3
        R = (self.b + self.f * mumatrix**2)**2 * Dmatrix
        FirstTerm = Const_alpha * simps( (Pmlistmatrix**2 * R**2 + 2./nn * Pmlistmatrix * R) * Le_matrix1 * Le_matrix2, mumatrix, axis=0 )/Vi

        if l1 == l2: LastTerm = (2*l1 + 1) * 2/ nn /Vs / Vi * (2 * pi)**3
        else : LastTerm = 0.0

        Total = FirstTerm + LastTerm

        covariance_mutipole_PP = np.zeros((len(kcenter),len(kcenter)))
        np.fill_diagonal(covariance_mutipole_PP,Total)

        #print 'covariance_PP {:>1.0f}{:>1.0f} is finished'.format(l1,l2)
        return covariance_mutipole_PP
            
            

    def RSDband_covariance_PP(self, l1, l2):
        #
        #
        #
        #
        from scipy.integrate import quad,simps
        from numpy import zeros, sqrt, pi, exp
        import cmath
        I = cmath.sqrt(-1)
    
        klist = self.klist
        kcenter = self.kcenter
        skbin = self.skbin
        mulist = self.mulist
        dk = self.dk
        dlnk = self.dlnk
        Pmlist = 1 #self.Pmlist
        matterpower = self.RealPowerBand
        s = self.s
        b = self.b
        f = self.f
        Vs = self.Vs
        nn = self.nn
    
        # FirstTerm + SecondTerm
        matrix1, matrix2 = np.mgrid[0:len(mulist),0:self.subN]
        mumatrix = self.mulist[matrix1]
        
        Le_matrix1 = Ll(l1,mumatrix)
        Le_matrix2 = Ll(l2,mumatrix)
        Vi = 4 * pi * kcenter**2 * dk + 1./3 * pi * (dk)**3
        
        Const_alpha = (2*l1 + 1.) * (2*l2 + 1.) * (2*pi)**3 /Vs
       
        resultlist1 = []
        resultlist2 = []
        for i in range(len(kcenter)):
            k = skbin[i*self.subN : i*self.subN+self.subN]
            Pm = matterpower[i*self.subN : i*self.subN+self.subN]
            kmatrix = k[matrix2]
            #Pmmatrix = Pm[matrix2]
            Dmatrix = np.exp(- kmatrix**2 * mumatrix**2 * self.s**2) #FOG matrix
            R = (self.b + self.f * mumatrix**2)**2 * Dmatrix
            Rintegral3 =  simps(  R**2 * Le_matrix1 * Le_matrix2, mumatrix, axis=0 )
            Rintegral2 =  simps(  R * Le_matrix1 * Le_matrix2, mumatrix, axis=0 )
            result1 = simps( 4 * pi * k**2 * Pm**2 * Rintegral3, k )
            result2 = simps( 4 * pi * k**2 * Pm * Rintegral2, k )
            resultlist1.append(result1)
            resultlist2.append(result2)
        FirstTerm = Const_alpha * Pmlist**2 * np.array(resultlist1)/Vi**2
        SecondTerm = Const_alpha * Pmlist* 2./nn * np.array(resultlist2)/Vi**2
        
        # LastTerm
        
        if l1 == l2:
            LastTerm = (2*l1 + 1.) * 2. * (2 * pi)**3/Vs/nn**2 /Vi
        else:
            LastTerm = 0.
        
        Total = FirstTerm + SecondTerm + LastTerm
        covariance_mutipole_PP = np.zeros((len(kcenter),len(kcenter)))
        np.fill_diagonal(covariance_mutipole_PP,Total)

        #print 'covariance_PP {:>1.0f}{:>1.0f} is finished'.format(l1,l2)
        return covariance_mutipole_PP
  
  
    def RSDband_covariance_PP_All(self):
    
        self.covariance_PP00 = np.array(self.RSDband_covariance_PP(0.0,0.0))
        self.covariance_PP02 = np.array(self.RSDband_covariance_PP(0.0,2.0))
        self.covariance_PP04 = np.array(self.RSDband_covariance_PP(0.0,4.0))
        self.covariance_PP22 = np.array(self.RSDband_covariance_PP(2.0,2.0))
        self.covariance_PP24 = np.array(self.RSDband_covariance_PP(2.0,4.0))
        self.covariance_PP44 = np.array(self.RSDband_covariance_PP(4.0,4.0))
  
        print 'covariance_PP_All is finished (l = 0,2,4)'
    
  
    def RSDband_covariance_PXi( self, l1, l2 ):


        import numpy as np
        from numpy import zeros, sqrt, pi, sin, cos, exp
        from numpy.linalg import inv
        from numpy import vectorize
        from scipy.interpolate import interp1d
        from scipy.integrate import simps, quad
        
        import cmath
        I = cmath.sqrt(-1)
    
        klist = self.klist
        kcenter = self.kcenter
        skbin = self.skbin
        skcenter =self.skcenter
        rlist = self.rlist
        rcenter = self.rcenter
        dr = self.dr
        rmin = self.rmin
        rmax = self.rmax
        mulist = self.mulist
        dk = self.dk
        dlnk = self.dlnk
        dr = self.dr
        Pmlist = self.Pmlist
        Pm = self.RealPowerBand
        s = self.s
        b = self.b
        f = self.f
        Vs = self.Vs
        nn = self.nn
    
        # FirstTerm + SecondTerm
        matrix1, matrix2, matrix3 = np.mgrid[0:len(mulist),0:self.subN, 0: len(rcenter)]
        matrix4, matrix5 = np.mgrid[ 0:self.subN, 0: len(rcenter)]
        mumatrix = mulist[matrix1]
        rminmatrix = rmin[matrix5]
        rmaxmatrix = rmax[matrix5]
        rmatrix = rcenter[matrix5]
        drmatrix = dr[matrix5]
    
    
    
        Le_matrix1 = Ll(l1,mumatrix)
        Le_matrix2 = Ll(l2,mumatrix)

        Const_beta = np.real(I**(l2)) * (2*l1 + 1.) * (2*l2 + 1.)/Vs

        resultlist1 = []
        resultlist2 = []
        resultlist3 = []
        for i in range(len(kcenter)):
            k = skcenter[i*self.subN : i*self.subN+self.subN]
            kmatrix = k[matrix2] # 3D
            Dmatrix = np.exp(- kmatrix**2 * mumatrix**2 * self.s**2) #FOG matrix
                
            R = (self.b + self.f * mumatrix**2)**2 * Dmatrix
            Rintegral3 = simps( R**2 * Le_matrix1 * Le_matrix2, mumatrix, axis= 0 )
            Rintegral2 = simps( R * Le_matrix1 * Le_matrix2, mumatrix, axis= 0 )
            
            kmatrix2 = k[matrix4] # 2D
            Pmmatrix = 1 #Pm[matrix4]
            Vir = 4 * pi * rmatrix**2 * drmatrix + 1./3 * pi * (drmatrix)**3
            AvgBesselmatrix = avgBessel(l2,kmatrix2,rminmatrix,rmaxmatrix)/Vir #2D
            result1 = simps( 4 * pi * kmatrix2**2 * Pmmatrix**2 * Rintegral3 * AvgBesselmatrix, k, axis=0 )
            result2 = simps( 4 * pi * kmatrix2**2 * 2./nn * Pmmatrix * Rintegral2 * AvgBesselmatrix, k, axis=0 )
            result3 = simps( 4 * pi * kmatrix2**2 * AvgBesselmatrix, k, axis=0 )
            
            resultlist1.append(result1)
            resultlist2.append(result2)
            resultlist3.append(result3)
    
        matrix1, matrix2 = np.mgrid[0:len(kcenter), 0: len(rcenter)]
        Pmlistmatrix = Pmlist[matrix1]
        kmatrix = kcenter[matrix1]
        dkmatrix = dk[matrix1]
        Vi = 4 * pi * kmatrix**2 * dkmatrix + 1./3 * pi * (dkmatrix)**3
                
        FirstTerm = Const_beta * Pmlistmatrix**2 * np.array(resultlist1).reshape((len(kcenter), len(rcenter)))/Vi
        SecondTerm = Const_beta * Pmlistmatrix * np.array(resultlist2).reshape((len(kcenter), len(rcenter)))/Vi
        
        if l1 == l2 :
            LastTerm = np.real(I**(l2)) * (2*l2 + 1.)*2 /Vs/nn**2 * np.array(resultlist3).reshape((len(kcenter), len(rcenter))) / Vi
        else : LastTerm = 0.
        
        covariance_multipole_PXi =  FirstTerm + SecondTerm + LastTerm
    
        print 'covariance_PXi {:>1.0f}{:>1.0f} is finished'.format(l1,l2)
        return covariance_multipole_PXi


    def RSDband_covariance_PXi_All(self):
        
        import pp, sys, time
        """
        from multiprocessing import Process, Queue
        
        def PXi_process(q, order, (l1, l2)):
            re = self.RSDband_covariance_PXi(l1, l2)
            q.put((order, re))
        
        inputs = ((0.0, 0.0),(0.0, 2.0),(0.0, 4.0),(2.0, 0.0),(2.0, 2.0),(2.0, 4.0),(4.0, 0.0),(4.0, 2.0),(4.0, 4.0))
        q = Queue()
        Processes = [Process(target = PXi_process, args=(q, z[0], z[1])) for z in zip(range(9), inputs)]
        for p in Processes: p.start()
        result = [q.get() for p in Processes]
        result.sort()
        result1 = [result[1] for r in result ]
        result2 = [result[0] for r in result ]
        print result2
        
        """
        ppservers = ()
        #ppservers = ("140.254.91.255","",)
    
        if len(sys.argv) > 1:
            ncpus = int(sys.argv[1])
            # Creates jobserver with ncpus workers
            job_server = pp.Server(ncpus, ppservers=ppservers)
        else:
            # Creates jobserver with automatically detected number of workers
            job_server = pp.Server(ppservers=ppservers)
        print "\nStarting pp with", job_server.get_ncpus(), "workers"
    
        inputs1 = ((0.0, 0.0,),(0.0, 2.0,),(0.0, 4.0,),(2.0, 0.0,),(2.0, 2.0,),(2.0, 4.0,),(4.0, 0.0,),(4.0, 2.0,),(4.0, 4.0,))
        jobs1 = [ job_server.submit(self.RSDband_covariance_PXi, input, (Ll,avgBessel,)) for input in inputs1]
        result1=[]
        for job in jobs1:
            re = job()
            result1.append(re)
        
        self.covariance_PXi00 = result1[0]
        self.covariance_PXi02 = result1[1]
        self.covariance_PXi04 = result1[2]
        self.covariance_PXi20 = result1[3]
        self.covariance_PXi22 = result1[4]
        self.covariance_PXi24 = result1[5]
        self.covariance_PXi40 = result1[6]
        self.covariance_PXi42 = result1[7]
        self.covariance_PXi44 = result1[8]
    
    

    def derivative_xi(self,l):
        """ Output : dXi/db, dXi/df, dXi/ds three 1d array\
            along r axis for each mode l ( l = 0,2,4 )"""
    
        import cmath
        I = cmath.sqrt(-1)
        
        b = self.b
        f = self.f
        s = self.s
        rlist = self.rlist
        rcenter = self.rcenter
        klist = self.klist
        kcenter = self.kcenter
        skcenter = self.skcenter
        mulist = self.mulist
        dlnk = self.dlnk
        sdlnk = self.sdlnk
        Pmlist = self.Shell_avg_band()
        matterpower = self.RealPowerBand
        rmin = self.rmin
        rmax = self.rmax
        dr = self.dr
        
        Const = (2*l + 1.)/2 * np.real(I**l) / (2 * pi**2)
        
        matrix1, matrix2 = np.mgrid[ 0:len(mulist), 0:len(skcenter) ]
        matrix4, matrix5 = np.mgrid[ 0:len(skcenter), 0: len(rcenter)]
        kmatrix = skcenter[matrix2]
        mumatrix = self.mulist[matrix1]
        
        rminmatrix = rmin[matrix5]
        rmaxmatrix = rmax[matrix5]
        rmatrix = rcenter[matrix5]
        drmatrix = dr[matrix5]
        Le_matrix = Ll(l, mumatrix)
        
        
        Dmatrix = np.exp(- kmatrix**2 * mumatrix**2 * self.s**2) #2D
        Rb = 2 *(self.b + self.f * mumatrix**2) * Dmatrix
        Rf = 2 * mumatrix**2 *(self.b + self.f * mumatrix**2) * Dmatrix
        Rs = (- kmatrix**2 * mumatrix**2)*(self.b + self.f * mumatrix**2)**2 * Dmatrix
        
        muintb = simps( Rb * Le_matrix, mumatrix, axis=0 ) #1D
        muintf = simps( Rf * Le_matrix, mumatrix, axis=0 )
        muints = simps( Rs * Le_matrix, mumatrix, axis=0 )
        
        muintbmatrix = muintb[matrix4] # 1D -> 2D along k axis
        muintfmatrix = muintf[matrix4]
        muintsmatrix = muints[matrix4]
        kmatrix2 = skcenter[matrix4]
        Pmmatrix = matterpower[matrix4]
        
        Vir = 4 * pi * rmatrix**2 * drmatrix + 1./3 * pi * (drmatrix)**3
        
        #AvgBesselmatrix = avgBessel(l,kmatrix2,rminmatrix,rmaxmatrix)/Vir #2D
        
        AvgBesselmatrix = np.array([ avgBessel(l,sk,rmin,rmax) for sk in skcenter ])/Vir

        resultb = Const * simps( kmatrix2**2 * Pmmatrix * muintbmatrix * AvgBesselmatrix, skcenter, axis = 0 )
        resultf = Const * simps( kmatrix2**2 * Pmmatrix * muintfmatrix * AvgBesselmatrix, skcenter, axis = 0 )
        results = Const * simps( kmatrix2**2 * Pmmatrix * muintsmatrix * AvgBesselmatrix, skcenter, axis = 0 )
    
        return resultb, resultf, results
            
            

    def derivative_Xi_All(self):

        from multiprocessing import Process, Queue
        
        def derivative_process(q, order, l):
            re = self.derivative_xi(l)
            q.put((order, re))
        
        inputs =[0.0, 2.0, 4.0]
        q = Queue()
        
        Ps = [Process(target=derivative_process, args=(q, z[0], z[1])) for z in zip(range(3), inputs)]
        for p in Ps:
            p.start()
        Dev = [q.get() for p in Ps]
        Dev.sort()
        Devlist = [De[1] for De in Dev]
        
        self.dxib0, self.dxif0, self.dxis0 = Devlist[0]
        self.dxib2, self.dxif2, self.dxis2 = Devlist[1]
        self.dxib4, self.dxif4, self.dxis4 = Devlist[2]
        
        """
        self.dxib0, self.dxif0, self.dxis0 = self.derivative_xi(0.0)
        self.dxib2, self.dxif2, self.dxis2 = self.derivative_xi(2.0)
        self.dxib4, self.dxif4, self.dxis4 = self.derivative_xi(4.0)
        """
        print 'derivative_Xi_All (dxib, dxif, dxif)'


    



    def RSDband_derivative_xi_All(self):
        #========================================================
        #
        #   Function for calculating multipole derivatives
        #   dXi/dp_i
        #
        #   Output : dXi/db, dXi/df, dXi/ds three 1d array along r axis
        #            for each mode l ( l = 0,2,4 )
        #
        #========================================================
        import cmath
        I = cmath.sqrt(-1)
        
        b = self.b
        f = self.f
        s = self.s
        rcenter = self.rcenter
        rmin = self.rmin
        rmax = self.rmax
        klist = self.klist
        kcenter = self.kcenter
        skbin = self.skbin
        mulist = self.mulist
        dlnk = self.dlnk
        Pmlist = 1. #self.Pmlist
        matterpower = self.RealPowerBand
        rmin = self.rmin
        rmax = self.rmax
        dr = self.dr
    
        matrix1, matrix2, matrix3 = np.mgrid[0:len(mulist), 0:self.subN, 0:len(rcenter) ]
        matrix4, matrix5 = np.mgrid[ 0:self.subN, 0: len(rcenter)]
        mumatrix = self.mulist[matrix1]
        rminmatrix = rmin[matrix5]
        rmaxmatrix = rmax[matrix5]
        rmatrix = rcenter[matrix5]
        drmatrix = dr[matrix5]
        
        Le_matrix0 = Ll(0.0,mumatrix)
        Le_matrix2 = Ll(2.0,mumatrix)
        Le_matrix4 = Ll(4.0,mumatrix)
        
        Const = lambda l: (2*l + 1.)/2 * np.real(I**l) / (2 * pi**2)
        Const0 = Const(0.0)
        Const2 = Const(2.0)
        Const4 = Const(4.0)
        
        resultlistb0=[]
        resultlistf0=[]
        resultlists0=[]
        resultlistb2=[]
        resultlistf2=[]
        resultlists2=[]
        resultlistb4=[]
        resultlistf4=[]
        resultlists4=[]
        for i in range(len(kcenter)):
            k = skbin[i*self.subN : i*self.subN+self.subN]
            Pm = matterpower[i*self.subN : i*self.subN+self.subN]
            kmatrix = k[matrix2]
            
            
            Dmatrix = np.exp(- kmatrix**2 * mumatrix**2 * self.s**2) #FOG matrix
            Rb = 2 *(self.b + self.f * mumatrix**2) * Dmatrix
            Rf = 2 * mumatrix**2 *(self.b + self.f * mumatrix**2) * Dmatrix
            Rs = (- kmatrix**2 * mumatrix**2)*(self.b + self.f * mumatrix**2)**2 * Dmatrix
            
            muintb0 = simps( Rb * Le_matrix0, mumatrix, axis=0 )
            muintf0 = simps( Rf * Le_matrix0, mumatrix, axis=0 )
            muints0 = simps( Rs * Le_matrix0, mumatrix, axis=0 )
            muintb2 = simps( Rb * Le_matrix2, mumatrix, axis=0 )
            muintf2 = simps( Rf * Le_matrix2, mumatrix, axis=0 )
            muints2 = simps( Rs * Le_matrix2, mumatrix, axis=0 )
            muintb4 = simps( Rb * Le_matrix4, mumatrix, axis=0 )
            muintf4 = simps( Rf * Le_matrix4, mumatrix, axis=0 )
            muints4 = simps( Rs * Le_matrix4, mumatrix, axis=0 )
            
            kmatrix2 = k[matrix4]
            Pmmatrix = Pm[matrix4]
            Vir = 4 * pi * rmatrix**2 * drmatrix + 1./3 * pi * (drmatrix)**3
            AvgBesselmatrix0 = avgBessel(0.0,kmatrix2,rminmatrix,rmaxmatrix)/Vir
            AvgBesselmatrix2 = avgBessel(2.0,kmatrix2,rminmatrix,rmaxmatrix)/Vir
            AvgBesselmatrix4 = avgBessel(4.0,kmatrix2,rminmatrix,rmaxmatrix)/Vir
            
            resultb0 = simps( kmatrix2**2 * Pmmatrix * muintb0 * AvgBesselmatrix0, k, axis = 0 )
            resultf0 = simps( kmatrix2**2 * Pmmatrix * muintf0 * AvgBesselmatrix0, k, axis = 0 )
            results0 = simps( kmatrix2**2 * Pmmatrix * muints0 * AvgBesselmatrix0, k, axis = 0 )
            resultb2 = simps( kmatrix2**2 * Pmmatrix * muintb2 * AvgBesselmatrix2, k, axis = 0 )
            resultf2 = simps( kmatrix2**2 * Pmmatrix * muintf2 * AvgBesselmatrix2, k, axis = 0 )
            results2 = simps( kmatrix2**2 * Pmmatrix * muints2 * AvgBesselmatrix2, k, axis = 0 )
            resultb4 = simps( kmatrix2**2 * Pmmatrix * muintb4 * AvgBesselmatrix4, k, axis = 0 )
            resultf4 = simps( kmatrix2**2 * Pmmatrix * muintf4 * AvgBesselmatrix4, k, axis = 0 )
            results4 = simps( kmatrix2**2 * Pmmatrix * muints4 * AvgBesselmatrix4, k, axis = 0 )
            
            resultlistb0.append(resultb0)
            resultlistf0.append(resultf0)
            resultlists0.append(results0)
            resultlistb2.append(resultb2)
            resultlistf2.append(resultf2)
            resultlists2.append(results2)
            resultlistb4.append(resultb4)
            resultlistf4.append(resultf4)
            resultlists4.append(results4)
        

        self.dxib0 = Const0 * np.sum( resultlistb0, axis=0)
        self.dxif0 = Const0 * np.sum( resultlistf0, axis=0)
        self.dxis0 = Const0 * np.sum( resultlists0, axis=0)
        self.dxib2 = Const2 * np.sum( resultlistb2, axis=0)
        self.dxif2 = Const2 * np.sum( resultlistf2, axis=0)
        self.dxis2 = Const2 * np.sum( resultlists2, axis=0)
        self.dxib4 = Const4 * np.sum( resultlistb4, axis=0)
        self.dxif4 = Const4 * np.sum( resultlistf4, axis=0)
        self.dxis4 = Const4 * np.sum( resultlists4, axis=0)
        
        print "RSDband_derivative_xi_All ( dxib, dxif, dxis )"
        return self.dxib0, self.dxif0, self.dxis0, self.dxib2, self.dxif2, self.dxis2, self.dxib4, self.dxif4, self.dxis4
    


    def covariance_original(self, l1, l2):
        #================================================================
        #
        #   < Function for calculating components of covariance matrix >
        #
        #   From klist, rlist, Pmlist, construct 2 or 3 dimensional array
        #   passing through whole power spectrum function and integrate
        #   along k axis.
        #   This function calls function 'sbess' from sbess fortran module and
        #   fuction 'eval_legendre' legen fortran module/
        #   For compiling, read 'readme.txt'
        #
        #   version2. modified Feb 13, 2015 by Sujeong Lee
        #
        #
        #   Output : submatrices C_ll' for each modes (l,l' = 0,2,4)
        #            size of each matrix is (# of r bins) x (# of r bins)
        #
        #   C_ll' = <X_l(ri)X_l'(rj)>
        #
        #
        #=================================================================

        #from scipy.special import eval_legendre
        from scipy.integrate import quad,simps
        from numpy import zeros, sqrt, pi, exp
        import cmath
        I = cmath.sqrt(-1)

        klist = self.klist
        kcenter = self.kcenter
        skbin = self.skbin
        skcenter = self.skcenter
        rlist = self.rlist
        rcenter = self.rcenter
        dr = self.dr
        rmin = self.rmin
        rmax = self.rmax
        mulist = self.mulist
        dk = self.dk
        dlnk = self.dlnk
        sdlnk = self.sdlnk
        dr = self.dr
        Pmlist = self.Pmlist
        Pm = self.RealPowerBand
        s = self.s
        b = self.b
        f = self.f
        Vs = self.Vs
        nn = self.nn

        #Total integ ftn except double bessel


        # generating 2-dim matrix for k and mu, matterpower spectrum, FoG term
        matrix1,matrix2 = np.mgrid[0:len(mulist),0:len(skcenter)]
        mulistmatrix = mulist[matrix1] # mu matrix (axis 0)
        klistmatrix = skcenter[matrix2] # k matrix (axis 1)
        Pmlistmatrix = Pm[matrix2] # matter power spectrum matrix same with k axis
        Dmatrix = exp(- klistmatrix**2 * mulistmatrix**2 * s**2) #D(klistmatrix,mulistmatrix) #FOG term k and mu matrix
        
        Le_matrix1 = Ll(l1, mulistmatrix)
        Le_matrix2 = Ll(l2, mulistmatrix)
        
        #
        #   constructing total power spectrum with matrices above
        #
        #   PP(k,mu) = (b + f*mu**2)**2 Pm * FoG
        #

        PP_matrix = (b + f * mulistmatrix * mulistmatrix)**2 * Pmlistmatrix * Dmatrix
        #
        # total integration except bessel term
        #
        # \int\int costant * k^3 (1/n + P(k,mu))^2 dlnk dmu
        #

        constant = (1./Vs) * (2*l1+1.)*(2*l2+1.)/(2*pi**2) * np.real(I**(l1+l2))
        Total_Power = constant * klistmatrix**2 *(PP_matrix**2 + 2./nn * PP_matrix)* Le_matrix1 * Le_matrix2
        # integrate along mu axis by using scipy.integrate.simpson
        integ = simps(Total_Power,mulist, axis=0) #result, 1d array for k
        # constructing 3 dimension matrix

        matrix1,matrix2,matrix3 = np.mgrid[0:len(skcenter),0:len(rcenter),0:len(rcenter)]
        rlistmatrix1 = rcenter[matrix2] # vertical
        rlistmatrix2 = rcenter[matrix3] # horizontal
        klistmatrix = skcenter[matrix1] # axis 0 #redefine to match with r dimension
        dr1 = dr[matrix2]
        dr2 = dr[matrix3]
        rminmatrix = rmin[matrix2]
        rmaxmatrix = rmax[matrix2]
        rminmatrix2 = rmin[matrix3]
        rmaxmatrix2 = rmax[matrix3]
        
        integ = integ[matrix1] # convert 1d array to 3 d array for next calculation (same axis with kmatrix)
        # Casting 3 dimensional array into double bessel function
        
        Vir1 = 4 * pi * rlistmatrix1**2 * dr1 * (1. + 1./12 * (dr1/rlistmatrix1)**2)
        Vir2 = 4 * pi * rlistmatrix2**2 * dr2 * (1. + 1./12 * (dr2/rlistmatrix2)**2)
        
        AvgBesselmatrix1 = avgBessel(l1,klistmatrix,rminmatrix,rmaxmatrix) /Vir1
        AvgBesselmatrix2 = avgBessel(l2,klistmatrix,rminmatrix2,rmaxmatrix2) /Vir2
        
        #Total = integ * doublebessel(klistmatrix,l1,l2,rlistmatrix1,rlistmatrix2) # 3 dim array
        Total = integ * AvgBesselmatrix1 * AvgBesselmatrix2
        
        # integrate along k axis
        Total_Integ = simps(Total,skcenter, axis=0)
        # output : 2 dimensional matrices for r1, r2.
        # k axis is compactified by simpson integration
        
        #Shot_noise term, diagonal 2d matrices
        if l1 == l2:
            Vi = 4 * pi * rcenter**2 * dr * (1. + 1./12 * (dr/rcenter)**2) #1d array
            Shot_noise = (2./Vs) * (2*l1+1)/nn**2 / Vi #1d array
        
            Shot_noise_term = np.zeros((len(rcenter),len(rcenter)))
            np.fill_diagonal(Shot_noise_term,Shot_noise)
        else : Shot_noise_term = 0.
        
        Result = Total_Integ + Shot_noise_term
        
        print 'covariance {:>1.0f}{:>1.0f} is finished'.format(l1,l2)
        return Result


    def covariance(self, l1, l2):

        
        #from scipy.special import eval_legendre
        from scipy.integrate import quad,simps
        from numpy import zeros, sqrt, pi, exp
        from legen import eval_legendre
        from sbess import sbess
        import cmath
        I = cmath.sqrt(-1)
        
        klist = self.klist
        kcenter = self.kcenter
        rlist = self.rlist
        rcenter = self.rcenter
        dr = self.dr
        rmin = self.rmin
        rmax = self.rmax
        mulist = self.mulist
        dk = self.dk
        dlnk = self.dlnk
        dr = self.dr
        Pmlist = self.Pmlist
        s = self.s
        b = self.b
        f = self.f
        Vs = self.Vs
        nn = self.nn
        
        #Total integ ftn except double bessel
        
        #mu-integration=================================================
        
        D = lambda k,mu : exp(- k**2 * mu**2 * s**2) # Finger of God
        
        # generating 2-dim matrix for k and mu, matterpower spectrum, FoG term
        matrix1,matrix2 = np.mgrid[0:len(mulist),0:len(kcenter)]
        mulistmatrix = mulist[matrix1] # mu matrix (axis 0)
        klistmatrix = kcenter[matrix2] # k matrix (axis 1)
        Pmlistmatrix = Pmlist[matrix2] # matter power spectrum matrix same with k axis
        Dmatrix = D(klistmatrix,mulistmatrix) #FOG term k and mu matrix
        Ll = lambda l,x : eval_legendre(l,x) #Legendre
        Ll = vectorize(Ll)
        Double_Le_matrix = Ll(l1,mulistmatrix)*Ll(l2,mulistmatrix) #L(l1,x)L(l2,x)
        
        
        #
        #   constructing total power spectrum with matrices above
        #
        #   PP(k,mu) = (b + f*mu**2)**2 Pm * FoG
        #
        
        PP_matrix = (b + f * mulistmatrix * mulistmatrix)**2 * Pmlistmatrix * Dmatrix
        
        #
        # total integration except bessel term
        #
        # \int\int costant * k^3 (1/n + P(k,mu))^2 dlnk dmu
        #
        
        constant = (2./Vs) * (2*l1+1)*(2*l2+1)/(2*np.pi)**2 * np.real(I**(l1+l2))
        Total_Power = constant * klistmatrix**2 *(PP_matrix**2 + 2./nn * PP_matrix)* Double_Le_matrix
        # integrate along mu axis by using scipy.integrate.simpson
        integ = simps(Total_Power,mulistmatrix, axis=0) #result, 1d array for k
        
        
        # Double Spherical Bessel ftn including r1 r2 ========================
        #
        #   jl(kr)jl'(kr)
        #
        def doublebessel(k,l1,l2,r1,r2):
            re = sbess(k*r1,l1)*sbess(k*r2,l2)
            return re
        doublebessel = vectorize(doublebessel) #vectorize
        
        
        
        # constructing 3 dimension matrix
        
        matrix1,matrix2,matrix3 = np.mgrid[0:len(kcenter),0:len(rcenter),0:len(rcenter)]
        rlistmatrix1 = rcenter[matrix2] # vertical
        rlistmatrix2 = rcenter[matrix3] # horizontal
        dlnk = np.log(klist)
        klistmatrix = kcenter[matrix1] # axis 0 #redefine to match with r dimension
        dlnkmatrix = dlnk[matrix1]
        integ = integ[matrix1] # convert 1d array to 3 d array for next calculation (same axis with kmatrix)
        # Casting 3 dimensional array into double bessel function
        
        Total = integ * doublebessel(klistmatrix,l1,l2,rlistmatrix1,rlistmatrix2) # 3 dim array
        
        # integrate along k axis
        Total_Integ = simps(Total,klistmatrix, axis=0)
        # output : 2 dimensional matrices for r1, r2.
        # k axis is compactified by simpson integration
        
        #Shot_noise term, diagonal 2d matrices
        if l1 == l2:
            Vi = 4 * pi * rcenter**2 * dr * (1. + 1./12 * (dr/rcenter)**2) #1d array
            Shot_noise = (2./Vs) * (2*l1+1)/nn**2 / Vi #1d array
            
            Shot_noise_term = np.zeros((len(rcenter),len(rcenter)))
            np.fill_diagonal(Shot_noise_term,Shot_noise)
        else : Shot_noise_term = 0.
        
        Result = Total_Integ + Shot_noise_term
        
        print 'covariance {:>1.0f}{:>1.0f} is finished'.format(l1,l2)
        return Result

        



    def RSD_shell_covariance(self, l1, l2):
        #================================================================
        #
        #   < Function for calculating components of covariance matrix >
        #
        #   From klist, rlist, Pmlist, construct 2 or 3 dimensional array
        #   passing through whole power spectrum function and integrate
        #   along k axis.
        #   This function calls function 'sbess' from sbess fortran module and
        #   fuction 'eval_legendre' legen fortran module/
        #   For compiling, read 'readme.txt'
        #
        #   version2. modified Feb 13, 2015 by Sujeong Lee
        #
        #
        #   Output : submatrices C_ll' for each modes (l,l' = 0,2,4)
        #            size of each matrix is (# of r bins) x (# of r bins)
        #
        #   C_ll' = <X_l(ri)X_l'(rj)>
        #
        #
        #=================================================================
    
        #from scipy.special import eval_legendre
        from scipy.integrate import quad,simps
        import numpy as np
        from numpy import zeros, sqrt, pi, exp
        import cmath
        
        I = cmath.sqrt(-1)
    
        klist = self.klist
        kcenter = self.kcenter
        skcenter = self.skcenter
        rlist = self.rlist
        rcenter = self.rcenter
        dr = self.dr
        rmin = self.rmin
        rmax = self.rmax
        mulist = self.mulist
        dk = self.dk
        dlnk = self.dlnk
        dr = self.dr
        Pmlist = self.Pmlist
        Pm = self.RealPowerBand
        s = self.s
        b = self.b
        f = self.f
        Vs = self.Vs
        nn = self.nn

        
        # generating 2-dim matrix for k and mu, matterpower spectrum, FoG term
        matrix1,matrix2 = np.mgrid[0:len(mulist),0:len(skcenter)]
        mulistmatrix = mulist[matrix1] # mu matrix (axis 0)
        klistmatrix = skcenter[matrix2] # k matrix (axis 1)
        
        Le_matrix1 = Ll(l1,mulistmatrix)
        Le_matrix2 = Ll(l2,mulistmatrix)
        Dmatrix = exp(- klistmatrix**2 * mulistmatrix**2 * self.s**2)

        R = (b + f * mulistmatrix**2)**2 * Dmatrix
        const_gamma = np.real(I**(l1+l2)) * 2.* (2*l1+1)*(2*l2+1) /(2*np.pi)**2 /Vs
        
        
        Rintegral3 = simps(R**2 * Le_matrix1 * Le_matrix2, mulist, axis=0 )
        Rintegral2 = simps(R * Le_matrix1 * Le_matrix2, mulist, axis=0 )
        
        result = const_gamma * skcenter**2 * (Rintegral3 * Pm**2 + Rintegral2 * Pm * 2./nn) # 1D array
        

        
        # constructing 3 dimension matrix

        
        matrix1,matrix2,matrix3 = np.mgrid[0:len(skcenter),0:len(rcenter),0:len(rcenter)]
        matrix4,matrix5 = np.mgrid[0:len(rcenter),0:len(rcenter)]
        rlistmatrix1 = rcenter[matrix4] # vertical
        rlistmatrix2 = rcenter[matrix5] # horizontal
        dr1 = dr[matrix4] # vertical
        dr2 = dr[matrix5] # horizontal
        rminmatrix = rmin[matrix2] # vertical
        rminmatrix2 = rmin[matrix3] # horizontal
        rmaxmatrix = rmax[matrix2] # vertical
        rmaxmatrix2 = rmax[matrix3] # horizontal
        klistmatrix = skcenter[matrix1] # axis 0 #redefine to match with r dimension
        #dlnk = np.log(kcenter)
        #dlnkmatrix = dlnk[matrix1]
        
        result = result[matrix1] # convert 1d array to 3 d array for next calculation (same axis with kmatrix)
        Vir1 = 4 * pi * rlistmatrix1**2 * dr1 * (1. + 1./12 * (dr1/rlistmatrix1)**2)
        Vir2 = 4 * pi * rlistmatrix2**2 * dr2 * (1. + 1./12 * (dr2/rlistmatrix2)**2)

        avgBesselmatrix1 = avgBessel(l1,klistmatrix,rminmatrix,rmaxmatrix)
        avgBesselmatrix2 = avgBessel(l2,klistmatrix,rminmatrix2,rmaxmatrix2)
        
        
        FirstSecond = simps(result * avgBesselmatrix1 * avgBesselmatrix2,skcenter, axis=0)/Vir1/Vir2

        
        # Last Term
        
        if l1 == l2:
            Vi = 4 * pi * rcenter**2 * dr * (1. + 1./12 * (dr/rcenter)**2) #1d array
            Last = (2./Vs) * (2*l1+1)/nn**2 / Vi #1d array
            LastTerm = np.zeros((len(rcenter),len(rcenter)))
            np.fill_diagonal(LastTerm,Last)
        else : LastTerm = 0.
    
        Total = FirstSecond + LastTerm
        
    
        print 'RSD_shell_covariance {:>1.0f}{:>1.0f} is finished'.format(l1,l2)
        return Total


    def RSD_covariance_Allmodes(self):
    
        """ Output : submatrices C_ll' for each modes (l,l' = 0,2,4)
                size of each matrix is (# of r bins) x (# of r bins)
    
           C_ll' = <X_l(ri)X_l'(rj)> """

        from scipy.integrate import simps
        from numpy import zeros, sqrt, pi, exp
        import cmath
        I = cmath.sqrt(-1)
    
        klist = self.klist
        kcenter = self.kcenter
        skcenter = self.skcenter
        rlist = self.rlist
        rcenter = self.rcenter
        dr = self.dr
        rmin = self.rmin
        rmax = self.rmax
        mulist = self.mulist
        dk = self.dk
        dlnk = self.dlnk
        dr = self.dr
        Pmlist = self.Pmlist
        Pm = self.RealPowerBand
        s = self.s
        b = self.b
        f = self.f
        Vs = self.Vs
        nn = self.nn
        
    
        # generating 2-dim matrix for k and mu, matterpower spectrum, FoG term
        matrix1,matrix2 = np.mgrid[0:len(mulist),0:len(skcenter)]
        mulistmatrix = mulist[matrix1] # mu matrix (axis 0)
        klistmatrix = skcenter[matrix2] # k matrix (axis 1)
        Le_matrix0 = Ll(0.0,mulistmatrix)
        Le_matrix2 = Ll(2.0,mulistmatrix)
        Le_matrix4 = Ll(4.0,mulistmatrix)
    
        Dmatrix = np.exp(- klistmatrix**2 * mulistmatrix**2 * self.s**2)
        R = (b + f * mulistmatrix**2)**2 * Dmatrix

        from multiprocessing import Process, Queue
        
        print 'Rintegral'
        def Rintegral(q, order, (l1, l2, Le1, Le2)):
            
            #import covariance_class2
            from numpy import pi, real
            from scipy.integrate import simps
            import cmath
            
            I = cmath.sqrt(-1)
            const_gamma = real(I**(l1+l2)) * 2.* (2*l1+1)*(2*l2+1) /(2*pi)**2 /self.Vs
            Rintegral3 = simps(R**2 * Le1 * Le2, self.mulist, axis=0 )
            Rintegral2 = simps(R * Le1 * Le2, self.mulist, axis=0 )
            result = const_gamma * skcenter**2 * (Rintegral3 * Pm**2 + Rintegral2 * Pm * 2./self.nn)
            
            q.put((order,result))
        
        inputs = (( 0.0, 0.0, Le_matrix0, Le_matrix0),( 0.0, 2.0, Le_matrix0, Le_matrix2),(0.0, 4.0,Le_matrix0, Le_matrix4),(2.0, 2.0, Le_matrix2, Le_matrix2),(2.0, 4.0, Le_matrix2, Le_matrix4),(4.0, 4.0, Le_matrix4, Le_matrix4))
        
        R_queue = Queue()
        R_processes = [Process(target=Rintegral, args=(R_queue, z[0], z[1])) for z in zip(range(6), inputs)]
        for p in R_processes:
            p.start()
        #for p in R_processes:
            #p.join()
        Rintegrals = [R_queue.get() for p in R_processes]
            #for p in R_processes:
            #p.terminate()

        Rintegrals.sort()
        Rintegrallist = [R[1] for R in Rintegrals]
        
        Rintegral00 = Rintegrallist[0] # 1D
        Rintegral02 = Rintegrallist[1]
        Rintegral04 = Rintegrallist[2]
        Rintegral22 = Rintegrallist[3]
        Rintegral24 = Rintegrallist[4]
        Rintegral44 = Rintegrallist[5]
    
        matrix4,matrix5 = np.mgrid[0:len(rcenter),0:len(rcenter)]
        rlistmatrix1 = rcenter[matrix4] # vertical
        rlistmatrix2 = rcenter[matrix5] # horizontal
        dr1 = dr[matrix4] # vertical
        dr2 = dr[matrix5] # horizontal
        rminmatrix = rmin[matrix4] # vertical
        rminmatrix2 = rmin[matrix5] # horizontal
        rmaxmatrix = rmax[matrix4] # vertical
        rmaxmatrix2 = rmax[matrix5] # horizontal
    
        Vir1 = 4 * pi * rlistmatrix1**2 * dr1 * (1. + 1./12 * (dr1/rlistmatrix1)**2)
        Vir2 = 4 * pi * rlistmatrix2**2 * dr2 * (1. + 1./12 * (dr2/rlistmatrix2)**2)
        Vi = 4 * pi * rcenter**2 * dr * (1. + 1./12 * (dr/rcenter)**2) #1d array

        print 'AvgBessel'
        
        def AvgBessel_q(q, order, (l, skcenter, rmin, rmax)):
            Avg = [avgBessel(l,k,rmin,rmax) for k in skcenter] #2D (kxr)
            q.put((order,Avg))
    
        
        inputs_bessel = [(0.0,skcenter,rmin,rmax),(2.0,skcenter,rmin,rmax), (4.0,skcenter,rmin,rmax) ]

        B_queue = Queue()
        B_processes = [Process(target=AvgBessel_q, args=(B_queue,z[0], z[1])) for z in zip(range(3), inputs_bessel)]
        for pB in B_processes:
            pB.start()
        Bessels = [B_queue.get() for pB in B_processes]
        Bessels.sort()
        Bessel_list = [ B[1] for B in Bessels] #2D bessel, (kxr)
        print 'bessel_process'

        """
        avgBesselmatrix03D = np.array([ Bessel_list[0] for r in rcenter]).swapaxes(0,1)
        avgBesselmatrix23D = np.array([ Bessel_list[1] for r in rcenter]).swapaxes(0,1) #3D (kxrxr)
        avgBesselmatrix43D = np.array([ Bessel_list[2] for r in rcenter]).swapaxes(0,1)
        """

        avgBesselmatrix0 = np.array(Bessel_list[0]) #2D, (kxr)
        avgBesselmatrix2 = np.array(Bessel_list[1])
        avgBesselmatrix4 = np.array(Bessel_list[2])

        print 'Add'
        matrix1, matrix2 = np.mgrid[0:len(skcenter), 0:len(rcenter)]
        Volume_double = Vir1 * Vir2


        def FirstSecond(queue, order, (l1, l2, result, avgBessel1, avgBessel2)):
        
            Rint_result = result[matrix1] # 2D
            
            relist = []
            for i in range(len(rcenter)/2):
                avgBmatrix = np.array(avgBessel1[:, i])[matrix1]
                re = simps(Rint_result * avgBmatrix * avgBessel2, skcenter, axis=0)
                relist.append(re)
            FirstTerm = relist/ Volume_double[0:len(rcenter)/2,:] #2D
            if l1 == l2:
                Last = (2./Vs) * (2*l1+1)/nn**2 / Vi #1d array
                LastTermmatrix = np.zeros((len(rcenter),len(rcenter)))
                np.fill_diagonal(LastTermmatrix,Last)
                LastTerm = LastTermmatrix[0:len(rcenter)/2,:]
            else : LastTerm = 0.
            
        
            re = FirstTerm+LastTerm
            queue.put((order,re))


        def FirstSecond2(queue, order, (l1, l2, result, avgBessel1, avgBessel2)):
    
            Rint_result = result[matrix1] # 2D
    
            relist = []
            for i in range(len(rcenter)/2, len(rcenter)):
                avgBmatrix = np.array(avgBessel1[:, i])[matrix1]
                re = simps(Rint_result * avgBmatrix * avgBessel2, skcenter, axis=0)
                relist.append(re)
            FirstTerm = relist/ Volume_double[len(rcenter)/2:len(rcenter),:] #2D
            if l1 == l2:
                Last = (2./Vs) * (2*l1+1)/nn**2 / Vi #1d array
                LastTermmatrix = np.zeros((len(rcenter),len(rcenter)))
                np.fill_diagonal(LastTermmatrix,Last)
                LastTerm = LastTermmatrix[len(rcenter)/2:len(rcenter),:]
                
            else : LastTerm = 0.
            
            re = FirstTerm+LastTerm
            queue.put((order,re))

        """
        
        matrix1, matrix2 = np.mgrid[0:len(skcenter), 0:len(rcenter)]
        Volume_double = Vir1 * Vir2
        def FirstSecond(queue, order, (l1, l2, result, avgBessel1, avgBessel2)):
        
            Rint_result = result[matrix1] # 2D

            relist = [ simps(Rint_result * avgBessel1[:, :, i] * avgBessel2[:, i, :], skcenter, axis=0) for i in range(len(rcenter)/2) ]
        
            FirstTerm = relist/ Volume_double[0:len(rcenter)/2,:] #2D
            if l1 == l2:
                Last = (2./Vs) * (2*l1+1)/nn**2 / Vi #1d array
                LastTermmatrix = np.zeros((len(rcenter),len(rcenter)))
                np.fill_diagonal(LastTermmatrix,Last)
                LastTerm = LastTermmatrix[0:len(rcenter)/2,:]
            else : LastTerm = 0.
            
            re = FirstTerm+LastTerm
            queue.put((order,re))
        
        
        def FirstSecond2(queue, order, (l1, l2, result, avgBessel1, avgBessel2)):
            
            Rint_result = result[matrix1] # 2D
            relist = [ simps(Rint_result * avgBessel1[:, :, i] * avgBessel2[:, i, :], skcenter, axis=0) for i in range(len(rcenter)/2, len(rcenter)) ]
            
            FirstTerm = relist/ Volume_double[len(rcenter)/2:len(rcenter),:] #2D
            if l1 == l2:
                Last = (2./Vs) * (2*l1+1)/nn**2 / Vi #1d array
                LastTermmatrix = np.zeros((len(rcenter),len(rcenter)))
                np.fill_diagonal(LastTermmatrix,Last)
                LastTerm = LastTermmatrix[len(rcenter)/2:len(rcenter),:]
            else : LastTerm = 0.
            
            re = FirstTerm+LastTerm
            queue.put((order,re))
            
        """
        
        F_inputs = (( 0.0, 0.0, Rintegral00, avgBesselmatrix0, avgBesselmatrix0),( 0.0, 2.0, Rintegral02,  avgBesselmatrix0, avgBesselmatrix2),(0.0, 4.0, Rintegral04, avgBesselmatrix0, avgBesselmatrix4 ),(2.0, 2.0, Rintegral22, avgBesselmatrix2, avgBesselmatrix2 ),(2.0, 4.0, Rintegral24, avgBesselmatrix2, avgBesselmatrix4 ),(4.0, 4.0, Rintegral44, avgBesselmatrix4, avgBesselmatrix4))
        
        F_queue = Queue()
        F_processes1 = [Process(target=FirstSecond, args=(F_queue, z[0], z[1])) for z in zip(range(6),F_inputs)]
        F_processes2 = [Process(target=FirstSecond2, args=(F_queue, z[0], z[1])) for z in zip(range(6,12),F_inputs)]
        F_processes = F_processes1 + F_processes2
        for pF in F_processes:
            pF.start()
        
        Ts = [F_queue.get() for pF in F_processes]
        Ts.sort()
        Total = [T[1] for T in Ts]

        self.covariance00 = np.vstack((Total[0], Total[6]))
        self.covariance02 = np.vstack((Total[1], Total[7]))
        self.covariance04 = np.vstack((Total[2], Total[8]))
        self.covariance22 = np.vstack((Total[3], Total[9]))
        self.covariance24 = np.vstack((Total[4], Total[10]))
        self.covariance44 = np.vstack((Total[5], Total[11]))

        print 'RSD_shell_covariance_AllModes is finished'
        return self.covariance00, self.covariance02, self.covariance04, self.covariance22, self.covariance24, self.covariance44





    def RSDband_covariance(self, l1, l2):
        #================================================================
        #
        #   < Function for calculating components of covariance matrix >
        #
        #   From klist, rlist, Pmlist, construct 2 or 3 dimensional array
        #   passing through whole power spectrum function and integrate
        #   along k axis.
        #   This function calls function 'sbess' from sbess fortran module and
        #   fuction 'eval_legendre' legen fortran module/
        #   For compiling, read 'readme.txt'
        #
        #   version2. modified Feb 13, 2015 by Sujeong Lee
        #
        #
        #   Output : submatrices C_ll' for each modes (l,l' = 0,2,4)
        #            size of each matrix is (# of r bins) x (# of r bins)
        #
        #   C_ll' = <X_l(ri)X_l'(rj)>
        #
        #
        #=================================================================
    
        #from scipy.special import eval_legendre
        from scipy.integrate import quad,simps
        from numpy import zeros, sqrt, pi, exp
        import cmath
        I = cmath.sqrt(-1)
    
        klist = self.klist
        kcenter = self.kcenter
        rlist = self.rlist
        rcenter = self.rcenter
        dr = self.dr
        rmin = self.rmin
        rmax = self.rmax
        mulist = self.mulist
        dk = self.dk
        dlnk = self.dlnk
        dr = self.dr
        Pmlist = self.Pmlist
        Pm = self.RealPowerBand
        s = self.s
        b = self.b
        f = self.f
        Vs = self.Vs
        nn = self.nn
        skbin = self.skbin
        skcenter = self.skcenter
    
        matrix1, matrix2, matrix3, matrix3_1 = np.mgrid[0:len(mulist),0:self.subN, 0: len(rcenter), 0:len(rcenter)]
        matrix4, matrix5, matrix6 = np.mgrid[ 0:self.subN, 0: len(rcenter),  0: len(rcenter)]
        mumatrix = mulist[matrix1]
        rminmatrix1 = rmin[matrix5]
        rmaxmatrix1 = rmax[matrix5]
        rmatrix1 = rcenter[matrix5]
        drmatrix1 = dr[matrix5]
        rminmatrix2 = rmin[matrix6]
        rmaxmatrix2 = rmax[matrix6]
        rmatrix2 = rcenter[matrix6]
        drmatrix2 = dr[matrix6]
    
        Le_matrix1 = Ll(l1,mumatrix)
        Le_matrix2 = Ll(l2,mumatrix)
    
        Const_alpha = np.real(I**(l1 + l2)) * (2*l1 + 1.) * (2*l2 + 1.) * 2 /(2 * pi)**2 /Vs
    
        resultlist1 = []
        resultlist2 = []
        for i in range(len(kcenter)):
            k = skcenter[i*self.subN : i*self.subN+self.subN]
            kmatrix = k[matrix2] # 3D
            Dmatrix = np.exp(- kmatrix**2 * mumatrix**2 * self.s**2) #FOG matrix
        
            R = (self.b + self.f * mumatrix**2)**2 * Dmatrix
            Rintegral3 =  simps( R**2 * Le_matrix1 * Le_matrix2, mumatrix, axis= 0 )
            Rintegral2 =  simps( R * Le_matrix1 * Le_matrix2, mumatrix, axis= 0 )
        
            kmatrix2 = k[matrix4] # 2D
            Pmmatrix = Pm[matrix4]
            Vir1 = 4 * pi * rmatrix1**2 * drmatrix1 + 1./3 * pi * (drmatrix1)**3
            Vir2 = 4 * pi * rmatrix2**2 * drmatrix2 + 1./3 * pi * (drmatrix2)**3
            AvgBesselmatrix1 = avgBessel(l2,kmatrix2,rminmatrix1,rmaxmatrix1)/Vir1
            AvgBesselmatrix2 = avgBessel(l2,kmatrix2,rminmatrix2,rmaxmatrix2)/Vir2
            result1 = simps( kmatrix2**2 * Pmmatrix**2 * Rintegral3 * AvgBesselmatrix1 * AvgBesselmatrix2, k, axis=0 )
            result2 = simps( kmatrix2**2 * Pmmatrix * Rintegral2 * AvgBesselmatrix1 * AvgBesselmatrix2, k, axis=0 )
        
            resultlist1.append(result1)
            resultlist2.append(result2)
        
        
        #matrix1, matrix2, matrix3 = np.mgrid[0:len(kcenter), 0: len(rcenter), 0: len(rcenter) ]
        Pmlistmatrix = 1. #Pmlist[matrix1]

        FirstTerm = Const_alpha * np.sum( resultlist1, axis=0 )
        SecondTerm =  2./nn * Const_alpha * np.sum( resultlist2, axis=0 )
        
        Vi = 4 * pi * rcenter**2 * dr + 1./3 * pi * (dr)**3
        LastTerm = np.zeros((len(self.rcenter), len(self.rcenter)))
        if l1 == l2 :
            la = np.real(I**(l1 * 2)) * (2 * l1 + 1.) * 2. /Vs /nn**2 /Vi
            np.fill_diagonal(LastTerm, la)
        else : pass
        
        print 'RSD_covariance_Xi {:>1.0f}{:>1.0f} is finished'.format(l1,l2)
        return FirstTerm + SecondTerm + LastTerm







class Window( RSD_covariance ):
    'sub class'
    
    def __init__(self, KMIN, KMAX, RMIN, RMAX, n, n2, subN):
        RSD_covariance.__init__(self, KMIN, KMAX, RMIN, RMAX, n, n2, subN)



    def window_tophat_k( self, k, kj ):
        """ 3D spherical top hat weight function
            ki : center of k grid
            R : radius """
        
        from numpy import pi, fabs, cos, sin
        R = self.RMAX
        V = 4./3 * pi * R**3
        kkj = fabs(k-kj)
        
        weightj = 4 * pi/np.sqrt(V)/kkj**3 * ( -kkj * R * cos(kkj*R) + sin(kkj*R) )
        
        result = weightj
        return result

    def window_tophat_r( self, r, ki):
        """ 3D spherical top hat weight in configuration space
            Fourier modes """
        
        from numpy import pi, e
        
        R = self.RMAX
        V = 4./3 * pi * R**3
        
        weighti = 4 * pi * ( -ki*R*cos(ki*R) + sin(ki * R)/ki**3/V )
        
        return weighti

    def unwindowed_P(self):
        
        kcenter, dlnk, Vs, nn = self.kcenter, self.dlnk, self.Vs, self.nn
        dk = self.dk
        Shell_avg_bandpower = self.Shell_avg_band()
        
        PowerCovariance = np.zeros((len(kcenter),len(kcenter)))
        PowerCov = (Shell_avg_bandpower + 1./nn)
        
        for i in range(len(kcenter)):
            PowerCovariance[i,i] = PowerCov[i]
        
        return PowerCovariance

    def unwindowed_dCdp(self):
        
        kcenter, dlnk, Vs, nn = self.kcenter, self.dlnk, self.Vs, self.nn
        dk = self.dk
        Shell_avg_bandpower = self.Shell_avg_band()
        
        V = 4./3 * np.pi * self.RMAX**3
        
        Vi = 4 * pi * kcenter**2 * dk + 1./3 * pi * (dk)**3
        #Mode = (2 * pi)**3 / Vi
        PowerCov = np.ones(len(kcenter))
        
        list = []
        for i in range(len(kcenter)):
            dCdp = np.zeros((len(kcenter),len(kcenter)))
            dCdp[i,i] = PowerCov[i]
            list.append(dCdp)
        
        dCdp_list = np.array(list)
        
        return dCdp_list


    def windowed_P(self):
        
        """ C_ij = S_ij + N_ij """
        
        klist = self.klist
        kcenter = self.kcenter
        skcenter = self.skcenter
        skbin = self.skbin
        dk = self.dk
        mulist = self.mulist
        dlnk = self.dlnk
        matterpower = self.RealPowerBand
        
        matrix1, matrix2 = np.mgrid[0:len(skcenter),0:len(skcenter)]
        matrix3, matrix4 = np.mgrid[0:len(skcenter), 0:self.subN]
        matrix5, matrix6 = np.mgrid[0:len(skcenter), 0:len(kcenter)]
        matrix7, matrix8 = np.mgrid[0:len(kcenter), 0:len(kcenter)]
        
        kmatrix = skcenter[matrix1]
        kjmatrix = skcenter[matrix2]
        
        R = self.RMAX  #subtract rmin ? check """
        V = 4./3 * pi * R**3
        diagonal = np.sqrt(V)
        
        
        # window function shell averaging
        
        Va = 4 * pi * kcenter**2 * dk * (1. + 1./12 * (dk/kcenter)**2)
        
        weightj = self.window_tophat_k(kmatrix, kjmatrix)
        np.fill_diagonal(weightj, diagonal)
        
        weightj_list=[]
        for i in range(len(kcenter)):
            sk = skcenter[i*self.subN:i*self.subN+self.subN]
            skmatrix = sk[matrix4]
            weightj_cut = weightj[:,i*self.subN:i*self.subN+self.subN]
            int = simps(4 * np.pi * skmatrix**2 * weightj_cut ,skmatrix, axis=1 )
            weightj_list.append(int)
        
        shell_avg_weightj = np.transpose(np.array(weightj_list)) /Va[matrix6]
        
        
        kmatrix2 = skcenter[matrix5]
        Pmatrix = matterpower[matrix5]
        normali_factor = simps(kmatrix2**2/(2*pi**2) * shell_avg_weightj**2, kmatrix2, axis=0 )
        #normali_factor = simps(4 * pi * kmatrix2**2 * shell_avg_weightj**2, kmatrix2, axis=0 )
        signal_list = []
        noise_list=[]
        for i in range(len(kcenter)):
            
            #weightj = self.window_tophat_k(kmatrix, kjmatrix)
            #np.fill_diagonal( weightj, diagonal )
            window_matrix = shell_avg_weightj * shell_avg_weightj[:, i][matrix5]
            #normali_factor_j = normali_factor_i[i] * np.ones(len(kcenter)) #simps(kmatrix2**2/(2*pi**2) * shell_avg_weightj[:, i][matrix5]**2, kmatrix2, axis=0 )
            signal = simps( kmatrix2**2/(2*pi**2) * Pmatrix * window_matrix, kmatrix2, axis=0 )
            noise = 1./self.nn * simps( kmatrix2**2/(2*pi**2) * window_matrix, kmatrix2, axis=0 )
            signal_list.append(signal)
            noise_list.append(noise)
        
        normali_factor_i = normali_factor[matrix7]
        normali_factor_j = normali_factor[matrix8]
        normali_factor_matrix = np.sqrt(normali_factor_i * normali_factor_j)
        
        def window_plotting():
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            fig = plt.figure()
            for i in np.arange(30, self.n, 2):
                
                test_weightj = self.window_tophat_k(skcenter[matrix5], kcenter[matrix6])
                window_i = shell_avg_weightj[:, i]**2
                normalized_window_i = window_i/normali_factor_matrix[i,i]
                plt.plot( self.skcenter, normalized_window_i, label='i = {}'.format(i))
            plt.legend(loc=1)
            plt.xlim(0.01, 0.5)
            plt.title('Top hat window function W_ii(k) in Fourier space ( Rmax = 2000 Mpc/h )\n i : index of k grid')
            #plt.show()
            pdfname = 'normalized.pdf'
            print ' pdf file saved : ', pdfname
            pdf=PdfPages( pdfname )
            pdf.savefig(fig)
            pdf.close()
        window_plotting()
        
        
        self.Signal_P = np.array(signal_list) /normali_factor_matrix
        self.Noise_P = np.array(noise_list) /normali_factor_matrix
        
        self.windowed_Power = self.Signal_P + self.Noise_P
        
        return self.windowed_Power


    def windowed_Xi(self):
        
        """ C_ij = S_ij + N_ij """
        
        klist = self.klist
        kcenter = self.kcenter
        skcenter = self.skcenter
        skbin = self.skbin
        dk = self.dk
        mulist = self.mulist
        dlnk = self.dlnk
        rcenter = self.rcenter
        dr = self.dr
        rmin = self.rmin
        rmax = self.rmax
        matterpower = self.RealPowerBand
        
        matrix1, matrix2 = np.mgrid[0:len(skcenter),0:len(rcenter)]
        matrix3, matrix4 = np.mgrid[0:len(rcenter),0:len(rcenter)]
        kmatrix = skcenter[matrix1]
        Pmatrix = matterpower[matrix1]
        rmatrix = rcenter[matrix2]
        rminmatrix = rmin[matrix2]
        rmaxmatrix = rmax[matrix2]
        
        R = self.RMAX  #subtract rmin ? check
        V = 4./3 * pi * R**3
        
        Vir = 4 * pi * rcenter**2 * dr * (1. + 1./12 * (dr/rcenter)**2)
        AvgBesselmatrix1 = avgBessel(0.0,kmatrix,rminmatrix,rmaxmatrix)/Vir[matrix2]
        
        self.Signal_Xi = np.array([ simps(kmatrix**2/(2 * pi**2) * Pmatrix * AvgBesselmatrix1 * AvgBesselmatrix1[:, i][matrix1], kmatrix, axis=0 )/V for i in range(len(rcenter)) ])
        self.Noise_Xi = 1./(self.nn * V) / Vir * np.identity(len(rcenter))# diagonal
        
        
        
        
        
        self.windowed_Xiance = self.Signal_Xi + self.Noise_Xi
        
        return self.windowed_Xiance



    def dPdp(self):
        
        from numpy.linalg import pinv
        
        kcenter = self.kcenter
        skcenter = self.skcenter
        dk = self.dk
        
        matrix1, matrix2 = np.mgrid[0:len(skcenter),0:len(skcenter)]
        matrix3, matrix4 = np.mgrid[0:len(skcenter), 0:self.subN]
        matrix5, matrix6 = np.mgrid[0:len(skcenter), 0:len(kcenter)]
        matrix7, matrix8 = np.mgrid[0:self.subN,0:len(skcenter)]
        matrix9, matrix10 = np.mgrid[0:len(kcenter), 0:len(kcenter)]
        matrix11, matrix12 = np.mgrid[0:len(kcenter), 0:self.subN]
        
        kmatrix = skcenter[matrix1]
        kjmatrix = skcenter[matrix2]
        kkmatrix1 = kcenter[matrix9]
        kkmatrix2 = kcenter[matrix10]
        
        R = self.RMAX  #subtract rmin ? check """
        V = 4./3 * pi * R**3
        diagonal = np.sqrt(V)
        
        # window function shell averaging
        weightj = self.window_tophat_k(kmatrix, kjmatrix)
        weightj_smallbin = self.window_tophat_k(kkmatrix1, kkmatrix2)
        np.fill_diagonal(weightj, diagonal)
        np.fill_diagonal(weightj_smallbin, diagonal)
        Va = 4 * pi * kcenter**2 * dk * (1. + 1./12 * (dk/kcenter)**2)
        
        normali_factor = simps( kkmatrix1**2/(2*pi**2) * weightj_smallbin**2, kkmatrix1, axis=0)
        
        
        integral = []
        for i in range(len(kcenter)):
            
            sk = skcenter[i*self.subN : i*self.subN+self.subN]
            skmatrix = sk[matrix7]
            weight_cut = weightj[i*self.subN:i*self.subN+self.subN,:]
            int = simps( skmatrix**2/(2*pi**2) * weight_cut**2, skmatrix, axis=0 )
            integral.append(int)
        integral = np.array(integral)
        
        
        dCdp_list=[]
        for i in range(len(kcenter)):
            sk = skcenter[i*self.subN:i*self.subN+self.subN]
            skmatrix = sk[matrix12]
            integral_cut = integral[:, i*self.subN:i*self.subN+self.subN]
            int = simps(4 * np.pi * skmatrix**2 * integral_cut, skmatrix, axis=1)
            dCdp_list.append(int)
        
        dCdp_list = np.transpose(np.array(dCdp_list))
        dPdp = (dCdp_list.diagonal()/normali_factor/Va) * np.identity(len(kcenter))
        
        
    
    
        """
        
        weightj_list=[]
        for i in range(len(kcenter)):
        sk = skcenter[i*self.subN:i*self.subN+self.subN]
        skmatrix = sk[matrix4]
        weightj_cut = weightj[:,i*self.subN:i*self.subN+self.subN]
        int = simps(4 * pi * skmatrix**2 * weightj_cut**2 ,skmatrix, axis=1 )
        weightj_list.append(int)
        
        double_shell_avg_weightj = np.transpose(np.array(weightj_list))/Va[matrix6]
        
        
        kmatrix3 = skcenter[matrix5]
        
        normali_factor = simps(kmatrix3**2/(2*pi**2) * double_shell_avg_weightj, kmatrix3, axis=0 )
        normali_factor_i = normali_factor[matrix9]
        normali_factor_j = normali_factor[matrix10]
        normali_factor_matrix = np.sqrt(normali_factor_i * normali_factor_j)
        
        
        
        dCdp_list = []
        for i in range(len(kcenter)):
        
        sk = skcenter[i*self.subN : i*self.subN+self.subN]
        skmatrix = sk[matrix7]
        window_list = double_shell_avg_weightj[i*self.subN:i*self.subN+self.subN,i]
        int = simps( sk**2/(2*pi**2) * window_list, sk, axis=0 )
        dCdp_list.append(int)
        dPdp = np.array(dCdp_list) * np.identity(len(kcenter)) / normali_factor_matrix
        """
        return dPdp




    def dCdp(self):
        """ C,a = dC_ij / dp_a """
        """ weight : top hat """
        
        from numpy.linalg import pinv
        
        kcenter = self.kcenter
        skcenter = self.skcenter
        dk = self.dk
        
        matrix1, matrix2 = np.mgrid[0:len(skcenter),0:len(skcenter)]
        matrix3, matrix4 = np.mgrid[0:len(skcenter), 0:self.subN]
        matrix5, matrix6 = np.mgrid[0:len(skcenter), 0:len(kcenter)]
        matrix7, matrix8 = np.mgrid[0:self.subN,0:len(kcenter)]
        matrix9, matrix10 = np.mgrid[0:len(kcenter), 0:len(kcenter)]
        
        kmatrix = skcenter[matrix1]
        kjmatrix = skcenter[matrix2]
        
        R = self.RMAX  #subtract rmin ? check """
        V = 4./3 * pi * R**3
        diagonal = np.sqrt(V)
        
        
        
        # window function shell averaging
        weightj = self.window_tophat_k(kmatrix, kjmatrix)
        np.fill_diagonal(weightj, diagonal)
        Va = 4 * pi * kcenter**2 * dk * (1. + 1./12 * (dk/kcenter)**2)
        
        weightj_list=[]
        for i in range(len(kcenter)):
            sk = skcenter[i*self.subN:i*self.subN+self.subN]
            skmatrix = sk[matrix4]
            weightj_cut = weightj[:,i*self.subN:i*self.subN+self.subN]
            int = simps(4 * np.pi * skmatrix**2 /(2*pi**2) * weightj_cut ,skmatrix, axis=1 )
            weightj_list.append(int)
        
        shell_avg_weightj = np.transpose(np.array(weightj_list))/Va[matrix6]
        
        
        kmatrix3 = skcenter[matrix5]
        
        normali_factor = simps(kmatrix3**2/(2*pi**2) * shell_avg_weightj**2, kmatrix3, axis=0 )
        normali_factor_i = normali_factor[matrix9]
        normali_factor_j = normali_factor[matrix10]
        normali_factor_matrix = np.sqrt(normali_factor_i * normali_factor_j)
        
        dCdp_list = []
        for i in range(len(kcenter)):
            sk = skcenter[i*self.subN : i*self.subN+self.subN]
            kmatrix = sk[matrix7]
            shell_avg_weightj_cut = shell_avg_weightj[i*self.subN:i*self.subN+self.subN, :]
            
            list = []
            for j in range(len(kcenter)):
                window_matrix = shell_avg_weightj_cut * shell_avg_weightj_cut[:, j][matrix7]
                int = simps( kmatrix**2/(2*pi**2) * window_matrix, kmatrix, axis=0 )
                list.append(int)
            dCdp_a = np.array(list)/normali_factor_matrix
            dCdp_list.append(dCdp_a)
        dCdp_list = np.array(dCdp_list)
        return dCdp_list






    def dCXidp(self):
        """ C,a = dCxi_ij / dp_a """
        """ weight : top hat """
        
        from numpy.linalg import pinv
        from numpy import pi
        
        kcenter = self.kcenter
        skcenter = self.skcenter
        rcenter = self.rcenter
        dr = self.dr
        rmin = self.rmin
        rmax = self.rmax
        
        
        matrix1, matrix2 = np.mgrid[0:self.subN , 0:len(rcenter)]
        
        rminmatrix = rmin[matrix2]
        rmaxmatrix = rmax[matrix2]
        
        R = self.RMAX
        V = 4./3 * pi * R**3
        Vir = 4 * pi * rcenter**2 * dr * (1. + 1./12 * (dr/rcenter)**2)
        
        
        dCdp_list = []
        for i in range(len(kcenter)):
            sk = skcenter[i*self.subN : i*self.subN+self.subN]
            skmatrix = sk[matrix1]
            AvgBesselmatrix = avgBessel(0.0,skmatrix,rminmatrix,rmaxmatrix)/Vir[matrix2]
            list = []
            for j in range(len(rcenter)):
                int = simps( skmatrix**2/(2*pi**2) * AvgBesselmatrix * AvgBesselmatrix[:,j][matrix1], skmatrix, axis=0 )
                list.append(int)
            dCdp_a = np.array(list)
            dCdp_list.append(dCdp_a)
        dCdp_list = np.array(dCdp_list)/V
        return dCdp_list



    def Quadratic_Fisher(self, Cov, dCov):
        
        from numpy.linalg import pinv
        
        invC = inv(Cov)
        
        Fisher = []
        for dC1 in dCov:
            for dC2 in dCov:
                Fisher_ab = 1./2 * np.trace( np.dot( np.dot( np.dot( invC, dC1  ), invC ), dC2  ))
                #Fisher_ab = 1./2 * np.trace( np.dot( np.dot( np.dot( invC, invC  ), dC1 ), dC2  ))
                Fisher.append(Fisher_ab)
        
        Fisher = np.reshape( np.array(Fisher), (len(dCov), len(dCov)))
        
        return Fisher


    def Quadratic_Cov(self, Cov, dCov):
        
        from numpy.linalg import pinv
        kcenter = self.kcenter
        dk = self.dk
        rcenter = self.rcenter
        dr = self.dr
        
        Vir = 4 * pi * rcenter**2 * dr * (1. + 1./12 * (dr/rcenter)**2)
        V = 4./3 * np.pi * self.RMAX**3
        invC = inv(Cov)
        
        Fisher = []
        for dC1 in dCov:
            for dC2 in dCov:
                Fisher_ab = 1./2 * np.trace( np.dot( np.dot( np.dot( invC, dC1  ), invC ), dC2  ))
                #inv_Cov_k = inv( np.dot(Cov, Cov) )
                #Fisher_ab = 1./2 * np.trace( np.dot( np.dot( dC1 ,inv_Cov_k), dC2  ))
                Fisher.append(Fisher_ab)
        
        Fisher = np.reshape( np.array(Fisher), (len(dCov), len(dCov)))
        Vi = 4 * pi * kcenter**2 * dk * (1. + 1./12 * (dk/kcenter)**2)
        
        Mode = (2 * np.pi)**3 / Vi /V
        matrix1, matrix2 = np.mgrid[0:len(kcenter), 0: len(kcenter)]
        Cov = inv(Fisher) * np.sqrt(Mode[matrix1] * Mode[matrix2]) #Mode * np.identity(len(kcenter))
        
        return Cov



    def Covaiance_window(self):
        
        klist = self.klist
        kcenter = self.kcenter
        skcenter = self.skcenter
        skbin = self.skbin
        dk = self.dk
        mulist = self.mulist
        dlnk = self.dlnk
        matterpower = self.RealPowerBand
        
        matrix1, matrix2 = np.mgrid[0:len(skcenter),0:len(skcenter)]
        matrix3, matrix4 = np.mgrid[0:len(skcenter), 0:self.subN]
        matrix5, matrix6 = np.mgrid[0:len(skcenter), 0:len(kcenter)]
        matrix7, matrix8 = np.mgrid[0:len(kcenter), 0:len(kcenter)]
        
        kmatrix = skcenter[matrix1]
        kjmatrix = skcenter[matrix2]
        
        R = self.RMAX  #subtract rmin ? check """
        V = 4./3 * pi * R**3
        diagonal = np.sqrt(V)
        
        
        # window function shell averaging
        
        Va = 4 * pi * kcenter**2 * dk * (1. + 1./12 * (dk/kcenter)**2)
        
        weightj = self.window_tophat_k(kmatrix, kjmatrix)
        np.fill_diagonal(weightj, diagonal)
        
        weightj_list=[]
        squared_weightj_list=[]
        for i in range(len(kcenter)):
            sk = skcenter[i*self.subN:i*self.subN+self.subN]
            skmatrix = sk[matrix4]
            weightj_cut = weightj[:,i*self.subN:i*self.subN+self.subN]
            int = simps(4 * np.pi * skmatrix**2 * weightj_cut ,skmatrix, axis=1 )
            int2 = simps(4 * np.pi * skmatrix**2 * weightj_cut**2 ,skmatrix, axis=1 )
            weightj_list.append(int)
            squared_weightj_list.append(int2)
        
        shell_avg_weightj = np.transpose(np.array(weightj_list)) /Va[matrix6]
        squared_shell_avg_weightj = np.transpose(np.array(squared_weightj_list)) /Va[matrix6]
        
        
        kmatrix2 = skcenter[matrix5]
        Pmatrix = matterpower[matrix5]
        normali_factor = simps(kmatrix2**2/(2*pi**2) * shell_avg_weightj**2, kmatrix2, axis=0 )
        normali_factor_i = normali_factor[matrix7]
        normali_factor_j = normali_factor[matrix8]
        normali_factor_matrix = np.sqrt(normali_factor_i * normali_factor_j)
        
        list = []
        for i in range(len(kcenter)):
            
            squared_window_matrix = squared_shell_avg_weightj * squared_shell_avg_weightj[:, i][matrix5]
            signal = 2./V * simps( kmatrix2**2/(2*pi**2) * (Pmatrix + 1./self.nn)**2 * squared_window_matrix , kmatrix2, axis=0 )
            #signal1 = 1./V * simps( kmatrix2**2/(2*pi**2) * (Pmatrix)**2 * squared_window_matrix , kmatrix2, axis=0 )
            #noise = 1./self.nn * simps( kmatrix2**2/(2*pi**2) * window_matrix, kmatrix2, axis=0 )
            list.append(signal)
        #noise_list.append(noise)
        
        self.covariance_P = np.array(list) /normali_factor_matrix**2
        
        return self.covariance_P


    def Covaiance_window_Xi(self):
    
        from sbess import sbess
        klist = self.klist
        kcenter = self.kcenter
        skcenter = self.skcenter
        skbin = self.skbin
        dk = self.dk
        mulist = self.mulist
        dlnk = self.dlnk
        matterpower = self.RealPowerBand
        rcenter = self.rcenter
    
        matrix1, matrix2 = np.mgrid[0:len(skcenter),0:len(rcenter)]
        matrix3, matrix4 = np.mgrid[0:len(rcenter), 0:len(rcenter)]
    
        skmatrix = skcenter[matrix1]
        Pmmatrix = matterpower[matrix1]
    
    
        R = self.RMAX  #subtract rmin ? check """
        V = 4./3 * pi * R**3
    
        result = []
        for i in range(len(rcenter)): # for r_a loop
            #for j in range(len(rcenter)): # for r_b loop
            spherical_Bessel_matrix = sbess(skmatrix * np.fabs(rcenter - rcenter[i]) ,0.0)
            int = simps(skmatrix**2 / (2 * np.pi**2) * (Pmmatrix + 1./self.nn)**2 * spherical_Bessel_matrix, skmatrix, axis=0 )
            result.append(int)
    
        print np.shape(result)
    
        Rmatrix = (1 - rcenter[matrix3]/R)**3
        result2 = 4 * (2 * np.pi)**3 /V * Rmatrix * np.transpose(Rmatrix) * np.array(result)
    
        return result2



    def dXidp(self):
    
        from numpy.linalg import pinv
        from sbess import sbess
    
        kcenter = self.kcenter
        skcenter = self.skcenter
        dk = self.dk
        rcenter = self.rcenter
    
        matrix7, matrix8 = np.mgrid[0:self.subN,0:len(rcenter)]
        matrix9, matrix10 = np.mgrid[0:len(kcenter),0:len(rcenter)]
        rmatrix = rcenter[matrix8]
    
    
        R = self.RMAX  #subtract rmin ? check """
        V = 4./3 * pi * R**3
    
        dXidp_list = []
        for i in range(len(kcenter)):
        
            sk = skcenter[i*self.subN : i*self.subN+self.subN]
            skmatrix = sk[matrix7]
            Bessel_matrix = sbess(skmatrix * rmatrix, 0.0)
            int = simps( skmatrix**2/(2*pi**2) * Bessel_matrix, sk, axis=0 )
            dXidp_list.append(int)
    
        Rmatrix = (1. - rcenter[matrix10]/R)**3
        dXidp_matrix = Rmatrix * dXidp_list
        return dXidp_matrix








def get_closest_index_in_data( value, data ):

    for i in range(len(data)):
        if data[i] < value : pass
        elif data[i] >= value :
            if np.fabs(value - data[i]) > np.fabs(value - data[i-1]):
                value_index = i-1
            else : value_index = i
            break
    
    return value_index


def Ll(l,x):
    
    """ Legendre Polynomial
        call fortran module in legen.f90 """
    
    import numpy as np
    from numpy import vectorize
    from legen import eval_legendre
    
    Le_func = lambda lp,xp : eval_legendre(lp,xp)
    Le_func = np.vectorize(Le_func)
    result = Le_func(l,x)
    
    return result

def avgBessel(l,k,rmin,rmax):
    
    """ Averaged spherical Bessel function in configuration space
        call fortran module in sici.f90 (sine integral ftn) """
    
    from numpy import vectorize, pi, cos, sin
    from sici import sici
    sici = vectorize(sici)
    
    if l == 0. :
        result = (4. * pi * (-k * rmax * cos(k * rmax) + k * rmin * cos(k * rmin) + sin(k * rmax) - sin(k * rmin)))/(k**3)
    elif l == 2. :
        result = 4.*pi* (k * rmax * cos(k * rmax) - k*rmin*cos(k*rmin)-4*sin(k*rmax) +
                          4*sin(k*rmin) + 3*sici(k * rmax) - 3*sici(k*rmin))/k**3
    else :
        result = (2.* pi/k**5) * ((105 * k/rmax - 2 * k**3 * rmax) * \
                                  cos(k * rmax) + (- (105 * k/rmin) + 2 *k**3 *rmin) *\
                                  cos(k * rmin) + 22 * k**2 * sin(k *rmax) - (105 * sin(k * rmax))/rmax**2 -\
                                  22 * k**2 *sin(k * rmin) + (105 * sin(k * rmin))/rmin**2 +\
                                  15 * k**2 * (sici(k * rmax) - sici(k * rmin)))
    return result


def CombineMatrix3by3(cov00, cov01, cov02, cov10, cov11, cov12, cov20, cov21, cov22):
    C_Matrix = np.array([[cov00,cov01,cov02],\
                        [cov10,cov11,cov12],\
                        [cov20,cov21,cov22]])
    return C_Matrix

def CombineMatrix2by2(cov00, cov01, cov10, cov11):
        #
        #   Input should be matrix
        #   matrices = [00, 02, 22]
        #
    C_Matrix = np.array([[cov00,cov01],\
                        [cov10,cov11]])
    return C_Matrix

def CombineMatrix3by2(cov00, cov01, cov10, cov11, cov20, cov21):
    C_Matrix = np.array([[cov00,cov01],\
                        [cov10,cov11],\
                        [cov20,cov21]])

    return C_Matrix

def CombineCovariance3(l, matrices):

    """ Input should be a list of matrices :
        matrices = [00, 02, 04, 22, 24, 44] """

    cov00 = matrices[0][0:l+1,0:l+1]
    cov02 = matrices[1][0:l+1,0:l+1]
    cov04 = matrices[2][0:l+1,0:l+1]
    cov20 = matrices[3][0:l+1,0:l+1]
    cov22 = matrices[4][0:l+1,0:l+1]
    cov24 = matrices[5][0:l+1,0:l+1]
    cov40 = matrices[6][0:l+1,0:l+1]
    cov42 = matrices[7][0:l+1,0:l+1]
    cov44 = matrices[8][0:l+1,0:l+1]
    
    C_Matrix1 = np.concatenate((cov00, cov02, cov04), axis=1)
    C_Matrix2 = np.concatenate((cov20, cov22, cov24), axis=1)
    C_Matrix3 = np.concatenate((cov40, cov42, cov44), axis=1)
    C_Matrix = np.vstack((C_Matrix1, C_Matrix2, C_Matrix3))

    return C_Matrix


def CombineCrossCovariance3(l1, l2, matrices):

    """ Input should be a list of matrices :
        matrices = [00, 02, 04, 20, 22, 24, 40, 42, 44] """
        
    cov00 = matrices[0][0:l1+1,0:l2+1]
    cov02 = matrices[1][0:l1+1,0:l2+1]
    cov04 = matrices[2][0:l1+1,0:l2+1]
    cov20 = matrices[3][0:l1+1,0:l2+1]
    cov22 = matrices[4][0:l1+1,0:l2+1]
    cov24 = matrices[5][0:l1+1,0:l2+1]
    cov40 = matrices[6][0:l1+1,0:l2+1]
    cov42 = matrices[7][0:l1+1,0:l2+1]
    cov44 = matrices[8][0:l1+1,0:l2+1]
    
    C_Matrix1 = np.concatenate((cov00, cov02, cov04), axis=1)
    C_Matrix2 = np.concatenate((cov20, cov22, cov24), axis=1)
    C_Matrix3 = np.concatenate((cov40, cov42, cov44), axis=1)
    C_Matrix = np.vstack((C_Matrix1, C_Matrix2, C_Matrix3))

    return C_Matrix


def CombineCovariance2(l, matrices):
    
    """ Input should be a list of matrices :
        matrices = [00, 02, 04, 20, 22, 24, 40, 42, 44] """
    
    cov00 = matrices[0][0:l+1,0:l+1]
    cov02 = matrices[1][0:l+1,0:l+1]
    cov20 = matrices[3][0:l+1,0:l+1]
    cov22 = matrices[4][0:l+1,0:l+1]
    
    C_Matrix1 = np.concatenate((cov00, cov02), axis=1)
    C_Matrix2 = np.concatenate((cov20, cov22), axis=1)
    C_Matrix = np.vstack((C_Matrix1, C_Matrix2))

    return C_Matrix


def CombineCrossCovariance2(l1, l2, matrices):
    """ Input should be a list of matrices :
        matrices = [00, 02, 04, 20, 22, 24, 40, 42, 44] """
    
    cov00 = matrices[0][0:l1+1,0:l2+1]
    cov02 = matrices[1][0:l1+1,0:l2+1]
    cov20 = matrices[3][0:l1+1,0:l2+1]
    cov22 = matrices[4][0:l1+1,0:l2+1]
    
    C_Matrix1 = np.concatenate((cov00, cov02), axis=1)
    C_Matrix2 = np.concatenate((cov20, cov22), axis=1)
    C_Matrix = np.vstack((C_Matrix1, C_Matrix2))
    
    return C_Matrix


def CombineDevXi(l, matrices):
    """ Input should be a list of matrices :
        matrices = [00, 02, 04, 20, 22, 24, 40, 42, 44] """

    dxib0 = matrices[0][0:l+1]
    dxib2 = matrices[1][0:l+1]
    dxib4 = matrices[2][0:l+1]
    dxif0 = matrices[3][0:l+1]
    dxif2 = matrices[4][0:l+1]
    dxif4 = matrices[5][0:l+1]
    dxis0 = matrices[6][0:l+1]
    dxis2 = matrices[7][0:l+1]
    dxis4 = matrices[8][0:l+1]
    
    Matrix1 = np.concatenate((dxib0, dxib2, dxib4), axis=1)
    Matrix2 = np.concatenate((dxif0, dxif2, dxif4), axis=1)
    Matrix3 = np.concatenate((dxis0, dxis2, dxis4), axis=1)
    Xi = np.vstack((Matrix1, Matrix2, Matrix3))

    Matrix1 = np.concatenate((dxib0, dxib2), axis=1)
    Matrix2 = np.concatenate((dxif0, dxif2), axis=1)
    Matrix3 = np.concatenate((dxis0, dxis2), axis=1)
    Xi2 = np.vstack((Matrix1, Matrix2, Matrix3))
    
    return Xi, Xi2

def CombineDevXi3(l, matrices):
    """ Input should be a list of matrices :
        matrices = [00, 02, 04, 20, 22, 24, 40, 42, 44] """
    
    cov00 = matrices[0][:,0:l+1]
    cov02 = matrices[1][:,0:l+1]
    cov04 = matrices[2][:,0:l+1]
    cov20 = matrices[3][:,0:l+1]
    cov22 = matrices[4][:,0:l+1]
    cov24 = matrices[5][:,0:l+1]
    cov40 = matrices[6][:,0:l+1]
    cov42 = matrices[7][:,0:l+1]
    cov44 = matrices[8][:,0:l+1]
    
    C_Matrix1 = np.concatenate((cov00, cov02, cov04), axis=1)
    C_Matrix2 = np.concatenate((cov20, cov22, cov24), axis=1)
    C_Matrix3 = np.concatenate((cov40, cov42, cov44), axis=1)
    Xi = np.vstack((C_Matrix1, C_Matrix2, C_Matrix3))
    
    C_Matrix1 = np.concatenate((cov00, cov02), axis=1)
    C_Matrix2 = np.concatenate((cov20, cov22), axis=1)
    Xi2 = np.vstack((C_Matrix1, C_Matrix2))
    
    return Xi, Xi2


def FisherProjection( deriv, CovMatrix ):
    
    """ Projection for Fisher Matrix """
    
    FisherMatrix = np.dot(np.dot(deriv, inv(CovMatrix)), np.transpose(deriv))
    for i in range(len(deriv)):
        for j in range(i, len(deriv)):
            FisherMatrix[j,i] = FisherMatrix[i,j]
    
    return FisherMatrix

def FisherProjection_Fishergiven( deriv, FisherM ):
    
    """ Projection for Fisher Matrix """
    
    FisherMatrix = np.dot(np.dot(deriv, FisherM), np.transpose(deriv))
    for i in range(len(deriv)):
        for j in range(i, len(deriv)):
            FisherMatrix[j,i] = FisherMatrix[i,j]
    
    return FisherMatrix

def FractionalError( param1, param2, CovarianceMatrix  ):
    
    """ fractional error on Parameter  \sigma P / P """
        
    error = np.sqrt(CovarianceMatrix.diagonal())
    return error[0]/param1, error[1]/param2


def FractionalErrorBand( params, CovarianceMatrix  ):
    
    """ fractional error on Parameter  \sigma P / P """
    
    error = np.sqrt(CovarianceMatrix.diagonal())
    return error/params


def CrossCoeff( Matrix ):
    
    """ Cross Corelation matrix   C_ij / Sqrt( C_ii * C_jj) """
    
    matrix1,matrix2 = np.mgrid[0:len(Matrix[0]),0:len(Matrix[0])]
    diagonal = Matrix.diagonal()
    Coeff = Matrix /np.sqrt(diagonal[matrix1] * diagonal[matrix2])
    return Coeff

def cumulative_SNR( data_Vec, Cov ):

    cumul_SNR = []
    for i in range(len(data_Vec)):
        InvCov = np.linalg.inv(Cov[0:i+1, 0:i+1])
        SNR = np.dot( np.dot( data_Vec[0:i+1], InvCov ), data_Vec[0:i+1])
        cumul_SNR.append(SNR)
        #print np.shape(data_Vec[0:i+1]),np.shape(InvCov)
    return np.array(cumul_SNR)
                     



def plotting(file, file2, pdf_name, title):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    Fl = np.loadtxt(file)
    Fl2 = np.loadtxt(file2)
    #Fl3=np.loadtxt('RSDPinversefisher.txt')
    Fl3=np.loadtxt('inversefisher_k_base.txt')

    position = np.array(Fl[:,0])
    Sigma_b = np.array(Fl[:,1])
    Sigma_f = np.array(Fl[:,2])
    Sigma_b_s = np.array(Fl[:,3])
    Sigma_f_s = np.array(Fl[:,4])

    position2 = np.array(Fl2[:,0])
    Sigma_b2 = np.array(Fl2[:,1])
    Sigma_f2 = np.array(Fl2[:,2])
    Sigma_b2_s = np.array(Fl2[:,3])
    Sigma_f2_s = np.array(Fl2[:,4])

    position3 = np.array(Fl3[:,0])
    Sigma_b3 = np.array(Fl3[:,1])
    Sigma_f3 = np.array(Fl3[:,2])

    fig1=plt.figure()

    plt.subplot(212)
    plt.semilogy(position,Sigma_b,'b-',label='all modes, determined')
    plt.semilogy(position,Sigma_b_s,'r-',label='all modes, marginalized')
    plt.semilogy(position2,Sigma_b2,'b--',label='two modes, determined')
    plt.semilogy(position2,Sigma_b2_s,'r--',label='two modes, marginalized')
    plt.semilogy(position3,Sigma_b3,'-.',label='fourier space')
    plt.xlim(0,60)
    plt.ylim(0.0005,5*10**(-2))
    plt.legend(loc=4,prop={'size':8})
    #plt.ylim(0.7 * np.min(Sigma_b),10**(-17))

    #plt.legend(loc=2)
    plt.xlabel('$r~(h~Mpc)$')
    plt.ylabel('Fractional Error on b')


    plt.subplot(211)

    plt.plot(position,Sigma_f,'b-',label='all modes, determined')
    plt.plot(position,Sigma_f_s,'r-',label='all modes, marginalized')
    plt.plot(position2,Sigma_f2,'b--',label='two mode 0 and 2, determined')
    plt.plot(position2,Sigma_f2_s,'r--',label='two modes 0 and 2, marginalized')
    plt.plot(position3,Sigma_f3,'-.',label='fourier space')
    plt.xlim(0,60)
    plt.ylim(0,0.07)
    plt.legend(loc=4,prop={'size':8})
    plt.xlabel('$r~(h~Mpc)$')
    plt.ylabel('Fractional Error on f')
    plt.title(title, fontsize=12, fontweight ='bold')
    #plt.title('Correlation Function', fontsize=14, fontweight ='bold')
    plt.show()

    pdf=PdfPages(pdf_name)
    pdf.savefig(fig1)
    pdf.close()
    print "\n pdf file saved : ", pdf_name


def plotting2(file, file2, pdf_name, title):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    Fl = np.loadtxt(file)
    Fl2 = np.loadtxt(file2)
    #Fl3=np.loadtxt('fourier_k.txt')
    Fl3=np.loadtxt('inversefisher_k_base.txt')
    
    position = np.array(Fl[:,0])
    Sigma_b = np.array(Fl[:,1])
    Sigma_f = np.array(Fl[:,2])
    Sigma_b_s = np.array(Fl[:,3])
    Sigma_f_s = np.array(Fl[:,4])
    
    position2 = np.array(Fl2[:,0])
    Sigma_b2 = np.array(Fl2[:,1])
    Sigma_f2 = np.array(Fl2[:,2])
    Sigma_b2_s = np.array(Fl2[:,3])
    Sigma_f2_s = np.array(Fl2[:,4])
    
    position3 = np.array(Fl3[:,0])
    Sigma_b3 = np.array(Fl3[:,1])
    Sigma_f3 = np.array(Fl3[:,2])
    
    fig1=plt.figure()
    
    plt.subplot(212)
    plt.semilogy(position,Sigma_b,'b-',label='all modes, determined')
    plt.semilogy(position,Sigma_b_s,'r-',label='all modes, marginalized')
    plt.semilogy(position2,Sigma_b2,'b--',label='two modes, determined')
    plt.semilogy(position2,Sigma_b2_s,'r--',label='two modes, marginalized')
    #plt.semilogy(position3,Sigma_b3,'-.',label='fourier space')
    plt.xlim(0,60)
    plt.ylim(0.0005,5*10**(-2))
    plt.legend(loc=4,prop={'size':8})
    #plt.ylim(0.7 * np.min(Sigma_b),10**(-17))
    
    #plt.legend(loc=2)
    plt.xlabel('$r~(h~Mpc)$')
    plt.ylabel('Fractional Error on b')
    
    
    plt.subplot(211)
    
    plt.plot(position,Sigma_f,'b-',label='all modes, determined')
    plt.plot(position,Sigma_f_s,'r-',label='all modes, marginalized')
    plt.plot(position2,Sigma_f2,'b--',label='two mode 0 and 2, determined')
    plt.plot(position2,Sigma_f2_s,'r--',label='two modes 0 and 2, marginalized')
    #plt.plot(position3,Sigma_f3,'-.',label='fourier space')
    plt.xlim(0,60)
    plt.ylim(0,0.07)
    plt.legend(loc=4,prop={'size':8})
    plt.xlabel('$r~(h~Mpc)$')
    plt.ylabel('Fractional Error on f')
    plt.title(title, fontsize=12, fontweight ='bold')
    #plt.title('Correlation Function', fontsize=14, fontweight ='bold')
    plt.show()
    
    pdf=PdfPages(pdf_name)
    pdf.savefig(fig1)
    pdf.close()
    print "\n pdf file saved : ", pdf_name



def time_measure(function):
    
    import timeit
    print '\n-------------------------------------------'
    print ' Time measurement for derivative_xi'
    time_deriv = timeit.timeit(function, setup = 'from __main__ import *', number=1)
    print ' time :', time_deriv , 'second(s)'
    print '-------------------------------------------'


def confidence_ellipse(x_mean, y_mean, labellist, *args, **kwargs ):
    import numpy as np
    import matplotlib.pyplot as plt
    from pylab import figure, show, rand
    from matplotlib.patches import Ellipse
    from matplotlib.backends.backend_pdf import PdfPages

    basename = kwargs.get('basename','k')
    title = kwargs.get('title', 'Confidence Ellipse, bandpower l=0,2,4')
    pdfname = kwargs.get('pdfname', 'conficence_test.pdf')
    
    
    # For BAO and RSDscales
    
    xmin = kwargs.get('xmin', x_mean*0.97 )
    xmax = kwargs.get('xmax', x_mean*1.03 )
    ymin = kwargs.get('ymin', y_mean*0.94 )
    ymax = kwargs.get('ymax', y_mean*1.06 )
    """
    xmin = kwargs.get('xmin', x_mean*0.99 )
    xmax = kwargs.get('xmax', x_mean*1.01 )
    ymin = kwargs.get('ymin', y_mean*0.97 )
    ymax = kwargs.get('ymax', y_mean*1.03 )
    """

    linecolor = ['r', 'b', 'r', 'b', 'g', 'm']
    linestyle = ['solid', 'solid', 'dashed', 'dashed', 'dashdot', 'dotted']
    ziplist = zip(args, linecolor, linestyle)
    
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]
    

    fig = figure()
    ax = fig.add_subplot(111)
    
    elllist = []
    
    for z in ziplist:
        vals, vecs = eigsorted(z[0])
        #print "values :", vals
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        nstd = np.sqrt(5.991) # 95% :4.605  #99% :9.210
        w, h = 2 * nstd * np.sqrt(vals)
        ell = Ellipse(xy=(x_mean, y_mean),
              width=w, height=h,
              angle=theta, color = z[1], ls = z[2], lw=1.5, fc= 'None')
              
        elllist.append(ell)

    for e in elllist:
        ax.add_artist(e)
        #e.set_alpha(0.2)
        e.set_clip_box(ax.bbox)

    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('b')
    ax.set_ylabel('f')
    plt.legend(elllist, labellist, loc=4, prop={'size':10})
    plt.scatter(x_mean, y_mean)
    plt.title( title )
    
    pdf_name = pdfname
    pdf=PdfPages(pdf_name)
    pdf.savefig(fig)
    pdf.close()
    print "\n pdf file saved : ", pdf_name



def Linear_plot( base, valuename, *args, **kwargs ):
    #
    #
    #
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    basename = kwargs.get('basename','k')
    title = kwargs.get('title', 'Fractional Error')
    pdfname = kwargs.get('pdfname', 'test.pdf')
    xmin = kwargs.get('xmin',10**(-4))
    xmax = kwargs.get('xmax', 1000.)
    ymin = kwargs.get('ymin', 10**(-7))
    ymax = kwargs.get('ymax', 10**(5))
    scale = kwargs.get('scale', None )
    ylabel = kwargs.get('ylabel', 'Fractional Error')
    
    #linestyles = ['b-', 'r.', 'g^','c.', 'm--', 'y.', 'k.']
    linestyles = ['b-', 'r^', 'g.','m-', 'g-', 'y--', 'k--','ro','co', 'mo', 'yo', 'ko']
    ziplist = zip(args, valuename, linestyles)
    
    fig = plt.figure()
    fig.suptitle( title , fontsize=10 )
    
    if scale == None:
        for z in ziplist: plt.plot( base, z[0], z[2], label = z[1] )
    elif scale == 'log':
        for z in ziplist: plt.loglog( base, z[0], z[2], label = z[1] )
    elif scale == 'semilogy':
        for z in ziplist: plt.semilogy( base, z[0], z[2], label = z[1] )
    elif scale == 'semilogx':
        for z in ziplist: plt.semilogx( base, z[0], z[2], label = z[1] )

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel( basename )
    plt.ylabel(ylabel)
    plt.legend(loc=3,prop={'size':10})
    plt.grid(True)
    #plt.show()
    
    pdf=PdfPages( pdfname )
    pdf.savefig(fig)
    pdf.close()
    #plt.clf()
    print " pdf file saved : ", pdfname

def Linear_plot2( base, base2, value2, valuename, *args, **kwargs ):
    #
    #
    #
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    
    basename = kwargs.get('basename','k')
    title = kwargs.get('title', 'Fractional Error')
    pdfname = kwargs.get('pdfname', 'test.pdf')
    xmin = kwargs.get('xmin',10**(-4))
    xmax = kwargs.get('xmax', 1000.)
    ymin = kwargs.get('ymin', 10**(-7))
    ymax = kwargs.get('ymax', 10**(5))
    scale = kwargs.get('scale', None )
    
    #linestyles = ['b-', 'r.', 'g^','c.', 'm--', 'y.', 'k.']
    linestyles = ['b--', 'r^', 'g.','c--', 'm--', 'y--', 'k--','ro','co', 'mo', 'yo', 'ko']
    ziplist = zip(args, valuename, linestyles)
    
    fig = plt.figure()
    fig.suptitle( title , fontsize=10 )
    
    if scale == None:
        for z in ziplist: plt.semilogx( base, z[0], z[2], label = z[1] )
    elif scale == 'log':
        for z in ziplist: plt.loglog( base, z[0], z[2], label = z[1] )
        plt.loglog(base2, value2, 'b-')
    
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel( basename )
    plt.ylabel('Fractional Error')
    plt.legend(loc=3,prop={'size':10})
    plt.grid(True)
    #plt.show()
    
    pdf=PdfPages( pdfname )
    pdf.savefig(fig)
    pdf.close()
    #plt.clf()
    print " pdf file saved : ", pdfname



def Contour_plot( base, crosscoeffdata, **kwargs ):
    #
    #   Make 2-D Contour Plot for Covariance Matrix and Fisher Matrix
    #
    import numpy as np
    import matplotlib.cm as cm
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    
    
    basename = kwargs.get('basename','log10(k)')
    title = kwargs.get('title', 'Covariance')
    pdfname = kwargs.get('pdfname', 'test.pdf')
    scale = kwargs.get('scale', None )
    
    k1 = np.log10(base)
    k2 = np.log10(base)
    
    fig, ax = plt.subplots()
    ax.set_title( title )
    ax.set_xlabel( basename )
    ax.set_ylabel( basename )
    
    if scale == None:
        data = crosscoeffdata
        label = 'Amplitude'
    elif scale == 'log':
        data = np.log10(crosscoeffdata)
        label = '$\log_{10}$(Amplitude)'
    elif scale == 'asinh':
        data = np.arcsinh(1.0 * crosscoeffdata)/1.0
        label = 'Amplitude'
    
    cax = ax.imshow(data, extent=(k1.min(), k1.max(), k2.max(), k2.min()), interpolation='nearest', cmap=cm.gist_rainbow)
                    
    cbar = fig.colorbar(cax, ax = ax )
    cbar.set_label( label )
                    
    pdf=PdfPages( pdfname )
    pdf.savefig(fig)
    pdf.close()
    print " pdf file saved : ", pdfname

def makedirectory(dirname):
    import os
    if not os.path.exists("./" + dirname + "/"):
        os.mkdir("./" + dirname + "/")
