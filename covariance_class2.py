import numpy as np
from numpy import zeros, sqrt, pi, sin, cos, exp
from numpy.linalg import inv
from numpy import vectorize
from scipy.interpolate import interp1d
from scipy.integrate import simps, quad
#from scipy.special import eval_legendre
from legen import eval_legendre
from numpy import vectorize
from sbess import sbess

class RSD_covariance():

    def __init__(self, KMIN, KMAX, RMIN, RMAX, n, n2):

        #parameter
        self.h=1.0
        self.b=2.0
        self.f=0.74
        self.Vs= 5.0*10**9 # survey volume
        self.s= 3.5  # sigma in Power spectrum
        self.nn=3.0 * 10**(-4) # shot noise : \bar{n}

        # k scale range
        self.KMIN = KMIN
        self.KMAX =  KMAX

        # r scale
        self.RMIN = RMIN #30.#0.1 #0.1 for Reid   #1.15 * np.pi / self.kmax
        self.RMAX = RMAX #180. #180. for Reid  #1.15 * np.pi / self.kmin

        self.n = n # converge # 201 for Reid # 201 # the number of k bins. the sample should be odd
        self.n2 = n2 #101 for Reid # number of r bins
        self.n3 = 21 # 101 for Reid number of mu bins

        #self.klist = np.logspace(np.log(self.KMIN),np.log(self.KMAX),self.n, base = np.e)


        self.rlist = np.logspace(np.log(self.RMAX),np.log(self.RMIN),self.n2, base = np.e)
        rlist = self.rlist
        self.rmax = np.delete(rlist,-1)
        self.rmin = np.delete(rlist,0)
        self.rcenter = np.array([ (rlist[i] + rlist[i+1])/2. for i in range(len(rlist)-1) ])
        
        self.dr = np.abs(self.rmax - self.rmin)
        self.dlnr = np.log(self.rmax/self.rmin)[3]
        
        self.mulist = np.linspace(-1.,1.,self.n3)
        
        self.subN = 51 #101 for Reid
        subN = self.subN
        
        self.skbin = np.logspace(np.log10(self.KMIN),np.log10(self.KMAX), subN * self.n + 1, base=10) #For Reid, delete '+1'
        self.skcenter = np.array([(self.skbin[i] + self.skbin[i+1])/2. for i in range(len(self.skbin)-1)])
        self.skmin = np.delete(self.skbin,-1)
        self.skmax = np.delete(self.skbin,0)
        self.sdlnk = np.log(self.skmax/self.skmin)
        
        self.klist = np.array([self.skbin[i*subN] for i in range(len(self.skbin)/subN + 1)]) #For Reid, delete '+1'


        self.kcenter = np.array([(self.klist[i] + self.klist[i+1])/2. for i in range(len(self.klist)-1)])
        self.kmin = np.delete(self.klist,-1)
        self.kmax = np.delete(self.klist,0)
        self.dk = self.kmax - self.kmin
        self.dlnk = np.log(self.kmax/self.kmin)[3]
        
        
        
        print '\n---------------------------------------------------\
        \n Fractional Error for parameter b, f \
        \n z = 0.55 \n number of k bins n ={}, kmax = {}\
        \n number of r bins n2 = {} \
        \n dlnr = {}, dlnk={}, sdlnk={} \
        \n ---------------------------------------------------'.format(self.n, self.KMAX, self.n2, np.log(self.rlist[1]/self.rlist[2]),np.log(self.klist[1]/self.klist[2]), self.sdlnk[2] )


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

    def RSDPowerSpectrum( self, mu ):
        
        k = self.skcenter
        D = np.exp(- k**2 * mu**2 * self.s**2)
        self.RSDPower = (self.b + self.f * mu**2 )**2 * self.RealPowerBand * D
        
        return self.RSDPower
    
    

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
        self.multipole_band = resultlist/Vi
        return self.multipole_band
    

    def RSDband_derivative_P(self,l):
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
            
            """
            Rintb = (2 * l + 1.)/2 * simps( 4 * pi * kmatrix**2 * Rb * Pmmatrix, mumatrix, axis=0 )
            Rintf = (2 * l + 1.)/2 * simps( 4 * pi * kmatrix**2 * Rf * Pmmatrix, mumatrix, axis=0 )
            Rints = (2 * l + 1.)/2 * simps( 4 * pi * kmatrix**2 * Rs * Pmmatrix, mumatrix, axis=0 )
            resultb = simps( muintb, k )
            resultf = simps( muintf, k )
            results = simps( muints, k )
            """
            resultlistb.append(resultb)
            resultlistf.append(resultf)
            resultlists.append(results)
        return resultlistb/Vi, resultlistf/Vi, resultlists/Vi


    def derivative_P_band(self,l):
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
        Pmlist = self.Shell_avg_band()
    
        matrix1, matrix2 = np.mgrid[0:len(mulist),0: self.subN ]
        mumatrix = self.mulist[matrix1]
        Le_matrix = Ll(l,mumatrix)
    
        Vi = 4 * pi * kcenter**2 * dk + 1./3 * pi * (dk)**3
    
        resultlist=[]
        for i in range(len(kcenter)):
            k = skcenter[i*self.subN : i*self.subN+self.subN]
            kmatrix = k[matrix2]
            Dmatrix = np.exp(- kmatrix**2 * mumatrix**2 * self.s**2) #FOG matrix
            R = (b + f*mumatrix**2)**2 * Dmatrix

            Rintegral = (2*l + 1.)/2 * simps( R * Le_matrix, mumatrix, axis = 0 )
            result = simps( 4 * pi * k**2 * Rintegral , k )

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
        Pmlist = self.Pmlist
        matterpower = self.RealPowerBand
        
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
            
            #R = (b + f*mumatrix**2)**2 * Dmatrix
            
            Rintegral = 1. #(2*l + 1.)/2 * simps( R * Le_matrix, mumatrix, axis = 0 )
            kmatrix2 = k[matrix4]
            AvgBesselmatrix = avgBessel(l, kmatrix2 ,rminmatrix,rmaxmatrix)/Vir
            
            result = np.real(I**l) * simps( kmatrix2**2 * Rintegral * AvgBesselmatrix /(2*pi**2) , kmatrix2, axis = 1 )
            resultlist.append(result)
        
        self.derivative_Xi_bandpower = np.array(resultlist).reshape((len(kcenter),len(rcenter)))
        return self.derivative_Xi_bandpower
    
    

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

        print 'covariance_PP {:>1.0f}{:>1.0f} is finished'.format(l1,l2)
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
        Pmlist = 1. #self.Pmlist
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
        
        print 'covariance_PP {:>1.0f}{:>1.0f} is finished'.format(l1,l2)
        return covariance_mutipole_PP
  
    
    def RSDband_covariance_PXi( self, l1, l2 ):

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
            Pmmatrix = 1. #Pm[matrix4]
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

    

    def derivative_xi(self,l):
        #========================================================
        #
        #   Function for calculating multipole derivatives
        #
        #
        #   Output : dXi/db, dXi/df, dXi/ds three 1d array along r axis
        #            for each mode l ( l = 0,2,4 )
        #
        #========================================================
        
        b = self.b
        f = self.f
        s = self.s
        rlist = self.rlist
        rcenter = self.rcenter
        klist = self.klist
        kcenter = self.kcenter
        mulist = self.mulist
        dlnk = self.dlnk
        Pmlist = self.Shell_avg_band()
        #Pmlist = self.Pmlist
        rmin = self.rmin
        rmax = self.rmax
        dr = self.dr
        
        
        D = lambda k,mu : np.exp(- k**2 * mu**2 * s**2) # Finger of God
        mulist = self.mulist #np.linspace(-1.,1.,self.n3) # angle \mu # odd number of sample for simps
        
        # generating 2-dim matrix for k and mu, matterpower spectrum, FoG term
        matrix1,matrix2 = np.mgrid[0:len(mulist),0:len(kcenter)]
        mulistmatrix = mulist[matrix1] # mu matrix (axis 0)
        klistmatrix = kcenter[matrix2] # k matrix (axis 1)
        Pmlistmatrix = Pmlist[matrix2] # matter power spectrum matrix same with k axis
        Dmatrix = D(klistmatrix,mulistmatrix) #FOG term k and mu matrix
        Ll = lambda l,x : eval_legendre(l,x) #Legendre
        Ll = vectorize(Ll)
        Le_matrix = Ll(l,mulistmatrix)
        
        
        #
        #   derivative dp/db, dp/df, dp/ds
        #
        
        db = (2.*l+1.)/2. * klistmatrix**2 * Pmlistmatrix * Le_matrix * (b + f* mulistmatrix*mulistmatrix)*2 * Dmatrix
        df = (2.*l+1.)/2. * klistmatrix**2 * Pmlistmatrix * (b + f*mulistmatrix*mulistmatrix)*2 * mulistmatrix*mulistmatrix * Le_matrix * Dmatrix
        ds = (2.*l+1.)/2. * klistmatrix**2 * Pmlistmatrix * (b + f*mulistmatrix * mulistmatrix)**2 * (-klistmatrix**2 * mulistmatrix**2) * Le_matrix * Dmatrix
        # mu integration
        intb = simps(db,mulistmatrix, axis=0) # return 1d array along k-axis
        intf = simps(df,mulistmatrix, axis=0) # return 1d array along k-axis
        ints = simps(ds,mulistmatrix, axis=0) # return 1d array along k-axis
    
    
        #bessel function
        
        pf = 1./(2.* pi**2)
        
        matrix1,matrix2 = np.mgrid[0:len(kcenter),0:len(rcenter)]
        rlistmatrix = rcenter[matrix2] # horizontal
        rminmatrix = rmin[matrix2]
        rmaxmatrix = rmax[matrix2]
        klistmatrix = kcenter[matrix1] # axis 0 #redefine to match with r dimension
        dlnk = np.log(kcenter)
        dlnkmatrix = dlnk[matrix1]
    
        intb = intb[matrix1] # convert 1d array to 2 d array for next calculation (same axis with kmatrix)
        intf = intf[matrix1]
        ints = ints[matrix1]
        # Casting 2 dimensional array into double bessel function
        avgBesselmatrix = avgBessel(l,klistmatrix, rminmatrix, rmaxmatrix)
        
        Totalb = intb * pf * avgBesselmatrix
        Totalf = intf * pf * avgBesselmatrix
        Totals = ints * pf * avgBesselmatrix
    
        Vi = 4 * pi * rcenter**2 * dr * (1. + 1./12 * (dr/rcenter)**2)
        
        Total_Integb = (-1.0)**(l/2.) * simps(Totalb,klistmatrix, axis=0)/Vi
        Total_Integf = (-1.0)**(l/2.) * simps(Totalf,klistmatrix, axis=0)/Vi
        Total_Integs = (-1.0)**(l/2.) * simps(Totals,klistmatrix, axis=0)/Vi
    
        return Total_Integb, Total_Integf, Total_Integs


    def RSDband_derivative_xi(self,l):
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
        Pmlist = self.Pmlist
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
        Le_matrix = Ll(l,mumatrix)
        
        Const = (2*l + 1.)/2 * np.real(I**l) / (2 * pi**2)
        resultlistb=[]
        resultlistf=[]
        resultlists=[]
        for i in range(len(kcenter)):
            k = skbin[i*self.subN : i*self.subN+self.subN]
            kmatrix = k[matrix2]
            Dmatrix = np.exp(- kmatrix**2 * mumatrix**2 * self.s**2) #FOG matrix
            Rb = 2 *(self.b + self.f * mumatrix**2) * Dmatrix
            Rf = 2 * mumatrix**2 *(self.b + self.f * mumatrix**2) * Dmatrix
            Rs = (- kmatrix**2 * mumatrix**2)*(self.b + self.f * mumatrix**2)**2 * Dmatrix
            
            muintb = simps( Rb * Le_matrix, mumatrix, axis=0 )
            muintf = simps( Rf * Le_matrix, mumatrix, axis=0 )
            muints = simps( Rs * Le_matrix, mumatrix, axis=0 )
            
            kmatrix2 = k[matrix4]
            Vir = 4 * pi * rmatrix**2 * drmatrix + 1./3 * pi * (drmatrix)**3
            AvgBesselmatrix = avgBessel(l,kmatrix2,rminmatrix,rmaxmatrix)/Vir
            
            resultb = simps( kmatrix2**2 * muintb * AvgBesselmatrix, k, axis = 0 )
            resultf = simps( kmatrix2**2 * muintf * AvgBesselmatrix, k, axis = 0 )
            results = simps( kmatrix2**2 * muints * AvgBesselmatrix, k, axis = 0 )
            
            resultlistb.append(resultb)
            resultlistf.append(resultf)
            resultlists.append(results)
        
        matrix1, matrix2 = np.mgrid[ 0: len(kcenter), 0: len(rcenter)]
        Pmlistmatrix = Pmlist[matrix1]
        
        resultlistb = Const * np.sum(Pmlistmatrix * resultlistb, axis=0)
        resultlistf = Const * np.sum(Pmlistmatrix * resultlistf, axis=0)
        resultlists = Const * np.sum(Pmlistmatrix * resultlists, axis=0)
            
        return resultlistb, resultlistf, resultlists
    


    def covariance(self, l1, l2):
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
        dr = self.dr
        Pmlist = self.Pmlist
        Pm = self.RealPowerBand
        s = self.s
        b = self.b
        f = self.f
        Vs = self.Vs
        nn = self.nn

        #Total integ ftn except double bessel

        #mu-integration=================================================

        D = lambda k,mu : exp(- k**2 * mu**2 * s**2) # Finger of God

        # generating 2-dim matrix for k and mu, matterpower spectrum, FoG term
        matrix1,matrix2 = np.mgrid[0:len(mulist),0:len(skcenter)]
        mulistmatrix = mulist[matrix1] # mu matrix (axis 0)
        klistmatrix = skcenter[matrix2] # k matrix (axis 1)
        Pmlistmatrix = Pm[matrix2] # matter power spectrum matrix same with k axis
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

        constant = (2./Vs) * (2*l1+1.)*(2*l2+1.)/(2*np.pi)**2 * np.real(I**(l1+l2))
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

        matrix1,matrix2,matrix3 = np.mgrid[0:len(skcenter),0:len(rcenter),0:len(rcenter)]
        rlistmatrix1 = rcenter[matrix2] # vertical
        rlistmatrix2 = rcenter[matrix3] # horizontal
        
        klistmatrix = skcenter[matrix1] # axis 0 #redefine to match with r dimension
        dlnk = np.log(skbin)
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
        
        """
        for i in range(len(self.rcenter)):
            for j in np.arange(i, len(self.rcenter)):
                
                Result[j,i] = Result[i,j]
        """
        
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
    
    
        D = lambda k,mu : exp(- k**2 * mu**2 * s**2) # Finger of God
    
        # generating 2-dim matrix for k and mu, matterpower spectrum, FoG term
        matrix1,matrix2 = np.mgrid[0:len(mulist),0:len(skcenter)]
        mulistmatrix = mulist[matrix1] # mu matrix (axis 0)
        klistmatrix = skcenter[matrix2] # k matrix (axis 1)
        Le_matrix1 = Ll(l1,mulistmatrix)
        Le_matrix2 = Ll(l2,mulistmatrix)
        Dmatrix = np.exp(- klistmatrix**2 * mulistmatrix**2 * self.s**2)

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




Ll = lambda l,x : eval_legendre(l,x) #Legendre
Ll = np.vectorize(Ll)


def avgBessel(l,k,rmin,rmax):
    #
    #
    #
    from scipy.special import sici
    sici = np.vectorize(sici)
    if l == 0. :
        result = (4. * pi * (-k * rmax * cos(k * rmax) + k * rmin * cos(k * rmin) + sin(k * rmax) - sin(k * rmin)))/(k**3)
    elif l == 2. :
        result = 4.*pi* (k * rmax * cos(k * rmax) - k*rmin*cos(k*rmin)-4*sin(k*rmax) +
                          4*sin(k*rmin) + 3*sici(k * rmax)[0] - 3*sici(k*rmin)[0])/k**3
    else :
        result = (2.* pi/k**5) * ((105 * k/rmax - 2 * k**3 * rmax) * \
                                  cos(k * rmax) + (- (105 * k/rmin) + 2 *k**3 *rmin) *\
                                  cos(k * rmin) + 22 * k**2 * sin(k *rmax) - (105 * sin(k * rmax))/rmax**2 -\
                                  22 * k**2 *sin(k * rmin) + (105 * sin(k * rmin))/rmin**2 +\
                                  15 * k**2 * (sici(k * rmax)[0] - sici(k * rmin)[0]))
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
        #
        #   Input should be matrix
        #   matrices = [00, 02, 04, 22, 24, 44]
        #
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
    
    
    #C_Matrix = CombineMatrix3by3(cov00, cov02, cov04, np.transpose(cov02), cov22, cov24, np.transpose(cov04), np.transpose(cov24), cov44)
    #C_Matrix = C_Matrix.swapaxes(1,2).reshape(3*(l+1),3*(l+1))
        #C_Matrix_inverse = inv(C_Matrix.reshape(3*(l+1),3*(l+1)))
        #C_Matrix_inverse = np.matrix(C_Matrix_inverse)
    return C_Matrix

def CombineCrossCovariance3(l1, l2, matrices):
    #
    #   Input should be matrix
    #   matrices = [00, 02, 04, 22, 24, 44]
    #
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
        #
        #   Input should be matrix
        #   matrices = [00, 02, 04, 22, 24, 44]
        #
    cov00 = matrices[0][0:l+1,0:l+1]
    cov02 = matrices[1][0:l+1,0:l+1]
    cov20 = matrices[3][0:l+1,0:l+1]
    cov22 = matrices[4][0:l+1,0:l+1]
    
    C_Matrix1 = np.concatenate((cov00, cov02), axis=1)
    C_Matrix2 = np.concatenate((cov20, cov22), axis=1)
    C_Matrix = np.vstack((C_Matrix1, C_Matrix2))

    return C_Matrix


def CombineCrossCovariance2(l1, l2, matrices):
    #
    #   Input should be matrix
    #   matrices = [00, 02, 04, 22, 24, 44]
    #
    cov00 = matrices[0][0:l1+1,0:l2+1]
    cov02 = matrices[1][0:l1+1,0:l2+1]
    cov20 = matrices[3][0:l1+1,0:l2+1]
    cov22 = matrices[4][0:l1+1,0:l2+1]
    
    C_Matrix1 = np.concatenate((cov00, cov02), axis=1)
    C_Matrix2 = np.concatenate((cov20, cov22), axis=1)
    C_Matrix = np.vstack((C_Matrix1, C_Matrix2))
    
    return C_Matrix


def CombineDevXi(l, matrices):

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


def FisherProjection( deriv, CovMatrix ):
    #
    #   Projection for Fisher Matrix
    #
    FisherMatrix = np.dot(np.dot(deriv, inv(CovMatrix)), np.transpose(deriv))
    for i in range(len(deriv)):
        for j in range(i, len(deriv)):
            FisherMatrix[j,i] = FisherMatrix[i,j]
    
    return FisherMatrix

def FractionalError( param1, param2, CovarianceMatrix  ):
        #
        #   fractional error on Parameter  \sigma P / P
        #
    error = np.sqrt(CovarianceMatrix.diagonal())
    return error[0]/param1, error[1]/param2


def FractionalErrorBand( params, CovarianceMatrix  ):
    #
    #   fractional error on Parameter  \sigma P / P
    #
    error = np.sqrt(CovarianceMatrix.diagonal())
    return error/params


def CrossCoeff( base, Matrix ):
    #
    #   Func for getting Cross Corelation matrix   C_ij / Sqrt( C_ii * C_jj)
    #
    matrix1,matrix2 = np.mgrid[0:len(base),0:len(base)]
    diagonal = Matrix.diagonal()
    Coeff = Matrix /np.sqrt(diagonal[matrix1] * diagonal[matrix2])
    return Coeff




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


"""
class Confidence_Level():


    def __init__(self, param1, param2, covmatrix, n1, n2):
        #
        #
        #
        self.param1 = param1
        self.param2 = param2
        self.covmatrix = covmatrix

        self.sigma1square = self.covmatrix[n1,n1]
        self.sigma2square = self.covmatrix[n2,n2]
        self.crosssigma = self.covmatrix[n1,n2]
        
    def ellipse( self, confi_level ):
        
        self.a_square = (self.sigma1square + self.sigma2square)/2. + np.sqrt((self.sigma1square-self.sigma2square)**2/4 + self.crosssigma)
        self.b_square = (self.sigma1square + self.sigma2square)/2. - np.sqrt((self.sigma1square-self.sigma2square)**2/4 + self.crosssigma)

        self.angle = 1./2 * np.arctan(2 * self.crosssigma/(self.sigma1square - self.sigma2square))

        # ellipse equation
        
        wsquare = confi_level * self.a_square
        hsquare = confi_level * self.b_square
        
        ((x - self.param1)*cos(self.angle) - (y-self.param2)*sin(self.angle))**2/wsquare \
        + ((x - self.param1)*sin(self.angle) + (y-self.param2)*sin(self.angle))**2/hsquare = 1
"""

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
        print "values :", vals
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
    plt.legend(elllist, labellist, loc=4, prop={'size':8})
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
    
    #linestyles = ['b-', 'r.', 'g^','c.', 'm--', 'y.', 'k.']
    linestyles = ['b-', 'r^', 'g.','c--', 'm--', 'y--', 'k--','ro','co', 'mo', 'yo', 'ko']
    ziplist = zip(args, valuename, linestyles)
    
    fig = plt.figure()
    fig.suptitle( title , fontsize=10 )
    
    if scale == None:
        for z in ziplist: plt.semilogx( base, z[0], z[2], label = z[1] )
    elif scale == 'log':
        for z in ziplist: plt.loglog( base, z[0], z[2], label = z[1] )
    
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
