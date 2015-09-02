import numpy.f2py.f2py2e as f2py2e
import sys

#sys.argv +=  "-c -m sbess sbess.f90".split()
#f2py2e.main()
sys.argv +=  "-c -m legen legen.f90".split()
f2py2e.main()
