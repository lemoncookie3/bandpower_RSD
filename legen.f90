subroutine eval_legendre(n,x,pl)
!======================================
! calculates Legendre polynomials Pn(x)
! using the recurrence relation
! if n > 100 the function retuns 0.0
!======================================
double precision, intent(out) ::  pl
double precision, intent(in) :: x
double precision pln(0:n)
integer, intent(in) :: n
integer k

pln(0) = 1.0
pln(1) = x

if (n <= 1) then
pl = pln(n)
else
do k=1,n-1
pln(k+1) = ((2.0*k+1.0)*x*pln(k) - float(k)*pln(k-1))/(float(k+1))
end do
pl = pln(n)
end if
return
end