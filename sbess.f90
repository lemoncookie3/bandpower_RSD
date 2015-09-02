MODULE nrtype 
	
    INTEGER, PARAMETER :: I4B = SELECTED_INT_KIND(9) 
    INTEGER, PARAMETER :: I2B = SELECTED_INT_KIND(4) 
    INTEGER, PARAMETER :: I1B = SELECTED_INT_KIND(2) 
    INTEGER, PARAMETER :: SP = KIND(1.0) 
    INTEGER, PARAMETER :: DP = KIND(1.0D0) 
    INTEGER, PARAMETER :: SPC = KIND((1.0,1.0)) 
    INTEGER, PARAMETER :: DPC = KIND((1.0D0,1.0D0)) 
    INTEGER, PARAMETER :: LGT = KIND(.true.) 
    REAL(SP), PARAMETER :: PI=3.141592653589793238462643383279502884197_sp 
    REAL(SP), PARAMETER :: PIO2=1.57079632679489661923132169163975144209858_sp 
    REAL(SP), PARAMETER :: TWOPI=6.283185307179586476925286766559005768394_sp 
    REAL(SP), PARAMETER :: SQRT2=1.41421356237309504880168872420969807856967_sp 
    REAL(SP), PARAMETER :: EULER=0.5772156649015328606065120900824024310422_sp 
    REAL(DP), PARAMETER :: PI_D=3.141592653589793238462643383279502884197_dp 
    REAL(DP), PARAMETER :: PIO2_D=1.57079632679489661923132169163975144209858_dp 
    REAL(DP), PARAMETER :: TWOPI_D=6.283185307179586476925286766559005768394_dp 
   	
    TYPE sprs2_sp 
       INTEGER(I4B) :: n,len 
       REAL(SP), DIMENSION(:), POINTER :: val 
       INTEGER(I4B), DIMENSION(:), POINTER :: irow 
       INTEGER(I4B), DIMENSION(:), POINTER :: jcol 
   
    END TYPE sprs2_sp
    TYPE sprs2_dp 
       INTEGER(I4B) :: n,len 
       REAL(DP), DIMENSION(:), POINTER :: val 
       INTEGER(I4B), DIMENSION(:), POINTER :: irow 
       INTEGER(I4B), DIMENSION(:), POINTER :: jcol 
    END TYPE sprs2_dp
  END MODULE nrtype
 	
!=====================================================================! 
FUNCTION Sbess(X,N) 
!=====================================================================! 
    use nrtype 
    implicit none 
    real(dp), intent(in) :: x 
    integer(i4b), intent(in) :: n 
    real(dp) :: sbess 
    LOGICAL(lgt) Down,Noacc 
    real(dp),  PARAMETER :: Eps=1.e-9_dp 
    real(dp),  PARAMETER :: P1=0.1_dp, P35=0.35_dp, & 
         P5=0.5_dp, C48=48._dp 
   	
! 
    real(dp), DIMENSION(0:999) :: a 
! 
    integer(i4b) :: i, j, nu, nu1, nu2, np, lam, mu 
    real(dp) :: fj, fj1, fj2, ap, b, Alog2e, u, alpha, xabs 
   	
!----------------------------------------------------------------------- 
   	
!
    Noacc=.FALSE. 
    xabs=ABS(X) 
    IF(xabs==0)THEN 
       IF(N==0) THEN 
          Fj=1._dp 
       ELSE 
          Fj=0._dp 
       END IF
    ELSE 
       IF(N>250) THEN 
          print*,"*** ORDER OF BESSELFUNCTION TOO LARGE ***" 
          STOP 
       END IF
       IF(xabs*xabs>=real(N*(N+1))) THEN 
          Fj=SIN(xabs)/xabs 
          IF(N>0) THEN 
             Fj1=Fj 
             Fj=Fj1/xabs-COS(xabs)/xabs 
             IF(N>1) THEN 
                Fj2=Fj 
                DO J=2,N 
                   Fj=real(2*J-1)*Fj2/xabs-Fj1 
                   Fj1=Fj2 
                   Fj2=Fj 
                enddo
             END IF
          END IF
       ELSE 
          Ap=P1 
          B=P35 
          Alog2e=C48 
          U = 2._dp * xabs/real(2*N+1) 
          Nu1 = N + INT(Alog2e*(Ap+B*U*(2._dp-U*U)/(2._dp*(1._dp-U*U)))) 
          Np=INT(xabs-P5+SQRT(B*xabs*Alog2e)) 
          IF(N>Np) THEN 
             U=2._dp*xabs/real(2*NP+1) 
             Nu2=Np+INT(Alog2e*(Ap+B*U*(2._dp-U*U)/(2._dp*(1._dp-U*U)))) 
          ELSE 
             Nu2=100000 
          END IF
          Nu=MIN(Nu1,Nu2,997) 
!         IF(Nu>=997) THEN 
!           WRITE(6,*) '*** AccURACY NOT HIGH ENOUGH ***' 
!           Noacc=.TRUE. 
!         END IF 
          Down=.TRUE. 
          A(Nu+1)=0._dp 
          DO I=Nu,1,-1 
             IF(Down) THEN 
                A(I)=xabs/(real(2*I+1)-xabs*A(I+1)) 
                IF(A(I)>1._dp) THEN 
                   Down=.FALSE. 
                   Lam=I 
                   A(I-1)=1._dp 
                END IF
             ELSE 
                A(I-1)=real(2*I+1)*A(I)/xabs-A(I+1) 
             END IF
          enddo
          IF(Down) THEN 
             A(0)=SIN(xabs)/xabs 
             Lam=0 
          ELSE 
             Alpha=1._dp/((A(0)-xabs*A(1))*COS(xabs)+xabs*A(0)*SIN(xabs)) 
             Mu=MIN(Lam,N)+1 
             DO J=0,Mu-1 
                A(J)=Alpha*A(J) 
             enddo
          END IF
          DO J=Lam+1,N 
             A(J)=A(J)*A(J-1) 
          enddo
          Fj=A(N) 
       END IF
    END IF
!   
    Sbess=Fj 
    RETURN 
  END FUNCTION Sbess

