subroutine thermo_wave_spd_w_Roe_avg(SL, SM, SR, rho_L, u_L, p_L, c_L, rho_R, u_R, p_R, c_R, T_vector,&
                                        betaL_vector, beta_vector, phase_vector, z_vector, x_vector, y_vector, i, NJ)
     use speed_of_sound, only: singlePhaseSpeedOfSound, twoPhaseSpeedOfSound
     ! input
     integer, intent(in) ::  i, NJ      !< index of which cell ur on, total no of cells
     real, intent(inout) ::  SL, SM, SR !< wavespeeds
     real, intent(in)    ::  rho_L, u_L, p_L, c_L
     real, intent(in)    ::  rho_R, u_R, p_R, c_R
     ! thermovars
     real,    dimension(NJ),     intent(in) :: beta_vector, betaL_vector, T_vector
     integer, dimension(NJ),     intent(in) :: phase_vector
     real,    dimension(nc, NJ), intent(in) :: z_vector, x_vector, y_vector
     ! inside vars
     real :: T_L, T_R, beta_L, betaL_L, beta_R, betaL_R, h, v
     integer, dimension(nc) :: phaseL, phaseR, phase_avg
     real, dimension(nc   ) :: z_L, z_R, z_avg, x_L, x_R, y_L, y_R
     real :: u_avg, c_avg, h_avg, h_L, h_R, v_avg, v_L, v_R
     real :: T_avg, p_avg, beta_avg, betaL_avg, molarMass
     real, dimension(2)     :: twoph_beta
     real, dimension(2, nc) :: twoph_X
     integer, dimension(2)  :: twoph_phase
     !print*, "in thermo wavespd roe avg"
     T_L = T_vector(i)
     T_R = T_vector(i+1)
     betaL_L = betaL_vector(i)  
     betaL_R = betaL_vector(i+1)
     beta_L = beta_vector(i)   
     beta_R = beta_vector(i+1)  
     phaseL = phase_vector(i)  
     phaseR = phase_vector(i+1)
     z_L = z_vector(:,i)
     z_R = z_vector(:,i+1)
     x_L = x_vector(:,i)
     x_R = x_vector(:,i+1)
     y_L = y_vector(:,i)
     y_R = y_vector(:,i+1)
 
     u_avg = (sqrt(rho_L)*u_L + sqrt(rho_R)*u_R)/(sqrt(rho_L)+ sqrt(rho_R))
 
     ! works for only 1 component per now, need nested do loop over nc components in z otherwise
     call enthalpy(T=T_L,p=p_L,x=x_L(1),phase=LIQPH,h=h) !thermopack function
     h_L = h*betaL_L
     call enthalpy(T=T_L,p=p_L,x=x_L(1),phase=VAPPH,h=h)
     h_L = h_L + h*beta_L
 
     call enthalpy(T=T_R,p=p_R,x=x_R(1),phase=LIQPH,h=h) !thermopack function
     h_R = h*betaL_R
     call enthalpy(T=T_R,p=p_R,x=x_R(1),phase=VAPPH,h=h)
     h_R = h_R + h*beta_R
     !print*, "called enthalpy functions"
     molarMass = moleWeight(z_L(1))*0.001
     h_avg = (sqrt(rho_L)*h_L/molarMass + sqrt(rho_R)*h_R/molarMass)/(sqrt(rho_L)+ sqrt(rho_R)) ! in mass units
     h_avg = h_avg*molarMass
     v_avg = molarMass/(0.5*(rho_L+rho_R)) ! this might be wrong.. idk
     T_avg = (sqrt(rho_L)*T_L + sqrt(rho_R)*T_R)/(sqrt(rho_L)+ sqrt(rho_R))
     p_avg = (sqrt(rho_L)*p_L + sqrt(rho_R)*p_R)/(sqrt(rho_L)+ sqrt(rho_R))
     beta_avg = (beta_L+beta_R)/2
     betaL_avg = (betaL_L + betaL_R)/2
     z_avg = z_R
     phase_avg = phaseR
     !print*, i
     !print*, molarMass/rho_L, molarMass/rho_R, v_avg
     !print*, h_L, h_R
     call twoPhaseHVsingleComp(t=T_avg,p=p_avg,Z=z_avg,beta=beta_avg,betaL=betaL_avg,X=x_L,Y=y_L,&
          hspec=h_avg,vspec=v_avg, phase=phase_avg(1))
 
     twoph_beta(1) = betaL_avg
     twoph_beta(2) = beta_avg
     twoph_X(1,:) = z_avg(:)
     twoph_X(2,:) = z_avg(:)
     twoph_phase(1) = 1
     twoph_phase(2) = 2
     ! if setning, beta sjekk.
     if ((twoph_beta(1) == 1.0) .or. (twoph_beta(2) == 1.0)) then  !if single phase:
        c_avg = singlePhaseSpeedOfSound(t=T_avg,p=p_avg,Z=z_avg(:),phase=phase_avg(1))
     else   ! else twophase
        c_avg = twoPhaseSpeedOfSound(2,T_avg,p_avg,z_avg,twoph_beta,twoph_X,twoph_phase)
     end if
 
     SL = min(u_L - c_L, u_avg - c_avg)
     SR = max(u_R + c_R, u_avg + c_avg)
     SM = (p_R - p_L + rho_L*u_L*(SL-u_L)- rho_R*u_R*(SR-u_R))/&
          (rho_L*(SL-u_L)-rho_R*(SR-u_R))
     ! this estimate works when called as a function on its own??
     SL = min(u_L - c_L, u_avg - c_avg)
     SR = max(u_R + c_R, u_avg + c_avg)
     SM = (p_R - p_L + rho_L*u_L*(SL-u_L)- rho_R*u_R*(SR-u_R))/&
         (rho_L*(SL-u_L)-rho_R*(SR-u_R))
   end subroutine thermo_wave_spd_w_Roe_avg

   !----------------------------------------------------------------------------
  subroutine thermo_single(T, rho, e, a, mu, h, p, s, Gamma, Cp, c2, kappa,&
   dpdv, dpdt, dmudrho, dmudt)!, dmudz)
! Wrapper routine to simplify calculations for single-phase, single-comp.
!
! INPUT:   T      Temperature  [K]
!          rho    Density  [kg/m3]
!
! OUTPUT:  e      Internal energy  [J/kg]
!          a      Helmholtz free energy  [J/kg]
!          mu     Chemical potential  [J/kg]
!          h      Enthalpy  [J/kg]
!          p      Pressure  [Pa]
!          s      Entropy  [J/kgK]
!          Gamma  Gruneisen coefficient  [-]
!          Cp     Heat cap. under const. press. [J/kgK]
!          c2     Sound velocity squared  [m2/s2]
!          kappa  Ratio of specific heats  [-]
!          dpdv   Pressure differential at const. temperature  [Pa/m3]
!
! GL, 2015-01-28
!--------------------------------------------------------------------------
use eosTV, only: pressure, internal_energy, free_energy, chemical_potential
use eos, only: moleWeight
use numconstants, only: rtol
implicit none
! Transferred variables
real, intent(in) :: T, rho
real, optional, intent(out) :: p, e, a, mu, h, s, dpdv, dpdt
real, optional, intent(out) :: Gamma, Cp, c2, kappa
real, optional, intent(out) :: dmudrho, dmudT!, dmudz
! Locals
real :: p_local, e_local, a_local
real :: kappa_local
real, dimension(1) :: Z
real :: mw, v
real :: mu_loc(1), dmudv_loc(1), dmudT_loc(1)!, dmudz_loc(1,1)
real :: dpdt_local, dpdv_local, dudt
logical :: get_derivs, get_e, get_a, get_p, get_mu, get_s, get_h, &
     get_kappa, get_Gamma, get_Cp, get_c2, get_dpdv
logical :: get_dpdt
logical :: get_dmudrho, get_dmudT, get_muderivs!,get_dmudz
logical :: recalculate
!--------------------------------------------------------------------------
! Physical check
if (rho <= rtol) then
  write (*,*) "Input:   T   =", T
  write (*,*) "         rho =", rho
  call stoperror('noneq_flash: unphysical density (rho<=0).')
end if
if (rho /= rho) call stoperror('thermo_single: rho is NaN.')
!
! Conversion to use Thermopack
Z = [1] ! Composition
mw = moleWeight(Z)*1e-3  ! kg/mol
v = mw/rho  ! m3/mol
! Get all thermodynamical properties
recalculate = .true.
get_e = present(e)
get_a = present(a)
get_p = present(p)
get_dpdv = present(dpdv)
get_dpdt = present(dpdt)
get_mu = present(mu)
get_h = present(h)
get_s = present(s)
get_Gamma = present(Gamma)
get_Cp = present(Cp)
get_c2 = present(c2)
get_kappa = present(kappa)
get_dmudrho = present(dmudrho)
get_dmudT   = present(dmudT)
!get_dmudz   = present(dmudz)
get_derivs = (get_Gamma .or. get_Cp .or. get_c2 .or. get_kappa)
get_muderivs = (get_dmudrho .or. get_dmudT) !.or. get_dmudz)
!
if (get_p .or. get_dpdv .or. get_mu .or. get_h .or. get_derivs ) then
  p_local = pressure(T, v, Z, dpdv=dpdv_local, dpdt=dpdt_local, recalculate=recalculate)
  recalculate = .false.
  if (get_p) then
    p = p_local
  end if
  if (get_dpdv) then
    dpdv = dpdv_local
 end if
 if (get_dpdt) then
    dpdt = dpdt_local
  end if
end if
if (get_e .or. get_s .or. get_h .or. get_derivs ) then
  call internal_energy(T, v, Z, e_local, dudt=dudt, recalculate=recalculate)
  recalculate = .false.
  e_local = e_local/mw  ! J/mol --> J/kg
  if (get_e) then
    e = e_local
  end if
end if
if (get_a .or. get_s .or. get_mu ) then
  call free_energy(T, v, Z, a_local, recalculate=recalculate)
  recalculate = .false.
  a_local = a_local/mw  ! J/mol --> J/kg
  if (get_a) then
    a = a_local
  end if
end if
if (get_mu) then
   call chemical_potential(T, v, Z, mu_loc)
   mu = mu_loc(1)/mw  ! J/mol --> J/kg
   !mu = a_local + p_local/rho  ! Chemical potential [J/kg]
end if
if (get_muderivs) then ! calculate mu twice possibly..
   call chemical_potential(T, v, Z, mu_loc, dmudv=dmudv_loc, dmudt=dmudt_loc) !dmudz=dmudz_loc)
   mu = mu_loc(1)/mw  ! J/mol --> J/kg
   !mu = a_local + p_local/rho  ! Chemical potential [J/kg]
   dmudrho = -(1./rho**2)*dmudv_loc(1)
   dmudT = dmudt_loc(1)
   !dmudz = dmudz_loc(1,1)
end if
if (get_s) then
  s = (e_local - a_local)/T  ! Entropy [J/kgK]
end if
if (get_h) then
  h = e_local + p_local/rho  ! Enthalpy [J/kg]
end if
if (get_derivs) then
  kappa_local = 1.-(dpdt_local**2*T)/(dpdv_local*dudt)  ! Ratio Cp/Cv [-]
  if (get_Gamma) then
    Gamma = v*dpdt_local/dudt  ! Gruneisen coeff. [-]
  end if
  if (get_Cp) then
    Cp = dudt*kappa_local/mw  ! Heat capacity [J/kgK]
  end if
  if (get_c2) then
    c2 = - v**2*dpdv_local*kappa_local/mw  ! Squared sound velocity [m2/s2]
  end if
  if (get_kappa) then
    kappa = kappa_local
  end if
end if
end subroutine thermo_single