
% FFT of Electric Field considering appropriate reshaping.

clc; clearvars ;
format long
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
NC = 1024; % Total number of cells 
n=NC+1; % Total number of nodal points where electric field data is stored
NUM_TS = 20000;
write_interval = 5;

max_iter = NUM_TS/write_interval;
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
eps0 = 8.85E-12;
kb = 1.38E-23;
me = 9.1E-31;
AMU = 1.667E-27;
mi = 40*AMU;
e = 1.6E-19;
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
alp = 1.0;
beta = 0.04;
n0 = 1E8;
vd = 8;

ni0 = (1+alp+beta)*n0;
nec0 = n0;
neh0 = alp*n0;
neb0 = beta*n0;
%--------------------------------------------------------------------------
Tec = 1*e;
Teh = 100*e;
Teb = 100*e;
Ti = 0.026*e;
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
LDC = sqrt(eps0*Tec/(nec0*e^2));
LDH = sqrt(eps0*Teh/(neh0*e^2));

wpi = sqrt(e^2*ni0/(eps0*mi));
wpe = sqrt(e^2*ni0/(eps0*me));
wpec = sqrt(e^2*nec0/(eps0*me));
wpeh = sqrt(e^2*neh0/(eps0*me));
dt = 0.02*(wpe^-1);
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
dx = 1;
actual_sim_time = NUM_TS*dt; %max_iter*dt*write_interval;
actual_sim_len = (NC)*dx; 
%--------------------------------------------------------------------------
%w = 2*pi*(0:max_iter-1)/(actual_sim_time); % length of max_iter

w = 2*pi*(0:NUM_TS)/(actual_sim_time); % length of NUM_TS

% Note: It does not matter what length of w array is chosen, such as
% max_iter or NUM_TS. Because, once we consider the halflen, only halflen 
% values are chosen. There, everything (i.e. the array size) becomes uniform. 
%--------------------------------------------------------------------------
fprintf("min(w):%f and max(w):%f\n",min(w),max(w));

k = 2*pi*(0:n-1)/(actual_sim_len); % length of n
fprintf("max(k):%f\n",max(k));

[Omega, K] = meshgrid(w,k);
fprintf("max(Omega):%f and max(K):%f\n",max(max(Omega)), max(max(K)));
disp(size(Omega))
%disp([size(Omega);size(K)])
%--------------------------------------------------------------------------
% Normalize Omega by wpe
Omega = Omega/wpe;
fprintf("max(Omega):%f\n",max(max(Omega)));
%--------------------------------------------------------------------------
% Calculate the dispersion relation of electron acoustic wave
for i=1:length(k)
    wea(i) = sqrt((wpec^2)*( (1+3*(k(i)*LDC)^2)/(1 + (1/(k(i)*LDH)^2)) ));
    wla(i) = sqrt((wpec^2*(1 + 3*k(i)^2*LDC^2)) + (wpeh^2*(1 + 3*k(i)^2*LDH^2)));
    ud = vd*sqrt(Tec/me);
    wbm(i) = k(i)*ud;
end
%--------------------------------------------------------------------------

figure(2)
plot(k, wea/wpe,'k--','LineWidth',1.5), hold on
plot(k, wla/wpe,'k--','LineWidth',1.5), hold on
plot(k, wbm/wpe,'k--','LineWidth',1.5), hold on
xlabel('k\lambda_{D}'), ylabel('\omega/\omega_{pe}')
ylim([0 3.0])
xlim([0 0.4])

