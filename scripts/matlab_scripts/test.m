% Dispersion Test
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
NC = 1024; % Total number of cells 
n=NC+1; % Total number of nodal points where electric field data is stored
NUM_TS = 20000;
write_interval = 5;
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
n0 = 1E10;
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
dx = 1.0;
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
actual_sim_time = NUM_TS*dt; %max_iter*dt*write_interval;
actual_sim_len = (NC)*dx; 
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
w = 2*pi*(0:NUM_TS)/(actual_sim_time); % length of NUM_TS
k = 2*pi*(0:n-1)/(actual_sim_len); % length of n
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for i=1:length(k)
    wea(i) = sqrt((wpec^2)*( (1+3*(k(i)*LDC)^2)/(1 + (1/(k(i)*LDH)^2)) ));
    wla(i) = sqrt((wpec^2*(1 + 3*k(i)^2*LDC^2)) + (wpeh^2*(1 + 3*k(i)^2*LDH^2)));
end
figure(1)
plot(k, wea/wpe,'b-.','LineWidth',2), hold on
%xlim([0 0.4])

figure(2)
plot(k, wla/wpe,'b-.','LineWidth',2), hold on
%xlim([0 0.4])
%ylim([0 3.0])