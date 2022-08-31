% Time Averaged Plot
% ARGON PLASMA (PIC-MCC)
clc; clearvars ;
format long
load results_1024.txt;
d = results_1024; 

% Universal Constants
eps = 8.85E-12;
e   = 1.6E-19;
eV  = 1.6E-19;
AMU = 1.66E-27;
% Simulation Parameters
n0 = 1E16;
Te  = 1*eV;
mi  = 40*AMU;
me  = 9.1E-31;
cs  = sqrt(Te/mi);
LD = sqrt(eps*Te/(n0*e^2));
wp = sqrt((n0*e^2)/(eps*me));
% Simulation Domain Parameters
NC  = 1024; 
n  = NC+1;
DT = 5E-12;
write_interval = 100;
% Maximum Iteration Steps
max_iter = length(d(:,1))/n;
%max_iter = 90;

% Initiate the Average Counters
x_avg = zeros(n,1);
ndi_avg = zeros(n,1);
nde_avg = zeros(n,1);
velix_avg = zeros(n,1);
phi_avg = zeros(n,1);
EF_avg = zeros(n,1);

% Loop Over 
for i=1: max_iter
x=d((i-1)*n+1:i*n,1);
ndi=d((i-1)*n+1:i*n,2);    
nde=d((i-1)*n+1:i*n,3);    

veli = d((i-1)*n+1:i*n,4);   
veli = veli*(wp*LD/cs);        

vele = d((i-1)*n+1:i*n,5); 

% Store wall data   
rho = d((i-1)*n+1:i*n,6);
phi=d((i-1)*n+1:i*n,7);
EF = d((i-1)*n+1:i*n,8);   

% Summing up the averages
x_avg = x_avg+x;
ndi_avg = ndi_avg + ndi;
nde_avg = nde_avg + nde;
velix_avg = velix_avg + veli;
phi_avg = phi_avg + phi;
EF_avg = EF_avg + EF;
end

% Time Averaged Quantities
x_avg = x_avg/max_iter;
ndi_avg = ndi_avg/max_iter;
nde_avg = nde_avg/max_iter;
velix_avg = velix_avg/max_iter;
phi_avg = phi_avg/max_iter;
EF_avg = EF_avg/max_iter;

%+++++++++++++++++++++++++++++ Plotting +++++++++++++++++++++++++++++++++++    
figure(1)
subplot(221), plot(x_avg/LD, phi_avg,'linewidth',2),grid on 
xlabel('x/\lambda_{D}'),ylabel('Electric Potential (V)')

subplot(222), 
plot(x_avg/LD, EF_avg,'linewidth',2),grid on                        
xlabel('x/\lambda_{D}'),ylabel('Electric Field')
legend('EF','location','southwest')

subplot(223), plot(x_avg/LD, ndi_avg, x_avg/LD, nde_avg,'linewidth',2),grid on
xlabel('x/\lambda_{D}'),ylabel('Densities')
legend('ndi','nde','location','southeast')

subplot(224), plot(x_avg/LD,velix_avg, 'b', 'linewidth',2),grid on 
%axis([0 30 -1.5 1.5])
xlabel('x/\lambda_{D}'), ylabel('Normalized Velocity')    
legend('v_{i}','location','southwest')




