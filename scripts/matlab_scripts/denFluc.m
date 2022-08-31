
clc; clearvars;

% Simulation Parameters
eps = 8.85E-12;
n0 = 1E16;
me = 9.1E-31;
e = 1.6E-19;
eV  = 1.6E-19;
Te = 1*eV;

LD = sqrt(eps*Te/(n0*e^2));
wp = sqrt((n0*e^2)/(eps*me));

% Load Files and Plot Ion Density Fluctuation Data
%load denlocE_1024.txt
%k = denlocE_1024;

h1 = sprintf('../denlocE_1024.txt');
k = importdata(h1);

plot(wp*k(:,1),abs(k(:,2)),'linewidth',1.5),grid on, hold on
xlabel('\omega_{p}t'), ylabel('Fluctuations')


% Load Files and Plot Electron Density Fluctuation Data
% load denlocI_1024.txt
% d = denlocI_1024;

h2 = sprintf('../denlocI_1024.txt');
d = importdata(h2);

plot(wp*k(:,1),abs(k(:,2)),'linewidth',2),grid on, hold on
%plot(wp*d(:,1),abs(d(:,2)),'linewidth',2),grid on, hold on

%legend('\Delta n_{e}','\Delta n_{i}','location','northwest')

