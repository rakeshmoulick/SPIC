
% FFT of Electric Field considering In-Appropriate reshaping.
clc; clearvars ;
format long
load ../results_1024.txt;
d = results_1024; 
NC = 1024; % Total number of cells 
n=NC+1; % Total number of nodal points where electric field data is stored
dt = 5E-12;
write_interval = 1000;
NUM_TS = 10000;
max_iter = length(d(:,1))/n;
wpec = 1783986365.98;
LD = 7.836e-05;

x = d(:,1);
E = d(:,10);
xx = reshape(x,[max_iter,n]);
EF = reshape(E,[max_iter,n]);
%-------------------------------------
E = zeros(max_iter,n);
for i=1:max_iter    
    EP = d((i-1)*n+1:i*n,10);
    E(i,:) = EP;
end
%-------------------------------------
dx = x(2)-x(1);
actual_sim_time = max_iter*dt*write_interval;
actual_sim_len = (NC+1)*dx; % This should be NC*dx. Anyway no significant difference is seen

w = 2*pi*(0:max_iter-1)/(actual_sim_time); % length of max_iter
k = 2*pi*(0:n-1)/(actual_sim_len); % length of n
[K,Omega] = meshgrid(k,w);
disp([size(Omega);size(K)])
%-----------------------------FFT of E-------------------------------------
Field = EF; % Choose Field either as E (manually reshaped) or EF (reshaped)
%--------------------------------------------------------------------------
F = fftn(Field);
halflen = zeros(floor(size(F)/2));
s = size(halflen);
Omega = Omega(1:s(1),1:s(2));
K = K(1:s(1),1:s(2));
F = F(1:s(1),1:s(2));
%--------------------------------------------------------------------------
w = w/wpec;
Omega = Omega/wpec;
K = K*LD;
Z = log(abs(F));

figure(1)
Time = linspace(0,max_iter,max_iter);
Position = d(1:n,1);
subplot(121), contourf(Time, Position, Field','edgecolor','none')

subplot(122), contourf(K, Omega, Z,'edgecolor','none')
xlabel('k\lambda_{D}'), ylabel('\omega/\omega_{pe}')
ylim([0 1])

