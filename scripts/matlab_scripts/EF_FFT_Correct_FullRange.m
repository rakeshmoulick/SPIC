
% FFT of Electric Field considering appropriate reshaping.

clc; clearvars ;
format long
load ../results_1024.txt;
d = results_1024; 
NC = 1024; % Total number of cells 
n=NC+1; % Total number of nodal points where electric field data is stored
dt = 3.54518e-12;
write_interval = 10;
NUM_TS = 20000;
max_iter = length(d(:,1))/n;
wpe =  5.64146e+09;
LD = 7.471E-05;

x = d(:,1);
E = d(:,5);
xx = reshape(x,[n,max_iter]);
EF = reshape(E,[n,max_iter]);
%-------------------------------------
E = zeros(max_iter,n);
for i=1:max_iter    
    EP = d((i-1)*n+1:i*n,3);
    E(i,:) = EP;
end
%-------------------------------------
dx = x(2)-x(1);
actual_sim_time = max_iter*dt*write_interval;
actual_sim_len = (NC+1)*dx; % This should be NC*dx. Anyway no significant difference is seen

wlen = linspace(0,max_iter,max_iter); 
klen = linspace(-n,n,n);

w = 2*pi*wlen/(actual_sim_time); % length of max_iter
k = 2*pi*klen/(actual_sim_len); % length of n

[Omega, K] = meshgrid(w,k);
%[K,Omega] = meshgrid(k,w);
disp([size(Omega);size(K)])
%-----------------------------FFT of E-------------------------------------
Field = EF; % Choose Field either as E (manually reshaped) or EF (reshaped)
%--------------------------------------------------------------------------
F = fftn(Field);
% halflen = zeros(floor(size(F)/2));
% s = size(halflen);
% Omega = Omega(1:s(1),1:s(2));
% K = K(1:s(1),1:s(2));
% F = F(1:s(1),1:s(2));
%--------------------------------------------------------------------------
%w = w/wpe;
Omega = Omega/wpe;
K = K*LD;
Z = log(abs(F));

figure(1)
Time = linspace(0,max_iter,max_iter);
Position = d(1:n,1);
subplot(121), contourf(Time, Position, Field,'edgecolor','none')
xlabel('Time'), ylabel('Position'), zlabel('Electric field')
subplot(122), 
surf(Time, Position, Field,'edgecolor','interp')
%surf(Time, Position, Field, 'FaceAlpha',0.5)
xlabel('Time'), ylabel('Position'), zlabel('Electric field')

figure(2)
contourf(K, Omega, Z,'edgecolor','none')
xlabel('k\lambda_{D}'), ylabel('\omega/\omega_{pe}')
%ylim([0 0.5])
