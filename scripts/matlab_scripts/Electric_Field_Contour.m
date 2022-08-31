
% ARGON PLASMA (PIC-MCC)
clc; clearvars ;

load results_1024.txt;
d = results_1024; 
NC = 1024; 
n=NC+1;

max_iter = length(d(:,1))/n;
T = linspace(0,max_iter,max_iter);
X=d(1:n,1);
E = zeros(max_iter,n);

for i=1:max_iter    
    EF = d((i-1)*n+1:i*n,10);
    E(i,:) = EF;
end

figure(1)
subplot(121), surf(T,X,E','edgecolor','none');
xlabel('Time'), ylabel('Position'), zlabel('E')

subplot(122), contourf(T,X,E','edgecolor','none');
xlabel('Time'), ylabel('Position')

