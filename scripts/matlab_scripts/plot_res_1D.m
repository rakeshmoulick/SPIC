% ARGON PLASMA (PIC-MCC)
clc; clearvars ;
format long
eps = 8.85E-12;
eV = 1.6E-19;
AMU = 1.66E-27;
Te = 1*eV;
mi = 40*AMU;
me = 9.1E-31;
cs = sqrt(Te/mi);

NC = 1024; 
load ../RUN-1/results_1024.txt;
d = results_1024; 
n=NC+1;
DT = 5E-12;
Time = 0;
n0 = 1E13;
LD = sqrt(eps*Te/(n0*eV^2));
write_interval = 1000;

max_iter = length(d(:,1))/n;
for i=1:max_iter %41 = 20 ns, 61 = 30 ns    
    x=d((i-1)*n+1:i*n,1);
    ndi=d((i-1)*n+1:i*n,2);    
    nde=d((i-1)*n+1:i*n,3);    
    ndn=d((i-1)*n+1:i*n,4);

    veli = d((i-1)*n+1:i*n,5);   
    veli = veli/cs;        

    vele = d((i-1)*n+1:i*n,6); 
    veln = d((i-1)*n+1:i*n,7);
    % Store wall data   
    rho = d((i-1)*n+1:i*n,8);
    phi=d((i-1)*n+1:i*n,9);
    EF = d((i-1)*n+1:i*n,10);        

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
    figure(1)
    subplot(221), 
    plot(x, phi,'linewidth',2),grid on 
    xlabel('x/\lambda_{D}'),ylabel('Electric Potential (eV)')
    
    h = sprintf('Time = %0.3g(micro sec), ts = %d',Time*1E6, i);
    title(h);
    
    subplot(222), 
    plot(x, EF,'linewidth',2),grid on                        
    xlabel('x/\lambda_{D}'),ylabel('Electric Field')
    legend('EF','location','southwest')
    
    subplot(223), 
    %plot(x, ndi,'r','linewidth',2),grid on
    plot(x, nde, 'g','linewidth',2),grid on
    %plot(x, ndn, 'b','linewidth',2),grid on
    xlabel('x/\lambda_{D}'),ylabel('Densities')
    
    %legend('ndi','nde','location','southeast')
   
    subplot(224), 
    plot(x,veln, 'b', 'linewidth',2),grid on 

    %axis([0 30 -0.2 0.2])
    xlabel('x/\lambda_{D}'), ylabel('Normalized Velocity')    
    legend('v_{i}','location','southwest')
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
        
%    mov(i) = getframe(gcf);    
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
pause(0.1) ;    
Time = Time + write_interval*DT;   
end
%movie2gif(mov, 'plot2.gif','DelayTime',0.5,'LoopCount',5)

