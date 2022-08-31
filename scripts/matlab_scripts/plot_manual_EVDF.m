
clc; clearvars;
% Iteration Steps
Max_time_step = 10000;
write_interval = 100;
max_iter=Max_time_step/write_interval;
ts=0;

for i=1:max_iter
    h = sprintf('output/e%d.txt',ts);
    d = importdata(h);
    x = d(:,1); % Position of electron 
    v = d(:,2); % Velocity of electron
    %disp([min(v),max(v)])
    
    minimum = -0.5E7;
    maximum = +0.5E7;
    nbins = 100;
    delta = (maximum - minimum)/(nbins);
    bin(1:nbins) = 0;
    vrange(1:nbins) = minimum;
    
    for j=1:length(v)
        b = floor((v(j)- minimum)/delta);       
        bin(b) = bin(b) + 1;        
    end
    for k = 2:nbins
        vrange(k) = vrange(k-1) + delta;
    end
    
    figure(1)    
    plot(vrange, bin,'LineWidth',1.5),grid on
    xlabel('v'), ylabel('Count')
    %axis([minimum maximum 0 15000])
    title(ts)
    
    pause(0.1)    
    ts = ts + write_interval; 
end