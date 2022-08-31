
clc; clearvars;
% Iteration Steps
Max_time_step = 50000;
write_interval = 100;
max_iter=Max_time_step/write_interval;
ts=0;

for i=1:max_iter
    h = sprintf('output/i%d.txt',ts);
    d = importdata(h);
    x = d(:,1); % Position of electron 
    v = d(:,2); % Velocity of electron
    
    figure(1)
    histogram(v,100)
    title(ts)    
    pause(0.1)
    
    ts = ts + write_interval; 
end