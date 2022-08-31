% ARGON PLASMA (PIC-MCC)
clc; clearvars;
Max_time_step = 100000;
write_interval = 1000;
n=Max_time_step/write_interval;
ts=0;

for i = 1:n 
h1 = sprintf('output/i%d.txt',ts);
d1 = importdata(h1);

h2 = sprintf('output/ec%d.txt',ts);
d2 = importdata(h2);

h3 = sprintf('output/eh%d.txt',ts);
d3 = importdata(h3);

xi = d1(:,1); % Position of positive ion 
vi = d1(:,2); % x-Velocity of positive ion

xe = d2(:,1); % Position of electron
ve = d2(:,2); % x-Velocity of electron

xn = d3(:,1); % Position of negative ion
vn = d3(:,2); % x-Velocity of negative ion

figure(1) 
plot(xi, vi, 'r.','MarkerSize',3), grid on
xlabel('x'), ylabel('v_{i}')
%axis([0 220 -5 5])

figure(2)
plot(xn, vn, 'b.', 'MarkerSize',3), grid on
xlabel('x'), ylabel('v_{n}')
%axis([0 220 -0.5 0.5])

figure(3)
plot(xe, ve, 'k.', 'MarkerSize',3), grid on
xlabel('x'), ylabel('v_{e}')
%axis([0 220 -20 20])
title(ts)

pause(0.1)
%mov(i) = getframe(gcf);
ts = ts + write_interval; 
end
%movie2gif(mov,'phaseSpace.gif','DelayTime',0.01);