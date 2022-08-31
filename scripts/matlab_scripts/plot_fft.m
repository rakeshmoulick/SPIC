
load potloc_1024.txt
p = potloc_1024;
t = p(:,1);
y = p(:,2);

eps = 8.85E-12;
n0 = 1E13;
me = 9.1E-31;
e = 1.6E-19;
eV  = 1.6E-19;
Te = 1*eV;

LD = sqrt(eps*Te/(n0*e^2));
wp = sqrt((n0*e^2)/(eps*me));
E_norm = (Te/(e*LD)); 
T = t(end);
Fs = 1/T;

figure(1)
%subplot(121)
plot(wp*t,y,'r'),grid on
xlabel('\omega_{p}t'), ylabel('Eletcric Field')

% subplot(122)
% psdfft(hamming(length(y/E_norm)).*detrend(y/E_norm),Fs);
% set(gca,'xscale','log','yscale','log')
%psdfft(y,Fs);

figure(2)
%pspectrum(y/E_norm)
[pxx,f] = pspectrum(y/E_norm, t);
semilogy(f,pow2db(pxx))
%semilogy((wp./f), pxx)
%axis([0 30 min(pxx) max(pxx)])
grid on
xlabel('Frequency (Hz)')
ylabel('Power Spectrum (dB)')


