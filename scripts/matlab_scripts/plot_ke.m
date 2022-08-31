
load ../data002/files/ke_1024.txt
ke = ke_1024;

% figure(1)
% subplot(221), plot(ke(:,1),ke(:,2),'r'); grid on
% xlabel('Time'), ylabel('Ion')
% 
% subplot(222), plot(ke(:,1),ke(:,3),'g'); grid on
% xlabel('Time'), ylabel('Cold Electron')
% 
% subplot(223), plot(ke(:,1),ke(:,4),'b'); grid on
% xlabel('Time'), ylabel('Hot Electron')
% 
% subplot(224), plot(ke(:,1),ke(:,2),'k'); grid on
% xlabel('Time'), ylabel('Beam Electron')

figure(2)
ke_tot = ke(:,2) + ke(:,3) + ke(:,4);
plot(ke(:,1), ke_tot, 'LineWidth',2); grid on
xlabel('\omega_{pe}t'), ylabel('Total KE')



