import numpy as np 
import matplotlib.pyplot as plt
squareimpulse = np.array([0,0,0,0,0,1,1,1,1,1,0,0,0,0,0])

img = (squareimpulse)
f = np.fft.fft(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = (np.abs(fshift))

plt.subplot(121)
plt.plot(img)
plt.title('Input Image')
plt.xticks([]), plt.yticks([])
plt.grid(True)

plt.subplot(122)
plt.plot(magnitude_spectrum)
plt.title('Magnitude Spectrum')
plt.xticks([]), plt.yticks([])
plt.grid(True)

plt.show()