import numpy as np 
import matplotlib.pyplot as plt
import scipy.integrate as intg
x = np.linspace(0, 10, 100)
#y = np.sin(x)
y = 20
TotalInt = intg.trapz(y,x)
print(TotalInt)

