import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math as math
import scipy.special as sp

#def func(x, a, b, c):
#    return a*np.exp(-b*x) + c

def func(x, sigmag, mu, alpha, c,a):
    normpdf = (1/(sigmag*np.sqrt(2*math.pi)))*np.exp(-(np.power((x-mu),2)/(2*np.power(sigmag,2))))
    normcdf = (0.5*(1+sp.erf((alpha*((x-mu)/sigmag))/(np.sqrt(2)))))
    return 2*a*normpdf*normcdf + c

x = np.linspace(0,100,100)
y = func(x, 10, 30, 0, 0, 1)

yn = y + 0.001*np.random.normal(size=len(x))

popt, pcov = curve_fit(func, x, yn, p0=(1./np.std(yn), np.argmax(yn) ,0,0,1))

y_fit = func(x,popt[0],popt[1],popt[2],popt[3],popt[4])

plt.plot(x, yn)
plt.plot(x, y_fit)
plt.show()
