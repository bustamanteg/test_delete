#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:26:54 2021
Last modification 20220725
%20220731 

I want to make a plot of the noise  in fm/sqrt(Hz)
compare the sensitivities



%20220907
Plot of the noise in fm/sqrt(Hz)

To do: verify that the amplifier amplifies the data by this number. 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv 
from scipy.optimize import curve_fit
import datetime
import os

kb = 1.3806488e-23 #Boltzmann constant (m^2kgs^-2 K-1) 
p = 2328 #silicon density kg/m^3
L = 228e-6 # Cantilever length (m)
W = 39e-6 # Cantilever width (m)
H = 7.3e-6 # Cantilever thickness (m)



# Read in the MFLI data
def Asd(file, centerfreq, xwidth = 5000, vacuum_air = 'vacuum', T = 293,\
        length = 228e-6, width = 38e-6, height =  7e-6, sens = 1 ,preamp_gain=1):

    dat = open(file, 'r').read().splitlines()
    time = dat[2][8:]
    print('Date/time of aquisition is',format(time))
    dat = dat[6:]
    d = []
    for i in range(len(dat)):
        d.append(dat[i].split(';'))
    d = np.asarray(d,dtype=str)
    d = d[:,0:2]
    d = np.asarray(d, dtype=np.float64)
    freqM = d[:,0]
    asdM = np.sqrt((d[:,1]/preamp_gain)**2/(freqM[1]-freqM[0])) # For PSD,square amp and divide by freq res sqrt

    # Define our fitting function for peaks
    #
    def Lorentz(f, Fd_k, f0, Q, base):
        #Fth = Fth/k
        w = f*2*np.pi
        w0 = f0*2*np.pi
        return (Fd_k)/np.sqrt((1-(w/w0)**2)**2+(w/w0/Q)**2)+base
    
    
    # Define our fitting function for in air AFM. Using cantilever params and
    # the Q and f0 form our lorentz fit, will calculate a spring const.
    def Sader_normalk(L, b, f0, Q):
        #L = 197e-6 # length [m]
        #b = 29e-6 # width [m]
    
        #f0 = 69.87e3 # resonance frequency [Hz]
        #   Q = 136
        # physical constant
        rho = 1.18  # mass density of air [kg/m^3]
        eta = 1.86e-5 # viscosiy of air kg/(mxs)=Pa.s 
    
        w0=2*np.pi*f0
        # Reynold's number
        Ren = lambda w, b: rho* w * b**2 /(4*eta)
    
        omre = lambda x: (0.91324 - 0.48274*x + 0.46842*x**2 - 0.12886*x**3 + \
        0.044055*x**4 - 0.0035117*x**5 + 0.00069085*x**6)/\
        (1 - 0.56964*x + 0.48690*x**2 - 0.13444*x**3 + 0.045155*x**4 - \
        0.0035862*x**5 + 0.00069085*x**6)
    
        omimag = lambda x: (-0.024134- 0.029256*x + 0.016294* x**2 - 0.00010961*x**3 + \
         0.000064577*x**4 - 0.000044510*x**5) / (1 - 0.59702*x + 0.55182*x**2 - \
         0.18357*x**3 + 0.079156*x**4 - 0.014369*x**5 + 0.0028361*x**6)
    
        om = lambda x:omre(x) +1j*omimag(x)
        omega = lambda x: om(np.log10(x))
    
        Gamma =lambda w, b: omega(Ren(w, b)) * \
            (1 + 4*1j*kv(1, -1j* np.sqrt(1j* Ren(w, b)))/\
            ( np.sqrt(1j* Ren(w, b)) * kv(0, -1j* np.sqrt(1j* Ren(w, b)))))
    
        kn = 0.1906 * rho * b**2 * L*Q * np.imag(Gamma(w0, b))*(w0)**2
        return kn
    
    def Li_normalk(L,b,H,f0):
        kn = 0.2427*p*L*b*H*(2*np.pi*f0)**2
        return kn
    
    # This peak will
    def peakfit(freq, psd):
        
        deltaf = freq[1]-freq[0]
        init_Fd_k = 1e-9
        init_f0 = centerfreq
        init_Q = 20000
        init_base = 1e-6
        initial_guess = (init_Fd_k, init_f0, init_Q, init_base)
        
        cond = np.logical_and(freq > centerfreq - xwidth/2, freq < centerfreq + xwidth/2)
        freq_fit = np.extract(cond, freq)
        data_fit = np.extract(cond, psd) # for better accuracy in fitting
        
        #param_bounds=([0,0,50,0],[1,165000,50000,1e-3])
        popt, pcov = curve_fit(Lorentz, freq_fit, data_fit, p0=initial_guess, maxfev = 10000)
        print(popt)
        
        [Fd_k, f0, Q, base] = popt
        #popt = [1e-9, 156530, 20000, 2.4e-6]
        #[Fd_k, f0, Q, base] = [1e-7, 156530, 20000, 2.4e-6]
        
        fitted_data = Lorentz(freq_fit, *popt)
        
        if vacuum_air == 'vacuum':
            kn1 = Li_normalk(length, width, height, f0)
            
        elif vacuum_air == 'air':
            kn1 = Sader_normalk(length, width, f0, Q)
        
        else:
            print('invalid vacuum_air arg')
            quit()    
        
        
        cond = np.logical_or(np.logical_and(freq > f0 - xwidth, freqM < f0 - xwidth/2),\
                             np.logical_and(freq < f0 + xwidth, freqM > f0 + xwidth/2))
        noise_dat = np.extract(cond, psd)
        base_away = np.mean(noise_dat)
        
        

        Fth_k = np.sqrt((4*kb*T)/(Q*2*np.pi*f0*kn1))

        
        print('Fth_k',Fth_k)
        Th_peak = Lorentz(f0, Fth_k, f0, Q, 0)  * 1e15 # Expected thermal noise peak [fm/sqrtHz]
        Re_peak_V = Lorentz(f0, Fd_k, f0, Q, base)  # Calculated noise peak (V/sqrt(Hz))
        Re_peak_fm = Re_peak_V * (1/sens) * 1e6 # Calculated noise peak (fm/sqrt(Hz))
        Re_peak_base = base * (1/sens) * 1e6
        print(Re_peak_base)
        
        
        today = datetime.date.today()
        print('analysis date:',today)
        print('\nfitting results')
        print('Frequency resolution = ' + str(deltaf) + '(Hz)')
        print('Resonance frequency = ' + str(f0) + '(Hz)')
        print('Q factor = ' + str(Q))
        print('kc= ' + str(kn1) + ' [N/m]')
        print('Sensitivity = ' + str(sens*1e3) + ' [mV/nm]')
        print('\n')
        
        print('ASD Amplitude V = ' + str(Re_peak_V) + r'(Vrms/sqrtHz)')
        print('ASD Amplitude m = ' + str(Re_peak_fm) + r'(fm/sqrtHz)')
        print('Expected Thermal ASD Amplitude m = ' + str(Th_peak) + r'(fm/sqrtHz)')
        print('\n')
        
        print('Baseline noise (from fit) = ' + str(base) + '(Vrms/sqrtHz)')
        print('Baseline noise/sens ='+str(base/sens*1e6)+'fm_rms/sqrt(Hz)')


        print('Signal to noise = ' + str(Re_peak_V/base))
        print('\n')
        
        text_file = open(filename[:-4]+'_params.txt', "w")
        text_file.write('script inputs are: ' + file+', ' + str(centerfreq)+', ' + str(xwidth)+', ' + vacuum_air+', ' +  str(T)+', '+ str(length)+', ' + str(width)+', ' +str(height)+', ' +str(sens)+'\n')
        text_file.write('analysis date:'+str(today)+'\n')
        text_file.write('fitting results'+'\n')
        text_file.write('Frequency resolution = ' + str(deltaf) + '(Hz)'+'\n')
        text_file.write('Resonance frequency = ' + str(f0) + '(Hz)'+'\n')
        text_file.write('Q factor = ' + str(Q)+'\n')
        text_file.write('kc= ' + str(kn1) + ' [N/m]'+'\n')
        text_file.write('Sensitivity = ' + str(sens*1e3) + ' [mV/nm]'+'\n')
        text_file.write('\n')
        
        text_file.write('ASD Amplitude V = ' + str(Re_peak_V) + r'(Vrms/sqrtHz)'+'\n')
        text_file.write('ASD Amplitude m = ' + str(Re_peak_fm) + r'(fm/sqrtHz)'+'\n')
        text_file.write('Expected Thermal ASD Amplitude m = ' + str(Th_peak) + r'(fm/sqrtHz)'+'\n')
        text_file.write('\n')
        
        text_file.write('Baseline noise (from fit) = ' + str(base) + '(Vrms/sqrtHz)'+'\n')
        text_file.write('Baseline noise/sens ='+str(base/sens*1e6)+'fm_rms/sqrt(Hz)')
        text_file.write('Signal to noise = ' + str(Re_peak_V/base))
        text_file.close()
        
        
        return (freq_fit, fitted_data, base_away, Q, data_fit)
        
    (freq_fitM, fitted_dataM, base_away, Q, raw_data_fit) = peakfit(freqM, asdM)
    
    plt.figure(1, figsize=(8,4))
    plt.semilogy(freqM,asdM/sens*1e6,'r',alpha=.8, label = 'MFLI data')
    plt.semilogy(freq_fitM,fitted_dataM/sens*1e6,'b')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('displacement spectral density (fm/sqrt(Hz))')
    #plt.xlim([0,500000])
    plt.savefig(str(file)[:-4]+'full.png',format='png')
    
    
    
    plt.figure(2, figsize=(8,4))
    plt.semilogy(freqM,asdM/sens*1e6,'r.', label = 'MFLI data')
    plt.semilogy(freq_fitM,fitted_dataM/sens*1e6,'g', linewidth = 2, label = 'fitted MFLI data')
    #plt.semilogy([centerfreq-xwidth/2,centerfreq+xwidth/2],[base_away,base_away], 'b', label = 'Noise away from peak')
    plt.legend()
    plt.xlim([centerfreq-xwidth/2,centerfreq+xwidth/2])
    plt.ylim([np.min(fitted_dataM/sens*1e6)*0.5, np.max(fitted_dataM/sens*1e6)*1.5])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('displacement spectral density (fm/sqrt(Hz))')
    plt.title(filename+format(time))
    plt.text(centerfreq,np.min(fitted_dataM), 'Q factor is '+"{:.0f}".format(Q))
    plt.savefig(str(file)[:-4]+'peak.png',format='png')
    plt.draw()
    plt.show()
    
    plt.figure(3, figsize=(8,3))
    plt.title('Residuals')
    plt.plot(freq_fitM,(raw_data_fit-fitted_dataM)/max(fitted_dataM),'k.')
    plt.plot([centerfreq-xwidth/2,centerfreq+xwidth/2],[0,0],'k--')
    plt.xlim([centerfreq-xwidth/2,centerfreq+xwidth/2])
    plt.savefig(str(file)[:-4]+'residuals.png',format='png')

filename = 'noise_spectra77K_far_away_S0.06V_nm_preampgain1_meas_grid_20221012_141801.txt'
foldername = '/data/20221012_run17'
os.chdir(''+foldername)    

Asd(filename, 172943, xwidth = 3000, vacuum_air = 'vacuum', T = 77,sens=0.06,preamp_gain=1)
