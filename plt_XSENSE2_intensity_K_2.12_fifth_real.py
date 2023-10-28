#==============================================================================
# 
# Bessel Factor Comparison to numerical version 
# of Eq. 13 from Dr. Kwange's Undulator Bessel Factor
#   
#==============================================================================

import matplotlib.pyplot as plt
from math import log,pi
from math import pi,e,cos,sin,sqrt,atan,asin,exp,log
from scipy.special import jv
import numpy as np


mc2 = 511e3                # electron mass [eV]
hbar = 6.582e-16           # Planck's constant [eV*s]
c = 2.998e8                # speed of light [m/s]
lambda0 = 800e-9           # same wavelength for each run
alpha = 1.0/137.0          # fine structure constant
r = 2.818e-15              # electron radius [m]
q = 1.602e-19              # electron charge [C]
eV2J = 1.602176565e-19     # eV --> Joules 
J2eV = 6.242e18            # Joules --> eV
hbarJ = 1.0545718e-34      # Planck's constant [J*s]
me = 9.10938356e-31        # electron mass [kg]
eps0 = 8.85e-12            # epsilon 0 (C^2/Nm^2)

#--------------------------------------------------
# eV to Angstrom Conversion - x-axis
#--------------------------------------------------
def ang(m):
    #E=(hc)/(lambda) => lambda=(hc)/E

    lambda_m_num=hbar*2*pi*c
    m_angstrom_conv=10**10

    lambda_meter=np.divide(lambda_m_num,m)
    ang=np.multiply(lambda_meter,m_angstrom_conv)

    return ang

#--------------------------------------------------
# Normalization Function
#--------------------------------------------------
def norm(m):
    maximum = max(m)

    norm= np.divide(m,maximum)

    return norm

if __name__ == '__main__':
    arg_file = open("config.in", "r")
    args = []
    for line in arg_file:
        i = 0
        while (line[i:i + 1] != " "):
            i += 1
        num = float(line[0:i])
        args.append(num)

    #-------------------
    # e-beam parameters
    #-------------------
    En0 = args[0]              # e-beam: mean energy [eV]
    sig_g = args[1]            # e-beam: relative energy spread
    sigma_e_x = args[2]        # e-beam: rms horizontal size [m]
    sigma_e_y = args[3]        # e-beam: rms vertical size [m]
    eps_x_n = args[4]          # e-beam: normalized horizontal emittance [m rad]
    eps_y_n = args[5]          # e-beam: normalized vertical emittance [m rad]

    #------------------
    # Laser parameters
    #------------------
    lambda0 = args[6]          # laser beam: wavelength [m]
    sign = args[7]             # laser beam: normalized sigma [ ]
    sigma_p_x = args[8]        # laser beam: horizontal laser waist size [m]
    sigma_p_y = args[9]        # laser beam: vertical laser waitt size [m]
    a0 = args[10]              # laser beam: field strength a0 []
    iTypeEnv = int(args[11])   # laser beam: laser envelope type
    if (iTypeEnv == 3):        # laser beam: load experimental data -Beth
        data_file = open("Laser_Envelope_Data.txt", "r")
        exp_data = []
        for line in data_file:
           exp_data.append(line.strip().split())
        exp_xi = []
        exp_a = []
        for line in exp_data:
            exp_xi.append(float(line[0]))
            exp_a.append(float(line[1]))
        exp_f=interp1d(exp_xi,exp_a,kind='cubic')       # laser beam: generate beam envelope function
    else:
        exp_xi = []
        exp_f = 0
    modType = int(args[12])    # laser beam: frequency modulation type
    fmd_xi=[]
    fmd_f=[]
    fmdfunc=0
    if (modType == 1):         # exact 1D chirp: TDHK 2014 (f(0)=1)
        a0chirp = a0                # laser beam: a0 chirping value
        fm_param = 0.0
    elif (modType == 2):       # exact 1D chirp: Seipt et al. 2015 (f(+/-inf)=1)
        a0chirp = a0                # laser beam: a0 chirping value
        fm_param = 0.0
    elif (modType == 3):       # RF quadratic chirping
        a0chirp = 0.0
        fm_param = float(args[13])   # laser beam: lambda_RF chirping value
    elif (modType == 4):       # RF sinusoidal chirping
        a0chirp = 0.0
        fm_param = float(args[13])   # laser beam: lambda_RF chirping value
    elif (modType == 5):       # exact 3D chirp: Maroli et al. 2018
        a0chirp = a0                 # laser beam: a0 chirping value
        fm_param = float(args[13])   # p parameter
    elif (modType == 6):       # chirp with ang. dep. (f(0)=1)
        a0chirp = a0                 # laser beam: a0 chirping value
        fm_param = float(args[13])   # theta_FM (optimization angle)
    elif (modType == 7):       # chirp with ang. dep. (f(+/-inf) = 1)
        a0chirp = a0                 # laser beam: a0 chirping value
        fm_param = float(args[13])   # theta_FM (optimization angle)
    elif (modType == 8):       # saw-tooth chirp
        a0chirp = a0                   # laser beam: a0 chirping value
        fm_param = float(args[13])     # chirping slope
    elif (modType == 9):       # read chirping data from a file and generate function -Beth
        data_file = open("Fmod_Data.txt", "r")
        fmod_data = []
        for line in data_file:
           fmod_data.append(line.strip().split())
        fmd_xi = []
        fmd_f = []
        for line in fmod_data:
            fmd_xi.append(float(line[0]))
            fmd_f.append(float(line[1]))
        fmdfunc=interp1d(fmd_xi,fmd_f,kind='cubic')
        a0chirp =  float(args[13])
        lambda_RF = 0.0
    elif (modType == 10):       # chirping from GSU 2013
        a0chirp = a0           
        fm_param = float(args[13])
    else:                      # no chirping
        a0chirp = 0.0
        fm_param = 0.0
    l_angle = args[14]         # laser beam: angle between laser & z-axis [rad]

    #---------------------
    # Aperture parameters
    #---------------------
    TypeAp = args[15]          # aperture: type: =0 circular; =1 rectangular
    L_aper = args[16]          # aperture: distance from IP to aperture [m]
    if (TypeAp == 0):
        R_aper = args[17]      # aperture: physical radius of the aperture [m]
        tmp = args[18]
        theta_max = atan(R_aper/L_aper)
    else:
        x_aper = args[17]
        y_aper = args[18]
 
    #-----------------------
    # Simulation parameters
    #-----------------------
    wtilde_min = args[19]      # simulation: start spectrum [norm. w/w0 units]
    wtilde_max = args[20]      # simulation: end spectrum [norm. w/w0 units]
    Nout = int(args[21])       # simulation: number of points in the spectrum
    Ntot = int(args[22])       # simulation: resolution of the inner intergral
    Npart = int(args[23])      # simulation: number of electron simulated
    N_MC = int(args[24])       # simulation: number of MC samples
    iFile = int(args[25])      # simulation: =1: read ICs from file; <> 1: MC
    iCompton = int(args[26])   # simulation: =1: Compton; <>1: Thomson
    RRmode = int(args[27])     # Radiation reaction model

    #--------------------------------------------
    # Compute basic parameters for the two beams
    #--------------------------------------------
    gamma = En0/mc2
    beta = sqrt(1.0-1.0/(gamma**2))
    c1 = gamma*(1.0 + beta)
    omega0 = 2.0*pi*c1*c1*(c/lambda0)
    ntothm1 = Ntot/2.0 - 1.0
    d_omega = (wtilde_max - wtilde_min)/(Nout-1)

    eps_x = eps_x_n/gamma
    eps_y = eps_y_n/gamma
    sigma_e_px = eps_x/sigma_e_x
    sigma_e_py = eps_y/sigma_e_y
    pmag0 = (eV2J/c)*sqrt(En0**2-mc2**2)
    sigmaPmag = (eV2J/c)*sqrt(((En0*(1.0+sig_g))**2)-mc2**2)-pmag0
    E_laser = 2*pi*hbar*c/lambda0

    omegas_E = []
    Es_E = []
    dNdEs_E = []
    omega_Peaks_E = []
    bessel_Peaks_E = []
    bessel_Peaks_Scaled_E = []
            
    data = open("output_SENSE.txt").readlines()
    data2 = []
    omega_i = []
    E_i = []
    dNdE_i = []
    dNdE_r_i = []
    for line in data:
        data2.append(line.strip().split())
    data = data2
    for line in data:
        if float(line[0]) != 0:
            E_i.append(float(line[0]))
            omega_i.append(float(line[1]))
            dNdE_i.append(float(line[2]))
    omegas_E.append(omega_i)
    dNdEs_E.append(dNdE_i)

    # Divide by area of apertuare in mrad^2

    dNdEdOmega_i=np.divide(dNdE_i,pi*(theta_max*1000)**2)

    # Multiply by E'

    dNdOmega_i = np.multiply(dNdEdOmega_i,E_i) 

    # Multiply by 0.1%BW

    dNdOmega_BW_i = np.multiply(dNdOmega_i,0.001)

    # Multiply to include radiation from negative frequency domains

    dNdOmega_BW_i = np.multiply(dNdOmega_BW_i,2)

    # Multiply by I/e to get photons/sec

    I=0.400

    dNdOmega_BW_i = np.multiply(dNdOmega_BW_i,I/q)

    # Convert eV to Angstrom


    angstrom_i= ang(E_i)


    #--------------------------------------------------
    #   Experimental Data
    #--------------------------------------------------

    #First Harmonic K=0.45
    K_0_45_harm_1x = [30.074,30.235,30.363,30.483,30.566,30.638,30.702,30.778,30.846,30.914,30.957,30.989,31.021,31.045,31.053,31.081,31.097,31.121,31.133,31.161,31.173,31.193,31.217,31.241,31.273,31.305,31.336,31.352,31.376,31.404,31.424,31.448,31.468,31.488,31.528,31.568,31.604,31.64,31.676,31.755,31.863,32.015,32.17,32.298,32.441,32.613,32.721,32.82,32.904,32.958]
    K_0_45_harm_1y = [2043422733077888,2298850574712640,3320561941251584,3831417624521056.5,3831417624521056.5,6385696040868448,7662835249042144,8684546615581088,10217113665389536,20434227330779040,30906768837803330,49042145593869730,62324393358876130,74074074074074080,84546615581098340,97573435504469980,113154533844189010,123116219667943800,140485312899106000,149169859514687100,154789272030651330,159386973180076600,161685823754789280,161685823754789280,158620689655172400,151213282247765000,144316730523627070,133077905491698600,125159642401021710,114942528735632180,103448275862068960,95019157088122610,89399744572158370,73818646232439330,61302681992337150,48020434227330780,40102171136653890,36015325670498080,28607918263090690,22733077905491716,14559386973180064,8939974457215840,6130268199233729,4853128991060031,4086845466155807.5,3831417624521056.5,3575989782886336,3065134099616864.5,3575989782886336,3575989782886336]

    K_0_45_norm_harm_1y= norm(K_0_45_harm_1y)


    #Third Harmonic K=0.45
    K_0_45_harm_2x = [10.244,10.273,10.306,10.331,10.347,10.359,10.362,10.373,10.388,10.404,10.42,10.43,10.442,10.456,10.474,10.486,10.523,10.561,10.626,10.699,10.748,10.783]
    K_0_45_harm_2y = [85227272727272,127840909090908.98,227272727272727,397727272727272,752840909090909,1420454545454545,2031250000000000,2784090909090909,3409090909090909,3892045454545454.5,3366477272727272.5,2755681818181818,2045454545454545,1406249999999999.5,1193181818181818,838068181818182,411931818181818,213068181818182.03,127840909090908.98,85227272727272,42613636363636,14204545454545]

    K_0_45_norm_harm_2y= norm(K_0_45_harm_2y)


    #First Harmonic K=2.12
    K_2_12_harm_1x = [90.251,90.602,90.843,91.084,91.385,91.625,91.826,92.087,92.187,92.488,92.528,92.649,92.749,92.829,92.89,92.97,93.03,93.09,93.211,93.311,93.391,93.492,93.552,93.652,93.773,93.913,94.094,94.294,94.535,94.736,94.896,95.077,95.278,95.478,95.699,95.809]
    K_2_12_harm_1y = [4878048780487808,6097560975609793,12195121951219520,17073170731707328,17073170731707328,19512195121951230,40243902439024380,95121951219512200,156097560975609800,259756097560975650,290243902439024400,300000000000000000,303658536585365900,296341463414634200,285365853658536600,264634146341463420,241463414634146370,212195121951219520,182926829268292700,146341463414634180,113414634146341500,89024390243902450,70731707317073220,50000000000000000,29268292682926852,20731707317073216,20731707317073216,17073170731707328,14634146341463426,10975609756097600,4878048780487808,7317073170731713,8536585365853696,8536585365853696,4878048780487808,3658536585365888]

    K_2_12_norm_harm_1y= norm(K_2_12_harm_1y)


    #Third Harmonic K=2.12
    K_2_12_harm_2x = [30.412,30.439,30.474,30.517,30.558,30.589,30.624,30.658,30.699,30.723,30.738,30.745,30.773,30.816,30.838,30.847,30.855,30.866,30.879,30.886,30.897,30.903,30.916,30.925,30.945,30.962,30.979,30.996,31.01,31.042,31.073,31.099,31.127,31.161,31.196,31.239,31.281,31.324,31.367,31.388]
    K_2_12_harm_2y = [6861063464837057,4802744425385984,6861063464837057,8919382504288193,8233276157804481,10977701543739328,14408233276157826,19210977701543744,28816466552315652,41166380789022340,50085763293310470,67238421955403140,125557461406518020,238765008576329380,298456260720411700,306689536878216200,313550600343053200,314922813036020600,311492281303602100,303259005145797600,295025728987993200,278559176672384260,258662092624356800,233962264150943420,201029159519725600,175643224699828500,146140651801029200,124871355060034300,102915951972555790,72041166380789060,56946826758147520,41852487135506050,32933104631217856,27444253859348228,19897084048027456,17152658662092674,13036020583190402,10977701543739328,9605488850771904,8919382504288193]

    K_2_12_norm_harm_2y= norm(K_2_12_harm_2y)


    #Fifth Harmonic K=2.12

    K_2_12_harm_3x = [18.343,18.351,18.359,18.362,18.369,18.374,18.38,18.386,18.39,18.4,18.409,18.413,18.42,18.427,18.439,18.447,18.452,18.461,18.467,18.477,18.484,18.49,18.496,18.5,18.503,18.508,18.51,18.514,18.518,18.522,18.527,18.529,18.535,18.54,18.545,18.549,18.551,18.556,18.558,18.564,18.568,18.572,18.579,18.58,18.586,18.592,18.598,18.606,18.61,18.617,18.625,18.637,18.643,18.648,18.653,18.659,18.666,18.673,18.679,18.683,18.691,18.695,18.698]
    K_2_12_harm_3y = [7075471698113184,7547169811320736,7547169811320736,12735849056603744,9433962264150914,10377358490566018,3773584905660352,7075471698113184,5188679245283009,10377358490566018,12264150943396192,15566037735849024,19339622641509410,26415094339622624,38679245283018850,54245283018867900,66981132075471680,85849056603773570,110377358490566020,140094339622641490,172169811320754700,180660377358490560,186792452830188670,194811320754716960,192924528301886800,194811320754716960,197169811320754700,191037735849056580,183018867924528300,174056603773584900,170283018867924500,159433962264150900,147169811320754700,131132075471698080,118396226415094300,108962264150943360,98113207547169810,94811320754716960,86792452830188670,78773584905660350,68867924528301860,58962264150943360,58962264150943360,53301886792452800,53301886792452800,45754716981132060,35849056603773570,34433962264150910,26886792452830176,21226415094339584,17924528301886752,15566037735849024,16037735849056576,15094339622641472,12735849056603744,10377358490566018,8018867924528288,9905660377358462,8018867924528288,10377358490566018,6132075471698080,4716981132075457,7075471698113184]

    K_2_12_norm_harm_3y= norm(K_2_12_harm_3y)


    color = '#0b5509'
    color2 = '#00b3ff'

    dNdOmega_BW_norm_i=norm(dNdOmega_BW_i)

    #plt.plot(angstrom_i, dNdOmega_BW_norm_i, '-',  color = color, label=r'output_XSENSE2') 
    plt.title("Fifth Harm. K=%s, I=%s mA" %(a0,I))
    plt.plot(angstrom_i, dNdOmega_BW_i, '-',  color = color, label=r'output_XSENSE2') 
    #plt.plot(K_2_12_harm_1x, K_2_12_norm_harm_1y, color = color2, label=r'exp. data')
    plt.plot(K_2_12_harm_3x, K_2_12_harm_3y, color = color2, label=r'exp. data')
    #plt.set_yscale('log')
    plt.xlabel('Photon Wavelength  (Angstrom)')
    plt.ylabel("dN/dΩ [photons/(s mrad^2 0.1%BW)]")
    plt.xlim(18.3, 18.8)
    plt.legend()
    plt.savefig('plt_XSENSE2_intensity_K_2.12_real_fifth.eps', format='eps', dpi=2000)
    plt.show()
