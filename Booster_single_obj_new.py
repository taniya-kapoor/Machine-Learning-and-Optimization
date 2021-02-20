# -*- coding: utf-8 -*-
"""
Created on Mon Feb 05 08:35:10 2018

@author: ahebbal
"""

# OBJET :   Calcule de la poussee et du GLOW 
#           d'un propulseur a poudre      
#
# USAGE :   [Obj,Cons] = Booster(X_)
#
# ENTREE : 
#  X_(1) : Dc : diametre au col de tuyere (en m) -- normalise 0 < X_(1) < 1 -- denormalise -- 0.05 < Dc < 1
#  X_(2) : Ds : diametre du sortie de tuyere (en m) -- normalise 0 < X_(2) < 1 -- denormalise -- 0.5 < Ds < 1.2
#  X_(3) : Pc : pression de combustion (en bars) -- normalise 0 < X_(3) < 1 -- denormalise -- 1 < Pc < 500
#  X_(4) : mp : masse de poudre embarquée (en kg) -- normalise 0 < X_(4) < 1 -- denormalise -- 2000 < mp < 15000
# 
# SORTIE :  
#  Objs  : Obj1 = - Pousse (N), Obj2 = GLOW (kg)
#  Cons  : vecteur des contraintes (dimension = 8) (G(X,Z)<=0 faisable)

import numpy as np
from scipy.optimize import fsolve

from scipy.integrate import quad
from numpy.linalg import solve
import math as m
import cma
import matplotlib.pyplot as plt
from numpy.matlib import rand,zeros,ones,empty,eye

def nari(Gama,Ae_At):
    
    lb = 1.8
    up = 10000000
    x0 = 2.
    f = lambda x : fuser(x,Gama,Ae_At)
    Pc_Pe = fsolve(f,x0)

    return Pc_Pe

def fuser(Pc_Pe,Gama,Ae_At):
    
    y = nar(Gama,Pc_Pe)-Ae_At
    
    return y

def nar(Gama, Pc_Pe):
    
    k1 = 1./(Gama - 1.)
    k2 = 1./Gama
    k3 = (Gama- 1.)/Gama
    Ae_At = ((2./(Gama + 1.))**k1 * Pc_Pe**k2)/np.sqrt(((Gama + 1.)/(Gama-1.))*(1.-(1./Pc_Pe)**k3))
    
    return Ae_At

def cf(Lambda,Gama,Pc_Pe,Pc_Pa,Ae_At):
    
    pe_pc = 1. / Pc_Pe
    pa_pc = 1. / Pc_Pa

    k1 = (Gama - 1.) / Gama
    k2 = (Gama + 1.) / 2. / (Gama - 1.)
    Ct = Lambda * Gama * np.sqrt( (2. / (Gama - 1.)) * (1. - pe_pc ** k1)) /((Gama + 1.) / 2.) ** k2 + Ae_At * (pe_pc - pa_pc)
    
    return Ct

def Poussee(X,parametres):
    
    # ENTREE :
    #  Dc : diamètre au col de tuyère (en m)
    #  Ds : diamètre du sortie de tuyère (en m)
    #  Pc : pression de combustion (en bars)
    #  Pa : pression ambiante de dimensionnement (en bars)
    #  

    # SORTIE :  
    #  S(1)       : Poussée (N)
    #  S(2)       : Indicateur Pc/Pa
    #  S(3)       : Indicateur As/Ac
    #  S(4)       : Indicateur Ps/Pc
    #  S(5)       : Indicateur Cf 
    
    
    Dc = X[0]
    Ds = X[1]
    Pa = X[2]
    Pc = X[3]
    
    Vref = parametres[0]
    Pref = parametres[1]
    n_ = parametres[2]
    c_etoile = parametres[3]
    ga = parametres[4]
    rhop = parametres[5]
    
    Ac = np.pi*Dc**2./4.
    As = np.pi*Ds**2./4.
    epsilon = As/Ac
    
    a_ = Vref*Pref**(-n_)
    exposant1 = 1./(1.-n_)
    a2 = (a_*(10.**(-5.))**n_)/1000.
    Kn = (Pc*100000.)**(1./exposant1)/(rhop*a2*c_etoile)
    
    Cf = 0.
    if (Pc > 1.1*Pa):
        if(As/Ac < 1.1):
            Ps = 1.01*Pc
        else:
            Ps = Pc/nari(ga,As/Ac)    
    
        if (Ps > 0.9*Pc):
            F = 0.
        else:
            Cf = cf(1,ga,Pc/Ps,Pc/Pa,As/Ac)
        
            if (Cf > 0.):
                F = 0.98*Cf*Pc*Ac*100000.
            else:
                F = 0.
    
    else:
        F = 0.
        Ps = Pa
#    Ps = Pc/nari(ga,As/Ac)
#    Cf = cf(1,ga,Pc/Ps,Pc/Pa,As/Ac)
#    F = 0.98*Cf*Pc*Ac*100000.
#    
#    if (Pc > 1.1*Pa):
#            if(As/Ac < 1.1):
#                Ps = 1.01*Pc 
    dc1 = (1.1*Pa - Pc)/Pa
    dc2 = 1.1 - As/Ac
    dc3 = (Ps - 0.9*Pc)/Pc
    dc4 = -Cf/2.
    
    S = np.zeros([5])
    S[0] = F
    S[1] = dc1
    S[2] = dc2
    S[3] = dc3
    S[4] = dc4
    
    return S

def Masse(X,parametres):
    
    # ENTREE :
    #  Dc : diamètre au col de tuyère (en m)
    #  mp : masse du bloc de poudre (en kg)
    #  Pc : pression de combustion (en bars)
    #  L0 : Longueur utile du bloc (en m)
    #  D0 : Diamètre hors tout du propulseur (en m)
    #
    # SORTIE :  
    #  M(1)       : Masse totale (kg)
    #  M(2)       : Masse sèche case (kg)
    #  M(3)       : Masses annexes (tuyère + allumeur) (kg)
    #  M(4)       : Contrainte de remplissage
    #  M(5)       : Contrainte de serrage
    #  M(6)      : Contrainte encombrement 1
    #  M(7)      : Contrainte encombrement 2
    #  M(8)      : Durée de combustion (en s)
    
    Dc = X[0]
    mp = X[1]
    Pc = X[2]
    L0 = X[3]
    D0 = X[4]
    
    m_payload = 500.
    
    Vref = parametres[0]
    Pref = parametres[1]
    n_ = parametres[2]
    c_etoile = parametres[3]
    ga = parametres[4]
    rhop = parametres[5]
    rhom = parametres[6]
    sigmar = parametres[7]
    eps = parametres[8]
    
    Crlim = 0.87
    rhopt = 1445.7
    Ac = np.pi*Dc**2./4.
    deb = Pc*100000.*Ac/c_etoile
    tc = mp/deb
    
    
    e = Pc*D0*eps/2./sigmar
    ept = 1.11/1000.


    Vp = mp/rhop
    Aport = np.pi*(D0-2.*(e+ept))**2./4. - Vp/L0
    Dport = np.sqrt(4.*Aport/np.pi)
    Scombeff = np.pi*Dport*L0
    a = Vref*Pref**(-n_)
    exposant1 = 1./(1.-n_)
    a2 = (a*(10**(-5))**n_)/1000.
    Kn = (Pc*100000.)**(1./exposant1)/(rhop*a2*c_etoile)
    Scomb = Kn*Ac
    Cr = mp/rhop/(L0*np.pi*(D0 - 2.*e - 2.*ept)**2.)*4.
    if (Scombeff <= 0):
        g1 = (Scomb - Scombeff)/Scomb
        g2 = (3.*Scombeff - Scomb)/Scomb
    else:
        g1 = (Scombeff - Scomb)/Scomb
        g2 = (Scomb - 3.*Scombeff)/Scomb

    M = np.zeros([8])
    M[0] = mp +1.05*( (rhom*e+rhopt*ept)*D0*np.pi*L0 + 2.*np.pi*D0**2./4.*e*rhom)+m_payload
    M[1] = 1.05*( (rhom*e+rhopt*ept)*D0*np.pi*L0 + 2.*np.pi*D0**2./4.*e*rhom)+m_payload
    M[2] = 0.05*(mp + (rhom*e+rhopt*ept)*D0*np.pi*L0+ 2.*np.pi*D0**2./4.*e*rhom)
    M[3] = (Cr-Crlim)/Crlim
    if Aport==0.:
        M[4] = Ac*1.3
    else:    
        M[4] = (Ac*1.3-Aport)/Aport 
    M[5] = g1
    M[6] = g2
    M[7] = tc
    
    return M



def Booster(X_):
    
    Z = np.array([0,0,0])
    
    lb_bnd = np.array([0.2,0.5,5,5000])
    up_bnd = np.array([1,1.2,100,15000])
    
    X = lb_bnd + (up_bnd - lb_bnd)*X_
    np_ = Z[0]  #0: Butalite, 1 : Butalane, 2 : Nitramite, 3 : pAIM-120
    nm = Z[1]  # 0: Acier, 1: Aluminium
    nmot = Z[2] #0: Moteur 1, 1: Moteur 2, 2: Moteur 3
    
    
    propergols=np.array([[1,	2698.,	1.22,	0.0244,	1598.,	14.500,	5000000.,	0.40,	1469.69150388984],
    [2,	3384.,	1.14,	0.0276,	1766.,	14.50,	5000000.,	0.40,	1585.94906686235],
    [3,	3160,	1.18,	0.025,	1757.,	   14.50,	5000000.,	0.40,	1590.28799661913],
    [4,	3000.,	1.18,	0.0265, 1729.98,	8.8646,	6894760.,	0.40,	1504.18794586416]])
    
    
    Vref = propergols[np_,5]
    Pref = propergols[np_,6]/100000
    n_ = propergols[np_,7]
    c_etoile = propergols[np_,8]
    ga = propergols[np_,2]
    rhop = propergols[np_,4]
    
    if nm==0:
        rhom = 7830.
        sigmar = 11000.
        eps = 1.3
    elif nm==1:
        rhom = 2800.
        sigmar = 4000.
        eps = 1.5
        
    if nmot==0:
        eff = 0.95
        m_penal = 0.90
    elif nmot ==1:
        eff = 1.0
        m_penal = 1.0
    elif nmot ==2:
        eff = 1.05
        m_penal = 1.1


    parametres = np.array([Vref,Pref,n_,c_etoile,ga, rhop,rhom,sigmar,eps])
        
    Xp = np.zeros([4]) 
    Xp[0] = X[0]
    Xp[1] = X[1]
    Xp[2] = 0.83
    Xp[3] = X[2]
    S = Poussee(Xp,parametres) 
    
    Xm = np.zeros([5])
    Xm[0] = X[0]
    Xm[1] = X[3]
    Xm[2] = X[2]
    Xm[3] = 11.
    Xm[4] = 1.07
    M = Masse(Xm,parametres)
   
    M[0] = M[0]
    M[1] = M[1]
    M[2] = M[2] 
    

    Isp = eff*S[0]/X[3]*M[7]/9.81
    mi = M[0]
    mf = M[1]
    Delta_V = 9.81*Isp*np.log(mi/mf)
    
    Obj2 = M[0]
    
    Cons = np.zeros([9])
    Cons[0] = S[1]
    Cons[1] = S[2]
    Cons[2] = S[3]
    Cons[3] = S[4]
    Cons[4] = M[3]
    Cons[5] = M[4]
    Cons[6] = M[5]
    Cons[7] = M[6]
    
    
    Obj1=-Delta_V/1000.
#    Obj2=(Obj2-5800)/(17500.-5800.)
    Cons[0]=Cons[0]*1/120.
    Cons[1]=Cons[1]*1/35.
    Cons[5]=Cons[5]*1/34.
    Cons[6]=Cons[6]*1/20.
    Cons[7]=Cons[7]*1/60.
    Cons[8]=(Obj2-11000.)/10000.
#    Cons[8]=(4500-Delta_V)/4500.
    # Cons[8]=(Cons[8]+1200)/(4400+1200)-0.214
    # Cons=np.where(Cons>0,Cons,0)

    return Obj1, Cons

##############################################################################################
##############################################################################################
