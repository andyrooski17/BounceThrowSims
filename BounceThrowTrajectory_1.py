# Last edit 8/10/2017 Andrew Dominijanni
# This code meant to be run in "cell mode", blocks separated by #%%

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import optimize

#%%
pi = np.pi
g = 9.806
R = .0368 #radius of ball
# unit vectors
xhat = np.array([1,0])
yhat = np.array([0,1])

#define tranformation matricies
T = np.array([[0,-1],[1,0]]) #90degree rotation CCW
Rev = np.array([[-1,0],[0,-1]]) # switch direction (180 degree rotation)

# defin conversions to mph, mps for convenience.
def mph(mps):
    return mps*2.236936292054402291
    
def mps(mph):
    return mph/2.236936292054402291

# define function to return drag coefficient. In simplest case, assume fixed.
# more refined model would recalculate as a function of speed and spin.
def Cd(v):
    return 0.35
    
# define lift coefficient as a function of spin ratio per Nathan, Sawicki.
def Cl(S):
    if S > 0.1:
        return 0.09+0.6*S
    else:
        return 1.5*S

# calculate acceleration in the x direction given a current velocity vector
# accounts for component of drag force (opposite direction of velocity)
# accounts for component lift force (perpendicular to direction of velo)
def ax(vvect, S=.175, m=.1453, rho=1.23, R=.0368):
    A = pi*R**2
    spd = np.linalg.norm(vvect)
    cd = Cd(spd)
    cl = Cl(S)
    Fm = 1/2*cl*rho*A*spd**2
    Fd = 1/2*cd*rho*A*spd**2
    Fddir = np.dot(Rev, vvect) / np.linalg.norm(np.dot(Rev, vvect))
    Fmdir = np.dot(T, vvect) / np.linalg.norm(np.dot(T, vvect))
    Fdx = np.dot(Fd*Fddir, xhat)
    Fmx = np.dot(Fm*Fmdir, xhat)
    return 1/m*(Fdx+Fmx)

# calculate acceleration in the y direction given a current velocity vector
# accounts for component of drag force (opposite direction of velocity)
# accounts for component lift force (perpendicular to direction of velo)
# accounts for gravitational force
def ay(vvect, S=.175, m=.1453, rho=1.23, R=.0368):
    A = pi*R**2
    spd = np.linalg.norm(vvect)
    cd = Cd(spd)
    cl = Cl(S)
    Fm = 1/2*cl*rho*A*spd**2
    Fd = 1/2*cd*rho*A*spd**2
    Fddir = np.dot(Rev, vvect) / np.linalg.norm(np.dot(Rev, vvect)) 
    Fmdir = np.dot(T, vvect) / np.linalg.norm(np.dot(T, vvect))
    Fdy = np.dot(Fd*Fddir, yhat)
    Fmy = np.dot(Fm*Fmdir, yhat)
    return 1/m*(Fdy+Fmy-m*g)

# calculate angular velocity from rpm
def rpmtow(rpm):
    return rpm/60*2*pi

# definte calculation of spin ratio
def SpnRat(vvect, omega, R=.0368):
    spd = np.linalg.norm(vvect)
    return R*omega/spd

# define mapping between release speed and spin rate
def mphtorpm(mph=80):
    #from savant
    return mph*19.55+417.38

# convert ft to m
def fttom(ft):
    return np.array(ft)*0.3048
    
# convert m to ft.
def mtoft(m):
    return np.array(m)/0.3048
    
# find nearest value and index of value in an array to the input value
def findnearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return array[idx] , idx
    
# convert degrees to radians
def rads(degrees):
    return np.array(degrees)*np.pi/180

# convert radians to degrees
def degs(radians):
    return np.array(radians)*180/np.pi

#%% fourth order #2
from scipy.interpolate import interp1d
def ThrowSim(Angle=0,Speed=35,Release=[0,1.75],TargetX=30,usebounce=True,steps=5000):
    start=0 #start time
    end=3 #end time
    #steps=3000 #steps
    t,h=np.linspace(start,end,steps,retstep=True)
    
    spdo=Speed
    #rpm=0
    rpm=mphtorpm(mph(spdo))
    w=rpmtow(rpm)
    #ang=6*pi/180 
    ang=Angle
    vo=np.array([spdo*np.cos(ang),spdo*np.sin(ang)])
    v=np.zeros((2,len(t)))
    v[0,0]=vo[0]
    v[1,0]=vo[1]
    a=v.copy()
    xo=Release
    #xo=np.array([0,1.75])
    x=a.copy()
    x[:,0]=xo
    ey=0.55
    ex=0.7
    mu=0.4
    bnc=False
    bncix=0
    bncendix=0
    t6=0;
    
    for n,tn in enumerate(t[0:-1]):
        #if (x[1,n-1]+(x[1,n]-x[1,n-1])) <= 0 and bnc is False:
        if (x[1,n]+v[1,n]*h) <= 0 and bnc is False and usebounce:
            vn=v[:,n]
            vy1=vn[1]
            vy2=-vy1*ey
            vn[1]=vy2
            vx1=vn[0]
            vx2=vx1+mu*(1+ey)*vy1
            vn[0]=vx2
            w1=w
            w=-w-(1/.4)*mu*(1+ey)*vy1/R #reverse for sign convention from Cross
            w2=-w
            #w=rpm/60*2*pi
            Sf=SpnRat(vn,w)
            
            bnc=True
            bncix=n
        else:
            vn=v[:,n]
            Sf=SpnRat(vn,w)
            
        k1=np.array([ax(vn,S=Sf),ay(vn,S=Sf)])
        k2=np.array([ax(vn+k1*h/2,S=Sf),ay(vn+k1*h/2,S=Sf)])
        k3=np.array([ax(vn+k2*h/2,S=Sf),ay(vn+k2*h/2,S=Sf)])
        k4=np.array([ax(vn+k3*h,S=Sf),ay(vn+k3*h,S=Sf)])
        an=1/6*(k1+2*k2+2*k3+k4)
        a[:,n]=an
        v[:,n+1]=vn+h*an
        x[:,n+1]=x[:,n]+h/2*(vn+v[:,n+1])
        
        if bncix!=0 and (x[1,n+1]-x[1,n])<0 and x[1,n-1]-x[1,n]<0:
            bncendix=n
        elif bnc!=0 and bncendix==0 and n==(len(t[0:-1])-1):
            bncendix=n
        elif bnc==0 and bncendix==0 and n==(len(t[0:-1])-1):
            bncendix=n
            bncix=n-2
            
    if usebounce and bnc:
        #acttargetx, targetix = findnearest(x[0,bncix:bncendix],TargetX)
        acttargetx= np.interp(TargetX,x[0,bncix:bncendix],x[0,bncix:bncendix])
        #inty = interp1d(x[0,bncix:bncendix],x[1,bncix:bncendix],kind='quadratic',bounds_error=False,fill_value=x[1,bncix])
        acttargety = np.interp(TargetX,x[0,bncix:bncendix],x[1,bncix:bncendix])
        #intt = interp1d(x[0,bncix:bncendix],t[bncix:bncendix],kind='quadratic',bounds_error=False,fill_value=t[bncix])
        acttargett = np.interp(TargetX,x[0,bncix:bncendix],t[bncix:bncendix])
    else:
        #acttargetx , targetix = findnearest(x[0,:],TargetX)
        acttargetx= np.interp(TargetX,x[0,:],x[0,:])
        #inty = interp1d(x[0,:],x[1,:],kind='quadratic',fill_value='extrapolate')
        acttargety = np.interp(TargetX,x[0,:],x[1,:])
        #intt = interp1d(x[0,bncix:bncendix],t[bncix:bncendix],kind='quadratic',bounds_error=False,fill_value=t[bncix])
        acttargett = np.interp(TargetX,x[0,:],t)
    return [acttargetx, acttargety,acttargett], x , v, bncix, bncendix
    #x2=x.copy()
    

    
#%%
mintest=angopt['x']
targx=30
targy=fttom(0.5)
tr,pos,v,bix,beix=ThrowSim(Angle=mintest,Speed=38,Release=[0,1.75],TargetX=targx,usebounce=True,steps=1000)

#%%
lw=2.25
fs=16
plt.figure(figsize=(12,4))
plt.plot(mtoft([targx,targx]),[0,30],'-k',linewidth=lw,label='First Baseman')
plt.plot(pos[0,:]*3.281,pos[1,:]*3.281,linewidth=lw,label='Air Throw')
plt.plot(mtoft(tr[0]),mtoft(tr[1]),'or')
#plt.xlim([0,mtoft(35)])
plt.ylim([0,np.amax(mtoft(pos[1,:]))])
#%%
spd2=38
targx=30
targy=fttom(0.5)
def TargetError(x):
    tr,pos,v,bix,beix = ThrowSim(Angle=x,Speed=spd2,Release=[0,1.75],TargetX=targx,usebounce=True,steps=2000)
    return np.sqrt((tr[1]-targy)**2+(tr[0]-targx)**2)

#%%
sp.optimize.minimize(TargetError,0.05,bounds=[[-0.2,0.3]])
#%%
sp.optimize.minimize(TargetError,-.04,method='Nelder-Mead')
#%%
mintest=sp.optimize.brute(TargetError,[[-0.2,0.2]],finish='fmin')
#%%
TargetError(angopt['x'])
#%%
mintest=sp.optimize.brute(TargetError,[[-0.2,0.2]])
mintest2=sp.optimize.minimize(TargetError,mintest,bounds=[[mintest-mintest*0.2,mintest+mintest*0.2]])

#%%
angopt=sp.optimize.minimize_scalar(TargetError,method='Bounded',bounds=(-0.3,0.4),options={'xatol':1e-5})
#%%
tr,pos,v,bix,beix=ThrowSim(Angle=angopt['x'],Speed=spd2,Release=[0,1.75],TargetX=targx,usebounce=True,steps=5000)
#%%
lw=2.25
fs=16
plt.figure(figsize=(12,4))
plt.plot(mtoft([tr[0],tr[0]]),[0,20],'-k',linewidth=lw,label='First Baseman')
plt.plot(pos[0,:]*3.281,pos[1,:]*3.281,linewidth=lw,label='Air Throw')
plt.xlim([0,mtoft(35)])
plt.ylim([0,10])
plt.grid('on')
#%%
from scipy import optimize
#%% HERE
spds=mps(np.linspace(60,100,10))
#spds=mps(np.array(95))
angle=np.zeros(spds.shape)
time=np.zeros(spds.shape)
trs=np.zeros((3,len(spds)))
angleb=np.zeros(spds.shape)
timeb=np.zeros(spds.shape)
trsb=np.zeros((3,len(spds)))
guess=.1
for i in range(len(spds)):
    spd2=spds[i]
    targx=30
    targy=fttom(0.5)
    def TargetError(x):
        tr,pos,v,bix,beix = ThrowSim(Angle=x,Speed=spd2,Release=[0,1.75],TargetX=targx,usebounce=True,steps=1000)
        return np.sqrt((tr[1]-targy)**2+(tr[0]-targx)**2)
    
    angopt=sp.optimize.minimize_scalar(TargetError,method='Bounded',bounds=(-0.15,0.3),options={'xatol':1e-5})
    angleb[i]=angopt['x']
    #guess = angle[i]
    
    trsb[:,i],pos,v,bix,beix=ThrowSim(Angle=angleb[i],Speed=spds[i],Release=[0,1.75],TargetX=targx,usebounce=True,steps=1000)
    timeb[i]=trsb[2,i]
    
for i in range(len(spds)):
    spd2=spds[i]
    targx=30
    targy=fttom(2)
    def TargetError(x):
        tr,pos,v,bix,beix = ThrowSim(Angle=x,Speed=spd2,Release=[0,1.75],TargetX=targx,usebounce=False,steps=1000)
        return np.sqrt((tr[1]-targy)**2+(tr[0]-targx)**2)
    
    angopt=sp.optimize.minimize_scalar(TargetError,method='Bounded',bounds=(-0.15,0.3),options={'xatol':1e-5})
    angle[i]=angopt['x']
    guess = angle[i]
    
    trs[:,i],pos,v,bix,beix=ThrowSim(Angle=angle[i],Speed=spds[i],Release=[0,1.75],TargetX=targx,usebounce=False,steps=1000)
    time[i]=trs[2,i]
#%%
plt.figure()
plt.plot(time,'ob')
plt.plot(timeb,'or')
#%%
plt.figure()
plt.plot(time-timeb,'og')
#%%
#%%
indx=5
tr,pos,v,bix,beix=ThrowSim(Angle=angle[indx],Speed=spds[indx],Release=[0,1.75],TargetX=targx,usebounce=False,steps=1000)

lw=2.25
fs=16
plt.figure(figsize=(12,4))
plt.plot(mtoft([tr[0],tr[0]]),[0,20],'-k',linewidth=lw,label='First Baseman')
plt.plot(pos[0,:]*3.281,pos[1,:]*3.281,linewidth=lw,label='Air Throw')
plt.scatter(mtoft(targx),mtoft(targy),50)
plt.xlim([0,mtoft(35)])
plt.ylim([0,10])
plt.grid('on')
plt.show()
#%% HERE for all dists
stps=2000
spds=mps(np.linspace(60,100,21))
dsts=fttom(np.linspace(60,130,21))
#spds=mps(np.array(95))
angle=np.zeros((len(spds),len(dsts)))
time=np.zeros((len(spds),len(dsts)))
trs=np.zeros((3,len(spds),len(dsts)))
angleb=angle.copy()
timeb=time.copy()
trsb=np.zeros((3,len(spds),len(dsts)))
error=np.zeros((len(spds),len(dsts)))
errorb=np.zeros((len(spds),len(dsts)))
success=np.zeros((len(spds),len(dsts)))
successb=np.zeros((len(spds),len(dsts)))
guess=.1
for j in range(len(dsts)):
    for i in range(len(spds)):
        spd2=spds[i]
        targx=dsts[j]
        targy=fttom(0.5)
        def TargetError(x):
            tr,pos,v,bix,beix = ThrowSim(Angle=x,Speed=spd2,Release=[0,1.75],TargetX=targx,usebounce=True,steps=stps)
            return np.sqrt((tr[1]-targy)**2+(tr[0]-targx)**2)
        
        angopt=sp.optimize.minimize_scalar(TargetError,method='Bounded',bounds=(-0.2,0.3),options={'xatol':1e-5})
        angleb[i,j]=angopt['x']
        errorb[i,j]=angopt['fun']
        successb[i,j]=angopt['success']*1
        #guess = angle[i]
        
        trsb[:,i,j],pos,v,bix,beix=ThrowSim(Angle=angleb[i,j],Speed=spds[i],Release=[0,1.75],TargetX=targx,usebounce=True,steps=stps)   
        timeb[i,j]=trsb[2,i,j]
        print([i,j,'b',errorb[i,j]*1000,successb[i,j]])
   
for j in range(len(dsts)):
    for i in range(len(spds)):
        spd2=spds[i]
        targx=dsts[j]
        targy=fttom(2)
        def TargetError(x):
            tr,pos,v,bix,beix = ThrowSim(Angle=x,Speed=spd2,Release=[0,1.75],TargetX=targx,usebounce=False,steps=stps)
            return np.sqrt((tr[1]-targy)**2+(tr[0]-targx)**2)
        
        angopt=sp.optimize.minimize_scalar(TargetError,method='Bounded',bounds=(-0.2,0.3),options={'xatol':1e-5})
        angle[i,j]=angopt['x']
        #guess = angle[i,j]
        error[i,j]=angopt['fun']
        success[i,j]=angopt['success']*1
        
        trs[:,i,j],pos,v,bix,beix=ThrowSim(Angle=angle[i,j],Speed=spds[i],Release=[0,1.75],TargetX=targx,usebounce=False,steps=stps)
        time[i,j]=trs[2,i,j]
        print([i,j,'a',error[i,j]*1000,success[i,j]])
        
np.save('spds',spds)
np.save('angleb',angleb)
np.save('dsts',dsts)
np.save('angle',angle)
np.save('trsb',trsb)
np.save('trs',trs)
np.save('timeb',timeb)
np.save('time',time)
np.save('errorb',errorb)
np.save('error',error)
np.save('successb',successb)
np.save('success',success)
#%%
spds=np.load('spds.npy')
angleb=np.load('angleb.npy')
dsts=np.load('dsts.npy')
angle=np.load('angle.npy')
trsb=np.load('trsb.npy')
trs=np.load('trs.npy')
timeb=np.load('timeb.npy')
time=np.load('time.npy')
errorb=np.load('errorb.npy')
error=np.load('error.npy')
successb=np.load('successb.npy')
success=np.load('success.npy')
#%% HERE for higher/lower catch point dists.
stps=2000
spds=mps(np.linspace(60,100,21))
dsts=fttom(np.linspace(60,130,21))
#spds=mps(np.array(95))
angleh2=np.zeros((len(spds),len(dsts)))
timeh2=np.zeros((len(spds),len(dsts)))
trsh2=np.zeros((3,len(spds),len(dsts)))
#anglebh=angleh.copy()
#timebh=timeh.copy()
#trsbh=np.zeros((3,len(spds),len(dsts)))
errorh2=np.zeros((len(spds),len(dsts)))
#errorbh=np.zeros((len(spds),len(dsts)))
successh2=np.zeros((len(spds),len(dsts)))
#successbh=np.zeros((len(spds),len(dsts)))
guess=.1
for j in range(len(dsts)):
    for i in range(len(spds)):
        spd2=spds[i]
        targx=dsts[j]
        targy=fttom(.5)
        def TargetError(x):
            tr,pos,v,bix,beix = ThrowSim(Angle=x,Speed=spd2,Release=[0,1.75],TargetX=targx,usebounce=False,steps=stps)
            return np.sqrt((tr[1]-targy)**2+(tr[0]-targx)**2)
        
        angopt=sp.optimize.minimize_scalar(TargetError,method='Bounded',bounds=(-0.2,0.3),options={'xatol':1e-5})
        angleh2[i,j]=angopt['x']
        #guess = angle[i,j]
        errorh2[i,j]=angopt['fun']
        successh2[i,j]=angopt['success']*1
        
        trsh2[:,i,j],pos,v,bix,beix=ThrowSim(Angle=angleh2[i,j],Speed=spds[i],Release=[0,1.75],TargetX=targx,usebounce=False,steps=stps)
        timeh2[i,j]=trsh2[2,i,j]
        print([i,j,'a',errorh2[i,j]*1000,successh2[i,j]])
        
#np.save('anglebh',anglebh)
np.save('angleh2',angleh2)
#np.save('trsbh',trsbh)
np.save('trsh2',trsh2)
#np.save('timebh',timebh)
np.save('timeh2',timeh2)
#np.save('errorbh',errorbh)
np.save('errorh2',errorh2)
#np.save('successbh',successbh)
np.save('successh2',successh2)        
        
#for j in range(len(dsts)):
#    for i in range(len(spds)):
#        spd2=spds[i]
#        targx=dsts[j]
#        targy=fttom(1)
#        def TargetError(x):
#            tr,pos,v,bix,beix = ThrowSim(Angle=x,Speed=spd2,Release=[0,1.75],TargetX=targx,usebounce=True,steps=stps)
#            return np.sqrt((tr[1]-targy)**2+(tr[0]-targx)**2)
#        
#        angopt=sp.optimize.minimize_scalar(TargetError,method='Bounded',bounds=(-0.25,0.35),options={'xatol':1e-5})
#        anglebh[i,j]=angopt['x']
#        errorbh[i,j]=angopt['fun']
#        successbh[i,j]=angopt['success']*1
#        #guess = angle[i]
#        
#        trsbh[:,i,j],pos,v,bix,beix=ThrowSim(Angle=anglebh[i,j],Speed=spds[i],Release=[0,1.75],TargetX=targx,usebounce=True,steps=stps)   
#        timebh[i,j]=trsbh[2,i,j]
#        print([i,j,'b',errorbh[i,j]*1000,successbh[i,j]])
#   
#        
#np.save('spds',spds)
#np.save('dsts',dsts)
#
#np.save('anglebh',anglebh)
##np.save('angleh',angleh)
#np.save('trsbh',trsbh)
##np.save('trsh',trsh)
#np.save('timebh',timebh)
##np.save('timeh',timeh)
#np.save('errorbh',errorbh)
##np.save('errorh',errorh)
#np.save('successbh',successbh)
##np.save('successh',successh)
#%%
indx=0
disti=1
tr,pos,v,bix,beix=ThrowSim(Angle=angleb[indx,disti],Speed=spds[indx],Release=[0,1.75],TargetX=dsts[disti],usebounce=True,steps=stps)

lw=2.25
fs=16
plt.figure(figsize=(12,4))
plt.plot(mtoft([tr[0],tr[0]]),[0,20],'-k',linewidth=lw,label='First Baseman')
plt.plot(pos[0,:]*3.281,pos[1,:]*3.281,linewidth=lw,label='Bounce Throw')
plt.scatter(mtoft(dsts[disti]),0.5,50)
plt.xlim([0,mtoft(dsts[disti]+10)])
plt.ylim([0,10])
plt.grid('on')
plt.show()
#%%
#%%
indx=0
disti=1
tr,pos,v,bix,beix=ThrowSim(Angle=angle[indx,disti],Speed=spds[indx],Release=[0,1.75],TargetX=dsts[disti],usebounce=False,steps=stps)

lw=2.25
fs=16
plt.figure(figsize=(12,4))
plt.plot(mtoft([tr[0],tr[0]]),[0,20],'-k',linewidth=lw,label='First Baseman')
plt.plot(pos[0,:]*3.281,pos[1,:]*3.281,linewidth=lw,label='Bounce Throw')
plt.scatter(mtoft(dsts[disti]),2,50)
plt.xlim([0,mtoft(dsts[disti]+10)])
plt.ylim([0,10])
plt.grid('on')
plt.show()
#%%
indx=17
disti=13
tr,pos,v,bix,beix=ThrowSim(Angle=anglebh[indx,disti],Speed=spds[indx],Release=[0,1.75],TargetX=dsts[disti],usebounce=True,steps=stps)

lw=2.25
fs=16
plt.figure(figsize=(12,4))
plt.plot(mtoft([tr[0],tr[0]]),[0,20],'-k',linewidth=lw,label='First Baseman')
plt.plot(pos[0,:]*3.281,pos[1,:]*3.281,linewidth=lw,label='Bounce Throw')
plt.scatter(mtoft(dsts[disti]),1,50)
plt.xlim([0,mtoft(dsts[disti]+10)])
plt.ylim([0,10])
plt.grid('on')
plt.show()
#%%
#%%
indx=0
disti=1
tr,pos,v,bix,beix=ThrowSim(Angle=angleh[indx,disti],Speed=spds[indx],Release=[0,1.75],TargetX=dsts[disti],usebounce=False,steps=stps)

lw=2.25
fs=16
plt.figure(figsize=(12,4))
plt.plot(mtoft([tr[0],tr[0]]),[0,20],'-k',linewidth=lw,label='First Baseman')
plt.plot(pos[0,:]*3.281,pos[1,:]*3.281,linewidth=lw,label='Bounce Throw')
plt.scatter(mtoft(dsts[disti]),4,50)
plt.xlim([0,mtoft(dsts[disti]+10)])
plt.ylim([0,10])
plt.grid('on')
plt.show()
#%%
indx=2
disti=2
tr,pos,v,bix,beix=ThrowSim(Angle=angle[indx,disti],Speed=spds[indx],Release=[0,1.75],TargetX=dsts[disti],usebounce=False,steps=5000)

lw=2.25
fs=16
plt.figure(figsize=(12,4))
plt.plot(mtoft([tr[0],tr[0]]),[0,20],'-k',linewidth=lw,label='First Baseman')
plt.plot(pos[0,:]*3.281,pos[1,:]*3.281,linewidth=lw,label='Bounce Throw')
plt.scatter(mtoft(dsts[disti]),2,50)
plt.xlim([0,mtoft(dsts[disti]+10)])
plt.ylim([0,10])
plt.grid('on')
plt.show()
#%%
plt.figure()
plt.plot(time.flatten(),'ob')
plt.plot(timeb.flatten(),'or')
#%%
plt.figure()
plt.plot(timeh.flatten(),'ob')
plt.plot(timebh.flatten(),'or')

#%%
plt.figure()
plt.plot(time.flatten(),'ob')
plt.plot(timeh.flatten(),'og')
#%%
plt.figure()
plt.plot(timeb.flatten()-time.flatten(),'ob')

#%% plotting
difr=timeh-timeh2
plt.figure()
for j in range(len(dsts)):
    for i in range(len(spds)):
        if errorh[i,j]<=.1 and errorh2[i,j]<=.1:
            if difr[i,j] <= 0:
                plt.plot(mph(spds[i]),mtoft(dsts[j]),'or')
            else:
                plt.plot(mph(spds[i]),mtoft(dsts[j]),'ob')
plt.xlim([55,105])
plt.ylim([55,135])
#%% plotting pct
difr=timeb-timeh
plt.figure()
for j in range(len(dsts)):
    for i in range(len(spds)):
        if errorb[i,j]<=.1 and error[i,j]<=.1:
            if difr[i,j] <= 0:
                plt.plot(mph(spds[i]),mtoft(dsts[j]),'or')
            else:
                plt.plot(mph(spds[i]),mtoft(dsts[j]),'ob')
plt.xlim([55,105])
plt.ylim([55,135])
#%% plotting
difr=timebh-time
plt.figure()
for j in range(len(dsts)):
    for i in range(len(spds)):
        if errorbh[i,j]<=.04 and errorh[i,j]<=.04:
            if difr[i,j] <= 0:
                plt.plot(mph(spds[i]),mtoft(dsts[j]),'or')
            else:
                plt.plot(mph(spds[i]),mtoft(dsts[j]),'ob')
plt.xlim([55,105])
plt.ylim([55,135])
#%%

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

S,D=np.meshgrid(spds,dsts)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(S, D, difr, cmap='cool',vmin=np.amin(difr),vmax=np.amax(difr))
#%%
fs=14
fs2=20
fstp=12
S,D=np.meshgrid(spds,dsts)
zers=np.zeros(S.shape)
difr=timeb-time
sz=70
fig = plt.figure()
axs = fig.add_subplot(111, projection='3d')
for j in range(len(dsts)):
    for i in range(len(spds)):
        if error[i,j]<=.01 and errorb[i,j]<=.01:
            if difr[i,j] <= 0:
                axs.scatter(mph(spds[i]), mtoft(dsts[j]), difr[i,j],s=sz, c='r')
            else:
                axs.scatter(mph(spds[i]), mtoft(dsts[j]), difr[i,j],s=sz, c='b')
axs.plot_surface(mph(S),mtoft(D),zers,color='k',alpha=0.2,linewidth=0)
axs.axis([60,100,60,130])
axs.set_zlim3d([-0.01, 0.01])
axs.view_init(20,-150)
#axs.view_init(90,-90)
axs.set_xlabel('Throw Speed (mph)',fontsize=fs)
axs.set_ylabel('Distance to First (ft)',fontsize=fs)
axs.set_zlabel('Bounce - Air Throw Time (s)',fontsize=fs)
plt.tick_params(axis='both', which='major', labelsize=fstp)
plt.tight_layout()
plt.savefig('FullSimData.png',dpi=600)
plt.show()
#%% higher air trgt
fs=14
fs2=20
fstp=12
S,D=np.meshgrid(spds,dsts)
zers=np.zeros(S.shape)
difr=timeb-timeh
sz=70
fig = plt.figure()
axs = fig.add_subplot(111, projection='3d')
for j in range(len(dsts)):
    for i in range(len(spds)):
        if errorh[i,j]<=.01 and errorb[i,j]<=.01:
            if difr[i,j] <= 0:
                axs.scatter(mph(spds[i]), mtoft(dsts[j]), difr[i,j],s=sz, c='r')
            else:
                axs.scatter(mph(spds[i]), mtoft(dsts[j]), difr[i,j],s=sz, c='b')
axs.plot_surface(mph(S),mtoft(D),zers,color='k',alpha=0.2,linewidth=0)
axs.axis([60,100,60,130])
axs.set_zlim3d([-0.015, 0.015])
axs.view_init(20,-150)
#axs.view_init(90,-90)
axs.set_xlabel('Throw Speed (mph)',fontsize=fs)
axs.set_ylabel('Distance to First (ft)',fontsize=fs)
axs.set_zlabel('Bounce - Air Throw Time (s)',fontsize=fs)
plt.tick_params(axis='both', which='major', labelsize=fstp)
plt.tight_layout()
plt.savefig('FullSimData_higherrgt.png',dpi=600)
plt.show()
#%% higher air trgt
fs=14
fs2=20
fstp=12
S,D=np.meshgrid(spds,dsts)
zers=np.zeros(S.shape)
difr=timeb-timeh2
sz=70
fig = plt.figure()
axs = fig.add_subplot(111, projection='3d')
for j in range(len(dsts)):
    for i in range(len(spds)):
        if errorh2[i,j]<=.01 and errorb[i,j]<=.01:
            if difr[i,j] <= 0:
                axs.scatter(mph(spds[i]), mtoft(dsts[j]), difr[i,j],s=sz, c='r')
            else:
                axs.scatter(mph(spds[i]), mtoft(dsts[j]), difr[i,j],s=sz, c='b')
axs.plot_surface(mph(S),mtoft(D),zers,color='k',alpha=0.2,linewidth=0)
axs.axis([60,100,60,130])
axs.set_zlim3d([-0.015, 0.015])
axs.view_init(20,-150)
#axs.view_init(90,-90)
axs.set_xlabel('Throw Speed (mph)',fontsize=fs)
axs.set_ylabel('Distance to First (ft)',fontsize=fs)
axs.set_zlabel('Bounce - Air Throw Time (s)',fontsize=fs)
plt.tick_params(axis='both', which='major', labelsize=fstp)
plt.tight_layout()
plt.savefig('FullSimData_lowertrgt.png',dpi=600)
plt.show()
#%% plot pct
fs=14
fs2=20
fstp=12
S,D=np.meshgrid(spds,dsts)
zers=np.zeros(S.shape)
difr=timeb-time
sz=70
fig = plt.figure()
axs = fig.add_subplot(111, projection='3d')
for j in range(len(dsts)):
    for i in range(len(spds)):
        if error[i,j]<=.01 and errorb[i,j]<=.01:
            if difr[i,j] <= 0:
                axs.scatter(mph(spds[i]), mtoft(dsts[j]), difr[i,j]/time[i,j],s=sz, c='r')
            else:
                axs.scatter(mph(spds[i]), mtoft(dsts[j]), difr[i,j]/time[i,j],s=sz, c='b')
axs.plot_surface(mph(S),mtoft(D),zers,color='k',alpha=0.2,linewidth=0)
axs.axis([60,100,60,130])
axs.set_zlim3d([-0.015, 0.015])
axs.view_init(20,-150)
#axs.view_init(90,-90)
axs.set_xlabel('Throw Speed (mph)',fontsize=fs)
axs.set_ylabel('Distance to First (ft)',fontsize=fs)
axs.set_zlabel('Difference Fraction of Throw Time',fontsize=fs)
plt.tick_params(axis='both', which='major', labelsize=fstp)
plt.tight_layout()
plt.savefig('FullSimData_pct.png',dpi=600)
plt.show()
#%% %%%%%%%%%%%%%%%%%%%%%
ar=6520/1667
scl=1/2.5
indx=10
disti=10
print(mph(spds[indx]))
print(mtoft(dsts[disti]))
print(error[indx,disti])
print(errorb[indx,disti])
tr,pos,v,bix,beix=ThrowSim(Angle=angleb[indx,disti],Speed=spds[indx],Release=[0,1.75],TargetX=dsts[disti],usebounce=True,steps=5000)
tr2,pos2,v2,bix2,beix2=ThrowSim(Angle=angle[indx,disti],Speed=spds[indx],Release=[0,1.75],TargetX=dsts[disti],usebounce=False
,steps=5000)

lw=2.25
fs=16
plt.figure(figsize=(12,4))
plt.plot(mtoft([tr[0],tr[0]]),[0,20],'-k',linewidth=lw,label='First Baseman')
plt.plot(mtoft(pos2[0,:]),mtoft(pos2[1,:]),'b',linewidth=lw,label='Air Throw')
plt.plot(mtoft(pos[0,:]),mtoft(pos[1,:]),'r',linewidth=lw,label='Bounce Throw')
l=range(len(t))
for i in l[0::50]:
    plt.plot([mtoft(pos[0,i]),mtoft(pos2[0,i])],[mtoft(pos[1,i]),mtoft(pos2[1,i])],'--k',linewidth=lw*.5)

plt.plot([mtoft(pos[0,i]),mtoft(pos2[0,i])],[mtoft(pos[1,i]),mtoft(pos2[1,i])],'--k',linewidth=lw*.5,label='Equal Time') 
#plt.scatter(mtoft(dsts[disti]),2,50)
plt.xlim([0,mtoft(dsts[disti])+5])
plt.ylim([0,13])
plt.ylim([0,scl*(1/ar)*(mtoft(dsts[disti])+5)])
plt.xlabel('Horizontal Position (ft)',fontsize=fs2)
plt.ylabel('Vertical Position (ft)',fontsize=fs2)
plt.title(str(np.around(mph(spds[indx]),1))+' mph Release Speed, '+str(int(np.around(mtoft(dsts[disti]),1)))+' ft Target Distance',fontsize=fs2)
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.legend(loc='lower left')
plt.grid('on')
plt.tight_layout()
plt.savefig('Trajectories_'+str(int(np.around(mph(spds[indx]),0)))+str(int(np.around(mtoft(dsts[disti]),0)))+'.png',dpi=600)
plt.show()
#%% %%%%%%%%%%%%%%%%%%%%% higher
ar=6520/1667
scl=1/2.5
indx=5
disti=18
print(mph(spds[indx]))
print(mtoft(dsts[disti]))
print(error[indx,disti])
print(errorb[indx,disti])
tr,pos,v,bix,beix=ThrowSim(Angle=angleb[indx,disti],Speed=spds[indx],Release=[0,1.75],TargetX=dsts[disti],usebounce=True,steps=5000)
tr2,pos2,v2,bix2,beix2=ThrowSim(Angle=angleh[indx,disti],Speed=spds[indx],Release=[0,1.75],TargetX=dsts[disti],usebounce=False
,steps=5000)

lw=2.25
fs=16
plt.figure(figsize=(12,4))
plt.plot(mtoft([tr[0],tr[0]]),[0,20],'-k',linewidth=lw,label='First Baseman')
plt.plot(mtoft(pos2[0,:]),mtoft(pos2[1,:]),'b',linewidth=lw,label='Air Throw')
plt.plot(mtoft(pos[0,:]),mtoft(pos[1,:]),'r',linewidth=lw,label='Bounce Throw')
l=range(len(t))
for i in l[0::50]:
    plt.plot([mtoft(pos[0,i]),mtoft(pos2[0,i])],[mtoft(pos[1,i]),mtoft(pos2[1,i])],'--k',linewidth=lw*.5)

plt.plot([mtoft(pos[0,i]),mtoft(pos2[0,i])],[mtoft(pos[1,i]),mtoft(pos2[1,i])],'--k',linewidth=lw*.5,label='Equal Time') 
#plt.scatter(mtoft(dsts[disti]),2,50)
plt.xlim([0,mtoft(dsts[disti])+5])
plt.ylim([0,13])
plt.ylim([0,scl*(1/ar)*(mtoft(dsts[disti])+5)])
plt.xlabel('Horizontal Position (ft)',fontsize=fs2)
plt.ylabel('Vertical Position (ft)',fontsize=fs2)
plt.title(str(np.around(mph(spds[indx]),1))+' mph Release Speed, '+str(int(np.around(mtoft(dsts[disti]),1)))+' ft Target Distance',fontsize=fs2)
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.legend(loc='lower left')
plt.grid('on')
plt.tight_layout()
plt.savefig('Trajectories_'+str(int(np.around(mph(spds[indx]),0)))+str(int(np.around(mtoft(dsts[disti]),0)))+'.png',dpi=600)
plt.show()
#%%
indx=3
disti=17
print(mph(spds[indx]))
print(mtoft(dsts[disti]))
tr,pos,v,bix,beix=ThrowSim(Angle=angleb[indx,disti],Speed=spds[indx],Release=[0,1.75],TargetX=dsts[disti],usebounce=True,steps=5000)
tr2,pos2,v2,bix2,beix2=ThrowSim(Angle=angleh[indx,disti],Speed=spds[indx],Release=[0,1.75],TargetX=dsts[disti],usebounce=False,steps=5000)

lw=2.25
fs=16
plt.figure(figsize=(12,4))
plt.plot(mtoft([tr[0],tr[0]]),[0,20],'-k',linewidth=lw,label='First Baseman')
plt.plot(mtoft(pos2[0,:]),mtoft(pos2[1,:]),'b',linewidth=lw,label='Air Throw')
plt.plot(mtoft(pos[0,:]),mtoft(pos[1,:]),'r',linewidth=lw,label='Bounce Throw')
l=range(len(t))
for i in l[0::75]:
    plt.plot([mtoft(pos[0,i]),mtoft(pos2[0,i])],[mtoft(pos[1,i]),mtoft(pos2[1,i])],'--k',linewidth=lw*.5)

plt.plot([mtoft(pos[0,i]),mtoft(pos2[0,i])],[mtoft(pos[1,i]),mtoft(pos2[1,i])],'--k',linewidth=lw*.5,label='Equal Time from Release') 
#plt.scatter(mtoft(dsts[disti]),2,50)
plt.xlim([0,mtoft(dsts[disti]+10)])
plt.ylim([0,15])
plt.grid('on')
plt.show()
#%% plot one dist
fs=18
fs2=20
sz=120
disti=10
difr=timeb-time
fig = plt.figure()
#axs = fig.add_subplot(111, projection='3d')
j=disti
for i in range(len(spds)):
    if error[i,j]<=.01 and errorb[i,j]<=.01:
        plt.scatter(mph(spds[i]),time[i,j],s=sz,color='b')
        plt.scatter(mph(spds[i]),timeb[i,j],s=sz,color='r')
plt.scatter(mph(spds[i]),time[i,j],s=sz,color='b',label='Air Throw')
plt.scatter(mph(spds[i]),timeb[i,j],s=sz,color='r',label='Bounce Throw')
plt.xlim([60,100])
plt.grid('on')
#plt.legend(scatterpoints=1,fontsize=fs)
plt.xlabel('Throw Speed (mph)',fontsize=fs2)
plt.ylabel('Release to First Base Time (s)',fontsize=fs2)
plt.title('Throw Distance - 95 Feet',fontsize=fs2)
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.tight_layout()
plt.savefig('ThrowDistance-95Feet.png',dpi=600)
plt.show()
#%%
print(difr[:,j])
#%%
bd=90
fst=np.array([bd*np.cos(np.pi/4), bd*np.sin(np.pi/4)])
snd=fst+np.array([-bd*np.cos(np.pi/4), bd*np.sin(np.pi/4)])
trd=snd+np.array([-bd*np.cos(np.pi/4), -bd*np.sin(np.pi/4)])

plt.figure()
plt.plot(0,0,'ob')
plt.plot(fst[0],fst[1],'ob')
plt.plot(snd[0],snd[1],'ob')
plt.plot(trd[0],trd[1],'ob')
plt.axis('square')

#%%













#%%
import matplotlib as mpl
import matplotlib.cm as cm

norm = mpl.colors.Normalize(vmin=np.amin(difr), vmax=np.amax(difr))
cmap = cm.cool
x = difr[4,5]

m = cm.ScalarMappable(norm=norm, cmap=cmap)
print(m.to_rgba(x))
#%%%%
#%% first order
spdo=40.2336
rpm=0
rpm=mphtorpm(mph(spdo))
w=rpm/60*2*pi
ang=0*pi/180 
vo=np.array([spdo*np.cos(ang),spdo*np.sin(ang)])
v=np.zeros((2,len(t)))
v[:,0]=vo
a=v.copy()
xo=np.array([0,1.75])
x=a.copy()
x[:,0]=xo
ey=0.55
ex=0.7
mu=0.4
bnc=False

for n,tn in enumerate(t[0:-1]):
    #if (x[1,n-1]+(x[1,n]-x[1,n-1])) <= 0 and bnc is False:
    if (x[1,n]+v[1,n]*h) <= 0 and bnc is False:
        vn=v[:,n]
        vy1=vn[1]
        vy2=-vy1*ey
        vn[1]=vy2
        vx1=vn[0]
        vx2=vx1+mu*(1+ey)*vy1
        vn[0]=vx2
        w1=w
        w=-w-(1/.4)*mu*(1+ey)*vy1/R #reverse for sign convention from Cross
        w2=-w
        #w=rpm/60*2*pi
        Sf=SpnRat(vn,w)
        
        bnc=True
    else:
        vn=v[:,n]
        Sf=SpnRat(vn,w)
    an=np.array([ax(vn,S=Sf),ay(vn,S=Sf)])
    a[:,n]=an
    v[:,n+1]=vn+h*an
    x[:,n+1]=x[:,n]+h/2*(vn+v[:,n+1])
    
x1=x.copy()
#%% fourth order #2
spdo=mps(60)
rpm=0
rpm=mphtorpm(mph(spdo))
w=rpm/60*2*pi
ang=6*pi/180 
vo=np.array([spdo*np.cos(ang),spdo*np.sin(ang)])
v=np.zeros((2,len(t)))
v[:,0]=vo
a=v.copy()
xo=np.array([0,1.75])
x=a.copy()
x[:,0]=xo
ey=0.55
ex=0.7
mu=0.4
bnc=False
t6=0;

for n,tn in enumerate(t[0:-1]):
    #if (x[1,n-1]+(x[1,n]-x[1,n-1])) <= 0 and bnc is False:
    if (x[1,n]+v[1,n]*h) <= 0 and bnc is False:
        vn=v[:,n]
        vy1=vn[1]
        vy2=-vy1*ey
        vn[1]=vy2
        vx1=vn[0]
        vx2=vx1+mu*(1+ey)*vy1
        vn[0]=vx2
        w1=w
        w=-w-(1/.4)*mu*(1+ey)*vy1/R #reverse for sign convention from Cross
        w2=-w
        #w=rpm/60*2*pi
        Sf=SpnRat(vn,w)
        
        bnc=True
    else:
        vn=v[:,n]
        Sf=SpnRat(vn,w)
    k1=np.array([ax(vn,S=Sf),ay(vn,S=Sf)])
    k2=np.array([ax(vn+k1*h/2,S=Sf),ay(vn+k1*h/2,S=Sf)])
    k3=np.array([ax(vn+k2*h/2,S=Sf),ay(vn+k2*h/2,S=Sf)])
    k4=np.array([ax(vn+k3*h,S=Sf),ay(vn+k3*h,S=Sf)])
    an=1/6*(k1+2*k2+2*k3+k4)
    a[:,n]=an
    v[:,n+1]=vn+h*an
    x[:,n+1]=x[:,n]+h/2*(vn+v[:,n+1])
    
 
x2=x.copy()

xc=x[1,:]-2*12*.0254
zc2 = np.where(np.diff(np.sign(xc)))[0]
#%%
a=x[1,:]-0.5*.0254
zc = np.where(np.diff(np.sign(a)))[0]
#%% fourth order #3
spdo=mps(60)
#rpm=-1000
rpm=mphtorpm(mph(spdo))
w=rpm/60*2*pi
ang=1*pi/180 
vo=np.array([spdo*np.cos(ang),spdo*np.sin(ang)])
v=np.zeros((2,len(t)))
v[:,0]=vo
a=v.copy()
xo=np.array([0,1.75])
x=a.copy()
x[:,0]=xo
ey=0.55
ex=0.7
mu=0.4
bnc=False

for n,tn in enumerate(t[0:-1]):
    #if (x[1,n-1]+(x[1,n]-x[1,n-1])) <= 0 and bnc is False:
    if (x[1,n]+v[1,n]*h) <= 0 and bnc is False:
        vn=v[:,n]
        vy1=vn[1]
        vy2=-vy1*ey
        vn[1]=vy2
        vx1=vn[0]
        vx2=vx1+mu*(1+ey)*vy1
        vn[0]=vx2
        w1=w
        w=-w-(1/.4)*mu*(1+ey)*vy1/R #reverse for sign convention from Cross
        w2=-w
        #w=rpm/60*2*pi
        Sf=SpnRat(vn,w)
        
        bnc=True
    else:
        vn=v[:,n]
        Sf=SpnRat(vn,w)
    k1=np.array([ax(vn,S=Sf),ay(vn,S=Sf)])
    k2=np.array([ax(vn+k1*h/2,S=Sf),ay(vn+k1*h/2,S=Sf)])
    k3=np.array([ax(vn+k2*h/2,S=Sf),ay(vn+k2*h/2,S=Sf)])
    k4=np.array([ax(vn+k3*h,S=Sf),ay(vn+k3*h,S=Sf)])
    an=1/6*(k1+2*k2+2*k3+k4)
    a[:,n]=an
    v[:,n+1]=vn+h*an
    x[:,n+1]=x[:,n]+h/2*(vn+v[:,n+1])
    
x3=x.copy()

xc=x[1,:]-0.5*12*.0254
zc3 = np.where(np.diff(np.sign(xc)))[0]
#%%
x1=x.copy()
np.arctan(vy1/vx1)*180/np.pi
#%%
np.abs(w2)/2*pi
#%%
((vx2**2+vy2**2)**.5)/((vx1**2+vy1**2)**.5)
#%%
x2=x.copy()
#%%
lw=2
fs=16
plt.figure(figsize=(10,4))
plt.plot([0,max(x[0,:])*3.281],[0,0],'--k',linewidth=lw)
plt.plot(x2[0,:]*3.281,x2[1,:]*3.281,linewidth=lw,label='Bounce')
plt.plot(x3[0,:]*3.281,x3[1,:]*3.281,'-r',linewidth=lw,label='Air')
plt.xlim([0,150])
plt.xlabel('Horizonatal Position (ft)',fontsize=fs)
plt.ylabel('Veritcal Position (ft)',fontsize=fs)
plt.title('85 mph release speed',fontsize=fs)
plt.grid('on')
plt.ylim([0,12])
plt.tight_layout()
plt.show()
#plt.savefig('TestPlot_1.png',dpi=300)
#%%
dist=100;
lw=2.25
fs=16
plt.figure(figsize=(12,4))
plt.plot([dist,dist],[0,20],'-k',linewidth=lw,label='First Baseman')
plt.plot(x2[0,:]*3.281,x2[1,:]*3.281,linewidth=lw,label='Air Throw')
plt.plot(x3[0,:]*3.281,x3[1,:]*3.281,'-r',linewidth=lw,label='Bounce Throw')
l=range(len(t))
for i in l[0::15]:
    plt.plot([x2[0,i]*3.281,x3[0,i]*3.281],[x2[1,i]*3.281,x3[1,i]*3.281],'--k',linewidth=lw*.5)

plt.plot([x2[0,i]*3.281,x3[0,i]*3.281],[x2[1,i]*3.281,x3[1,i]*3.281],'--k',linewidth=lw*.5,label='Equal Time from Release') 
plt.xlim([0,dist+10])
plt.xlabel('Horizonatal Position (ft)',fontsize=fs)
plt.ylabel('Veritcal Position (ft)',fontsize=fs)
plt.title('60 mph Release Speed, 1590 RPM',fontsize=fs)
plt.grid('on')
plt.legend(loc='lower left')
plt.ylim([0,8])
plt.tight_layout()
plt.show()
plt.savefig('TestPlot_6.png',dpi=300)
#%%
lw=2
fs=16
plt.figure(figsize=(10,4))
plt.plot([0,max(x[0,:])*3.281],[0,0],'--k',linewidth=lw)
plt.plot(x[0,:]*3.281,x[1,:]*3.281,linewidth=lw,label='Bounce Throw')
#plt.plot(x2[0,:]*3.281,x2[1,:]*3.281,'-r',linewidth=lw,label='Air Throw')
plt.xlim([0,130])
plt.xlabel('Horizonatal Position (ft)',fontsize=fs)
plt.ylabel('Veritcal Position (ft)',fontsize=fs)
#plt.title('70 mph release speed',fontsize=fs)
plt.grid('on')
plt.ylim([0,12])
plt.tight_layout()
plt.show()
#%%
lw=2
fs=16
plt.figure(figsize=(10,4))
plt.plot([0,max(x[0,:])*3.281],[0,0],'--k',linewidth=lw)
plt.plot(x1[0,:]*3.281,x1[1,:]*3.281,linewidth=lw,label='First Order')
plt.plot(x2[0,:]*3.281,x2[1,:]*3.281,'--r',linewidth=lw,label='4th order')
plt.xlim([0,150])
plt.xlabel('Horizonatal Position (ft)',fontsize=fs)
plt.ylabel('Veritcal Position (ft)',fontsize=fs)
plt.title('70 mph release speed',fontsize=fs)
plt.grid('on')
plt.ylim([0,12])
plt.tight_layout()
plt.show()
plt.savefig('TestPlot.png',dpi=300)
#%%
plt.figure()
plt.plot(x[0,:]*3.281,x2[0,:]*3.281)
plt.plot([0,150],[0,150],'--k')

plt.xlim([0,150])
plt.axis('square')

plt.show()
 #%%
plt.figure()
plt.plot(t,x[0,:]*3.281)
plt.plot(t,x2[0,:]*3.281)
plt.show()  

#%% Sim speeds
spds=np.linspace(25,45,50)
angs=np.linspace(-4,20,50)*pi/180
#%%
v=np.zeros((2,len(t),len(spds),len(angs))) # first dim x.y second t, third initial speed.
a=v.copy()
x2=a.copy()
for k, ango in enumerate(angs):
    for q,spdo in enumerate(spds):
        spdo=spdo
        rpm=mphtorpm(mph(spdo))
        w=rpm/60*2*pi
        Sf=R*w/spdo
        ang=ango
        #ang=2*pi/180
        vo=np.array([spdo*np.cos(ang),spdo*np.sin(ang)])
        #v=np.zeros((2,len(t)))
        v[:,0,q,k]=vo
    
        xo=np.array([0,1.75])
    
        x2[:,0,q,k]=xo
        ey=0.55
        ex=0.7
        mu=0.4
        bnc=False
        
        for n,tn in enumerate(t[0:-1]):
            if (x2[1,n-1,q,k]+(x2[1,n,q,k]-x2[1,n-1,q,k])) <= 0 and bnc is False:
                vn=v[:,n,q,k]
                #vn[1]=-vn[1]*ey
                vy1=vn[1]
                vy2=-vy1*ey
                vn[1]=vy2
                vx1=vn[0]
                vx2=vx1+mu*(1+ey)*vy1
                vn[0]=vx2
                w1=w
                w=-w-(1/.4)*mu*(1+ey)*vy1/R #reverse for sign convention from Cross
                w2=-w
                #w=rpm/60*2*pi
                Sf=SpnRat(vn,w)                
                
                bnc=True
            else:
                vn=v[:,n,q,k]
                Sf=SpnRat(vn,w)
                
            k1=np.array([ax(vn,S=Sf),ay(vn,S=Sf)])
            k2=np.array([ax(vn+k1*h/2,S=Sf),ay(vn+k1*h/2,S=Sf)])
            k3=np.array([ax(vn+k2*h/2,S=Sf),ay(vn+k2*h/2,S=Sf)])
            k4=np.array([ax(vn+k3*h,S=Sf),ay(vn+k3*h,S=Sf)])
            an=1/6*(k1+2*k2+2*k3+k4)    
            #an=np.array([ax(vn,S=Sf),ay(vn,S=Sf)])
            a[:,n,q,k]=an
            v[:,n+1,q,k]=vn+h*an
            x2[:,n+1,q,k]=x2[:,n,q,k]+h/2*(vn+v[:,n+1,q,k])
    
np.save('AllSims_3',x2)
#%%
x2=np.load('AllSims_3.npy')
#%%
spdin=4
dist=130
 #%%
plt.figure()
plt.plot(x2[0,:,spdin,:]*3.281,x2[1,:,spdin,:]*3.281)
plt.scatter(dist,0.5,s=50,c='r')
plt.scatter(dist,2,s=50,c='b')
plt.show() 
#%%
plt.figure()
plt.plot(v[0,:,spdin,:]*3.281,v[1,:,spdin,:]*3.281)
plt.scatter(dist,0.5,s=50,c='r')
plt.scatter(dist,2,s=50,c='b')
plt.show() 
#%%
dist=127
fly=[fttom(dist),fttom(2)]
bounce=[fttom(dist),fttom(0.5)]
tols=[fttom(0.1/12),fttom(2/12)]
#bounceloctimes=np.zeros(2,1,50) #x,y,t, speed
#flyloctimes=np.zeros(2,1,50) #x,y,t, speed
fromfly=np.zeros(x2.shape)
frombounce=fromfly.copy()
#%%
dind=np.zeros((1,1,50,50))
td=np.zeros((1,1,50,50))
xd=np.zeros((1,1,50,50))
yd=np.zeros((1,1,50,50))
dindb=np.zeros((1,1,50,50))
tdb=np.zeros((1,1,50,50))
xdb=np.zeros((1,1,50,50))
ydb=np.zeros((1,1,50,50))
d=fttom(100)
for k in range(50):
    for j in range(50):
        di=np.argmin(abs(x2[0,:,j,k]-d))
        if v[1,di,j,k]<0:
            dind[:,:,j,k]=di
            td[:,:,j,k]=t[di]
            xd[:,:,j,k]=x2[0,di,j,k]
            yd[:,:,j,k]=x2[1,di,j,k]
        else:
            dindb[:,:,j,k]=di
            tdb[:,:,j,k]=t[di]
            xdb[:,:,j,k]=x2[0,di,j,k]
            ydb[:,:,j,k]=x2[1,di,j,k]
            
#dindb=dindb[dind !=0]
#%% use this
dind=np.zeros((50,50))
td=np.zeros((50,50))
xd=np.zeros((50,50))
yd=np.zeros((50,50))
dindb=np.zeros((50,50))
tdb=np.zeros((50,50))
xdb=np.zeros((50,50))
ydb=np.zeros((50,50))
d=fttom(127)

bix=np.zeros(50,dtype=np.int64)
by=np.zeros(50)
bt=np.zeros(50)
aix=np.zeros(50,dtype=np.int64)
ay=np.zeros(50)
at=np.zeros(50)
for j in range(50):
    for k in range(50):
        di=np.argmin(abs(x2[0,:,j,k]-d))
        if (x2[1,di,j,k]-x2[1,di-1,j,k])<0:
            dind[j,k]=di
            td[j,k]=t[di]
            xd[j,k]=x2[0,di,j,k]
            yd[j,k]=x2[1,di,j,k]
        else:
            dindb[j,k]=di
            tdb[j,k]=t[di]
            xdb[j,k]=x2[0,di,j,k]
            ydb[j,k]=x2[1,di,j,k]
            
for j in range(50):
    by[j],bix[j] = findnearest(ydb[j,:],fttom(0.5))
    bt[j]=tdb[j,bix[j]]
    
    ay[j],aix[j] = findnearest(yd[j,:],fttom(2))
    at[j]=td[j,aix[j]]
#%%
spdi0=0
plt.figure()
plt.plot(mph(spds[spdi0:]),at[spdi0:],'ob')
plt.plot(mph(spds[spdi0:]),bt[spdi0:],'or')
#%%
from scipy import ndimage
#%%
spdi0=6
plt.figure()
plt.plot(mph(spds),at-bt,'ob')
plt.plot(mph(spds[spdi0:]),ndimage.filters.gaussian_filter1d(at[spdi0:]-bt[spdi0:],3),'-b')
plt.plot(mph(spds[spdi0:]),ndimage.filters.gaussian_filter1d(at[spdi0:],3)-ndimage.filters.gaussian_filter1d(bt[spdi0:],3),'--r')
plt.grid('on')
#plt.plot(spds,bt,'or')
#%%
j=13
plt.figure()
plt.plot(mtoft(x2[0,:,j,bix[j]]),mtoft(x2[1,:,j,bix[j]]),'r')
plt.plot(mtoft(x2[0,:,j,aix[j]]),mtoft(x2[1,:,j,aix[j]]),'b')
plt.scatter(mtoft(d),0.5,s=50,c='r')
plt.scatter(mtoft(d),2,s=50,c='b')
plt.xlim([0,mtoft(d)+10])
plt.ylim([0,10])
#%%
j=30
lw=2.25
fs=16
plt.figure(figsize=(12,4))
#plt.plot([dist,dist],[0,20],'-k',linewidth=lw,label='First Baseman')
plt.plot(mtoft(x2[0,:,j,aix[j]]),mtoft(x2[1,:,j,aix[j]]),'b',label='Air Throw')
plt.plot(mtoft(x2[0,:,j,bix[j]]),mtoft(x2[1,:,j,bix[j]]),'r',label='Bounce Throw')
#plt.plot(mtoft(x2[0,:,j,aix[j]]),mtoft(x2[1,:,j,aix[j]]),'b')
#plt.plot(x2[0,:]*3.281,x2[1,:]*3.281,linewidth=lw,label='Air Throw')
#plt.plot(x3[0,:]*3.281,x3[1,:]*3.281,'-r',linewidth=lw,label='Bounce Throw')
l=range(len(t))
for i in l[0::40]:
    plt.plot([mtoft(x2[0,i,j,bix[j]]),mtoft(x2[0,i,j,aix[j]])],[mtoft(x2[1,i,j,bix[j]]),mtoft(x2[1,i,j,aix[j]])],'--k',linewidth=lw*.5)

plt.plot([mtoft(x2[0,i,j,bix[j]]),mtoft(x2[0,i,j,aix[j]])],[mtoft(x2[1,i,j,bix[j]]),mtoft(x2[1,i,j,aix[j]])],'--k',linewidth=lw*.5,label='Equal Time from Release')
plt.scatter(mtoft(d),2,s=50,c='b',label='Air Target') 
plt.scatter(mtoft(d),0.5,s=50,c='r',label='Bounce Target')
plt.xlim([0,mtoft(d)+10])
plt.xlabel('Horizonatal Position (ft)',fontsize=fs)
plt.ylabel('Veritcal Position (ft)',fontsize=fs)
plt.title(str(np.around(mph(spds[j]),1))+' mph Release Speed',fontsize=fs)
plt.grid('on')
plt.legend(loc='upper right',scatterpoints=1)
plt.ylim([0,15])
plt.tight_layout()
plt.show()
plt.savefig('FinalPlot_3_MidMid.png',dpi=300)
#%%
#%% use this for all dists
dind=np.zeros((50,50))
td=np.zeros((50,50))
xd=np.zeros((50,50))
yd=np.zeros((50,50))
dindb=np.zeros((50,50))
tdb=np.zeros((50,50))
xdb=np.zeros((50,50))
ydb=np.zeros((50,50))
ds=fttom(np.array([90,127]))
# fttom(np.linspace(70,127,20))

atall=np.zeros([len(ds),50])
btall=np.zeros([len(ds),50])
atallsm=np.zeros([len(ds),50])
btallsm=np.zeros([len(ds),50])
diffsm=np.zeros([len(ds),50])
spdi0=6
diffsm=np.zeros([len(ds),len(spds[spdi0:])])
for i in range(len(ds)):
    d = ds[i]
    bix=np.zeros(50,dtype=np.int64)
    by=np.zeros(50)
    bt=np.zeros(50)
    aix=np.zeros(50,dtype=np.int64)
    ay=np.zeros(50)
    at=np.zeros(50)
    for j in range(50):
        for k in range(50):
            di=np.argmin(abs(x2[0,:,j,k]-d))
            if (x2[1,di,j,k]-x2[1,di-1,j,k])<0:
                dind[j,k]=di
                td[j,k]=t[di]
                xd[j,k]=x2[0,di,j,k]
                yd[j,k]=x2[1,di,j,k]
            else:
                dindb[j,k]=di
                tdb[j,k]=t[di]
                xdb[j,k]=x2[0,di,j,k]
                ydb[j,k]=x2[1,di,j,k]
                
    for j in range(50):
        by[j],bix[j] = findnearest(ydb[j,:],fttom(0.5))
        bt[j]=tdb[j,bix[j]]
        
        ay[j],aix[j] = findnearest(yd[j,:],fttom(2))
        at[j]=td[j,aix[j]]
    atall[i,:]=at
    btall[i,:]=bt
    diffsm[i,:]=ndimage.filters.gaussian_filter1d(atall[i,spdi0:]-btall[i,spdi0:],3)
    
#%%
distin=1
plt.figure()
plt.plot(mph(spds[spdi0:]),atall[distin,spdi0:],'ob')
plt.plot(mph(spds[spdi0:]),btall[distin,spdi0:],'or')
#plt.plot(mph(spds[spdi0:]),diffsm[distin,:],'b')
#%%
plt.figure()
for i in range(len(ds)):
    plt.plot(mph(spds[spdi0:]),diffsm[i,:],'-b')
#%%
spdin=40
plt.figure()
#plt.plot(xd[:,:,spdin,:].flatten(),yd[:,:,spdin,:].flatten())
plt.plot(yd[spdin,:],td[spdin,:],'ob')
plt.plot(ydb[spdin,:],tdb[spdin,:],'or')

#plt.xlim([d-1,d+1])

#%%
testval, testix = findnearest(ydb[spdin,:],fttom(0.5))
#%%
j=20
for k in range(50):
    for j in range(50):
        for i in range(len(t)):
            fromfly[:,i,j,k]=x2[:,i,j,k]-fly
            frombounce[:,i,j,k]=x2[:,i,j,k]-fly
            
for k in range(50):
    for j in range(50):
        for i in range(len(t)-1):
            if abs(fromfly[0,i-,j,k])<abs(fromfly[0,i-1,j,k]) and abs(fromfly[0,i-,j,k])<abs(fromfly[0,i+1,j,k]) and
#%%
spdin=14
plt.figure()
plt.semilogy(t,fromfly[1,:,spdin,:]**2+fromfly[0,:,spdin,:]**2)
#%%
xd=x2[0,:,spdin,:]-dist/3.281
yd=x2[1,:,spdin,:]-2/3.281
md=xd**2+yd**2
#%%
plt.figure()
plt.plot(x2[0,:,spdin,:]*3.281,x2[1,:,spdin,:]*3.281)
plt.show() 
#%%
plt.figure()
plt.semilogy(t,md)
 #%%
plt.figure()
plt.plot(t,x[0,:]-x2[0,:])
plt.show()  
#%% min 200 pitches, average speed and spin rate
SpdSpn=np.load('SpdSpn.npy')
Spd=SpdSpn.copy()
Spd[:,1]=Spd[:,1]*0+1

beta=np.linalg.lstsq(Spd,SpdSpn[:,1])[0]
print(beta)

Spnmod=np.dot(Spd,beta)
spdfit=np.linspace(60,120,100)
spnfit=spdfit*beta[0]+beta[1]
#%%
R2=np.square(np.corrcoef(SpdSpn[:,1].squeeze(),Spnmod.squeeze())[0,1])
#%%
fs=18
fs2=20
plt.figure()
plt.scatter(SpdSpn[:,0],SpdSpn[:,1],s=50)
plt.plot(spdfit,spnfit,'--k',linewidth=2)
plt.axis([80,105,1500,3000])
plt.grid('on')
plt.xlabel('Average Fastball Release Speed (mph)',fontsize=fs2)
plt.ylabel('Average Spin Rate (rpm)',fontsize=fs2)
plt.title('2017 Pitchers, Minimum 200 Fastballs',fontsize=fs2)
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.tight_layout()
plt.savefig('SpeedAndSpin.png',dpi=600)


#%%
plt.close('all')