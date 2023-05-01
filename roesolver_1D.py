import numpy as np

simtype = "gauss"
boundaryType = "mirror"

a = 0
b = 100
N = 500 
dx = (b-a)/N

nGhost = 2 
# N cells + 2*nghost ghost cells
xVals = np.linspace(a-(nGhost+1)*dx/2, b+(nGhost+1)*dx/2, N+2*nGhost)
# N+1 boundaries + 2*nghost boundaries
xEdgeVals = np.linspace(a-nGhost*dx, b+nGhost*dx, N+1+2*nGhost)

gamma = 1.4

dt = 0.25
Nt = 5000
ti = 0
tf = Nt*dt

# Definition of State Variables:
# q0 = rho*etot
# q1 = rho*u
# q2 = rho*v
# q3 = rho*w
# q4 = rho

def eFromP(P, rho):
    etot = P/(rho*(gamma-1))

    return etot


def q0Init(xVals, shape):
    if shape == "gauss":
        xmid = 0.5*(b+a)
        sigrho = 0.1*(b-a)
        rho = (1 + 0.3*np.exp(-(xVals - xmid)**2 / sigrho**2))
        sigP = 0.1*(b-a)
        P = 1 + np.exp(-(xVals - xmid)**2 / sigP**2)
        vals = eFromP(P, rho)

    elif shape == "shock":
        PL = 5
        PR = 0.1
        vals = eFromP(PR, 0.125*np.ones_like(xVals))
        vals[0:N//2] = eFromP(PL, 1)

    return vals


def q1Init(xVals):
    vals = np.zeros_like(xVals)

    return vals


def q2Init(xVals):
    vals = np.zeros_like(xVals)

    return vals


def q3Init(xVals):
    vals = np.zeros_like(xVals)

    return vals


def q4Init(xVals, shape):
    if shape == "gauss":
        xmid = 0.5*(b+a)
        dg = 0.1*(b-a)
        vals = (1 + 0.3*np.exp(-(xVals - xmid)**2 / dg**2))

    elif shape == "shock":
        vals = 0.125*np.ones_like(xVals)
        vals[0:N//2] = 1

    return vals


def boundaryConds(q0, q1, q2, q3, q4, type):
    # could definitely refactor this to be cleaner
    midq0 = q0[2:-2]
    midq1 = q1[2:-2]
    midq2 = q2[2:-2]
    midq3 = q3[2:-2]
    midq4 = q4[2:-2]

    if type == "periodic":
        q0[0:2] = q0[-4:-2]
        q0[-2:] = q0[2:4]

        q1[0:2] = q1[-4:-2]
        q1[-2:] = q1[2:4]

        q2[0:2] = q2[-4:-2]
        q2[-2:] = q2[2:4]

        q3[0:2] = q3[-4:-2]
        q3[-2:] = q3[2:4]

        q4[0:2] = q4[-4:-2]
        q4[-2:] = q4[2:4]
        
        
    elif type == "mirror":
        q0 = np.pad(midq0, (2, 2), "reflect")
        q1[0] = -q1[3]
        q1[1] = -q1[2]
        q1[-2] = -q1[-3]
        q1[-1] = -q1[-4]
        q2 = np.pad(midq2, (2, 2), "reflect")
        q3 = np.pad(midq3, (2, 2), "reflect")
        q4 = np.pad(midq4, (2, 2), "reflect")

    elif type == "inoutflow":
        q0 = np.pad(midq0, (2, 2), "edge")
        q1 = np.pad(midq1, (2, 2), "edge")
        q2 = np.pad(midq2, (2, 2), "edge")
        q3 = np.pad(midq3, (2, 2), "edge")
        q4 = np.pad(midq4, (2, 2), "edge")

    elif type == "outflow":
        q0 = np.pad(midq0, (2, 2), "edge")
        q1[0:2] = -np.abs(q1[2])
        q1[-2:] = np.abs(q1[-3])
        q2 = np.pad(midq2, (2, 2), "edge")
        q3 = np.pad(midq3, (2, 2), "edge")
        q4 = np.pad(midq4, (2, 2), "edge")


    return q0, q1, q2, q3, q4


def timeStep(xEdgeVals, l1, l2, l3, l4, l5):

    maxl = np.maximum(l1, np.maximum(l2, np.maximum(l3, np.maximum(l4, l5))))
    minl = np.minimum(l1, np.minimum(l2, np.minimum(l3, np.minimum(l4, l5))))

    dti = (xEdgeVals[1:-1] - xEdgeVals[:-2])/(maxl - minl)

    dt = np.min(dti)
    
    return dt


def minmod(a, b):
    aMask = (np.abs(a) < np.abs(b)) & (a*b > 0)
    bMask = (np.abs(b) < np.abs(a)) & (a*b > 0)

    diff = np.zeros_like(a)

    diff[aMask] = a[aMask]
    diff[bMask] = b[bMask]

    return diff


def superbee(r):
    vals = np.maximum(0, np.maximum(np.minimum(1, 2*r), np.minimum(2, r)))

    return vals


def qStates(qi):
    qL = qi[:-1]
    qR = qi[1:]

    dq = qR - qL

    return qL, qR, dq


def roeAverages(q0L, q1L, q2L, q3L, q4L, q0R, q1R, q2R, q3R, q4R):
    rhoL = q4L
    rhoR = q4R

    uL = q1L/q4L
    uR = q1R/q4R
    vL = q2L/q4L
    vR = q2R/q4R
    wL = q3L/q4L
    wR = q3R/q4R

    etotL = q0L/q4L
    etotR = q0R/q4R

    ekinL = (uL**2 + vL**2 + wL**2)/2
    ekinR = (uR**2 + vR**2 + wR**2)/2

    ethL = etotL - ekinL
    ethR = etotR - ekinR

    pL = (gamma-1)*rhoL*ethL
    pR = (gamma-1)*rhoR*ethR

    htotL = etotL + pL/rhoL
    htotR = etotR + pR/rhoR

    sqrtRhoL = np.sqrt(rhoL)
    sqrtRhoR = np.sqrt(rhoR)
    denominator = sqrtRhoL + sqrtRhoR

    uHat = (sqrtRhoL*uL + sqrtRhoR*uR)/denominator
    vHat = (sqrtRhoL*vL + sqrtRhoR*vR)/denominator
    wHat = (sqrtRhoL*wL + sqrtRhoR*wR)/denominator
    htotHat = (sqrtRhoL*htotL + sqrtRhoR*htotR)/denominator

    ekinHat = (uHat**2 + vHat**2 + wHat**2)/2

    CsHat = np.sqrt((gamma-1)*(htotHat - ekinHat))

    return uHat, vHat, wHat, htotHat, ekinHat, CsHat


def dqTildas(uHat, vHat, wHat, htotHat, CsHat, ekinHat, dq0, dq1, dq2, dq3, dq4):
    zeta = uHat*dq1 + vHat*dq2 + wHat*dq3 - dq0

    a = 0.5*(gamma-1)*(ekinHat*dq4 - zeta)/CsHat**2
    b = 0.5*(dq1 - uHat*dq4)/CsHat

    dq1Tilda = a - b

    dq2Tilda = a + b

    dq3Tilda = 0.5*(gamma-1)*((htotHat - 2*ekinHat)*dq4 + zeta)/CsHat**2

    dq4Tilda = dq2 - vHat*dq4

    dq5Tilda = dq3 - wHat*dq4

    return dq1Tilda, dq2Tilda, dq3Tilda, dq4Tilda, dq5Tilda


def eigenValues(uHat, CsHat):
    l1 = uHat - CsHat
    l2 = uHat + CsHat
    l3 = uHat
    l4 = uHat
    l5 = uHat

    return l1, l2, l3, l4, l5


def e1eigen(uHat, vHat, wHat, htotHat, CsHat):
    e0 = htotHat - CsHat*uHat

    e1 = uHat - CsHat

    e2 = vHat

    e3 = wHat

    e4 = np.ones_like(uHat)

    return e0, e1, e2, e3, e4


def e2eigen(uHat, vHat, wHat, htotHat, CsHat):
    e0 = htotHat + CsHat*uHat

    e1 = uHat + CsHat

    e2 = vHat

    e3 = wHat

    e4 = np.ones_like(uHat)

    return e0, e1, e2, e3, e4


def e3eigen(uHat, vHat, wHat, htotHat, CsHat):
    e0 = 0.5*uHat**2

    e1 = uHat

    e2 = vHat

    e3 = wHat

    e4 = np.ones_like(uHat)

    return e0, e1, e2, e3, e4


def e4eigen(uHat, vHat, wHat, htotHat, CsHat):
    e0 = vHat**2

    e1 = np.zeros_like(uHat)

    e2 = np.ones_like(uHat)

    e3 = np.zeros_like(uHat)

    e4 = np.zeros_like(uHat)

    return e0, e1, e2, e3, e4


def e5eigen(uHat, vHat, wHat, htotHat, CsHat):
    e0 = wHat**2

    e1 = np.zeros_like(uHat)

    e2 = np.zeros_like(uHat)

    e3 = np.ones_like(uHat)

    e4 = np.zeros_like(uHat)

    return e0, e1, e2, e3, e4


def rTildas(dqTilda, l):
    gtMask = l[1:-1] >= 0
    ltMask = l[1:-1] <= 0

    leftdq = dqTilda[:-2]
    rightdq = dqTilda[2:]
    middq = dqTilda[1:-1]
    zeroMask = middq != 0

    r = np.zeros_like(middq)

    r[gtMask & zeroMask] = leftdq[gtMask & zeroMask]/middq[gtMask & zeroMask]
    r[ltMask & zeroMask] = rightdq[ltMask & zeroMask]/middq[ltMask & zeroMask]

    r = np.pad(r, (1, 1), "edge")

    return r


def thetaVals(l):
    lMask = l < 0
    theta = np.ones_like(l)
    theta[lMask] = -1

    return theta


def unlimitedFluxes(rho, htot, u, v, w, P):
    fx0 = rho*htot*u
    fx1 = rho*u**2 + P
    fx2 = rho*v*u
    fx3 = rho*w*u
    fx4 = rho*u

    return fx0, fx1, fx2, fx3, fx4


def consUpdate(vals, flux, dt, xEdgeVals):
    valsOut = np.zeros_like(vals)
    valsOut[2:-2] = vals[2:-2] - dt*(flux[2:-1] - flux[1:-2])/(xEdgeVals[3:-2] - xEdgeVals[2:-3])

    return valsOut

def hydrostep(q0, q1, q2, q3, q4, xVals, xEdgeVals, limtype):
    # most of this could be made more readable with a for loop. should function the same tho?
    
    # Extract variables
    rho = q4
    etot = q0/rho
    u = q1/rho
    v = q2/rho
    w = q3/rho
    P = (gamma-1)*rho*(etot - (u**2 + v**2 + w**2)/2)
    htot = etot + P/rho

    #Define left/right states and differences
    q0L, q0R, dq0 = qStates(q0)
    q1L, q1R, dq1 = qStates(q1)
    q2L, q2R, dq2 = qStates(q2)
    q3L, q3R, dq3 = qStates(q3)
    q4L, q4R, dq4 = qStates(q4)

    # calculate roe averages
    uHat, vHat, wHat, htotHat, ekinHat, CsHat = roeAverages(q0L, q1L, q2L, q3L, q4L, q0R, q1R, q2R, q3R, q4R)
    
    # calculate dTilda values
    dq1Tilda, dq2Tilda, dq3Tilda, dq4Tilda, dq5Tilda = dqTildas(uHat, vHat, wHat, htotHat, CsHat, ekinHat, dq0, dq1, dq2, dq3, dq4)

    # calculate eigenvectors
    e10, e11, e12, e13, e14 = e1eigen(uHat, vHat, wHat, htotHat, CsHat)
    e20, e21, e22, e23, e24 = e2eigen(uHat, vHat, wHat, htotHat, CsHat)
    e30, e31, e32, e33, e34 = e3eigen(uHat, vHat, wHat, htotHat, CsHat)
    e40, e41, e42, e43, e44 = e4eigen(uHat, vHat, wHat, htotHat, CsHat)
    e50, e51, e52, e53, e54 = e5eigen(uHat, vHat, wHat, htotHat, CsHat)

    # calculate eigenvalues
    l1, l2, l3, l4, l5 = eigenValues(uHat, CsHat)

    # calculate dt
    dt = timeStep(xEdgeVals, l1, l2, l3, l4, l5)

    # calculate r_{i-1/2}
    r1 = rTildas(dq1Tilda, l1)
    r2 = rTildas(dq2Tilda, l2)
    r3 = rTildas(dq3Tilda, l3)
    r4 = rTildas(dq4Tilda, l4)
    r5 = rTildas(dq5Tilda, l5)

    # Get phi (limited slopes)
    if limtype == "donor":
        phi1 = np.zeros_like(r1)
        phi2 = np.zeros_like(r2)
        phi3 = np.zeros_like(r3)
        phi4 = np.zeros_like(r4)
        phi5 = np.zeros_like(r5)
    elif limtype == "minmod":
        phi1 = minmod(np.ones_like(r1), r1)
        phi2 = minmod(np.ones_like(r2), r2)
        phi3 = minmod(np.ones_like(r3), r3)
        phi4 = minmod(np.ones_like(r4), r4)
        phi5 = minmod(np.ones_like(r5), r5)
    elif limtype == "superbee":
        phi1 = superbee(r1)
        phi2 = superbee(r2)
        phi3 = superbee(r3)
        phi4 = superbee(r4)
        phi5 = superbee(r5)

    # Get theta vals (plus or minus 1, depending on eigenvalue)
    theta1 = thetaVals(l1)
    theta2 = thetaVals(l2)
    theta3 = thetaVals(l3)
    theta4 = thetaVals(l4)
    theta5 = thetaVals(l5)

    # Construct unlimited flux
    uf0, uf1, uf2, uf3, uf4 = unlimitedFluxes(rho, htot, u, v, w, P)

    # Construct total flux
    # I made the sign after the first theta negative. Is this correct???
    temp1 = theta1 - phi1*(l1*dt/(xVals[1:]-xVals[:-1]) - theta1)
    temp2 = theta2 - phi2*(l2*dt/(xVals[1:]-xVals[:-1]) - theta2)
    temp3 = theta3 - phi3*(l3*dt/(xVals[1:]-xVals[:-1]) - theta3)
    temp4 = theta4 - phi4*(l4*dt/(xVals[1:]-xVals[:-1]) - theta4)
    temp5 = theta5 - phi5*(l5*dt/(xVals[1:]-xVals[:-1]) - theta5)

    f0 = 0.5*(uf0[1:] + uf0[:-1]) - 0.5*(l1*dq1Tilda*e10*temp1 + \
                                         l2*dq2Tilda*e20*temp2 + \
                                         l3*dq3Tilda*e30*temp3 + \
                                         l4*dq4Tilda*e40*temp4 + \
                                         l5*dq5Tilda*e50*temp5)

    f1 = 0.5*(uf1[1:] + uf1[:-1]) - 0.5*(l1*dq1Tilda*e11*temp1 + \
                                         l2*dq2Tilda*e21*temp2 + \
                                         l3*dq3Tilda*e31*temp3 + \
                                         l4*dq4Tilda*e41*temp4 + \
                                         l5*dq5Tilda*e51*temp5)

    f2 = 0.5*(uf2[1:] + uf2[:-1]) - 0.5*(l1*dq1Tilda*e12*temp1 + \
                                         l2*dq2Tilda*e22*temp2 + \
                                         l3*dq3Tilda*e32*temp3 + \
                                         l4*dq4Tilda*e42*temp4 + \
                                         l5*dq5Tilda*e52*temp5)

    f3 = 0.5*(uf3[1:] + uf3[:-1]) - 0.5*(l1*dq1Tilda*e13*temp1 + \
                                         l2*dq2Tilda*e23*temp2 + \
                                         l3*dq3Tilda*e33*temp3 + \
                                         l4*dq4Tilda*e43*temp4 + \
                                         l5*dq5Tilda*e53*temp5)

    f4 = 0.5*(uf4[1:] + uf4[:-1]) - 0.5*(l1*dq1Tilda*e14*temp1 + \
                                         l2*dq2Tilda*e24*temp2 + \
                                         l3*dq3Tilda*e34*temp3 + \
                                         l4*dq4Tilda*e44*temp4 + \
                                         l5*dq5Tilda*e54*temp5)

    # Update cells, but not ghost cells
    newq0 = consUpdate(q0, f0, dt, xEdgeVals)
    newq1 = consUpdate(q1, f1, dt, xEdgeVals)
    newq2 = consUpdate(q2, f2, dt, xEdgeVals)
    newq3 = consUpdate(q3, f3, dt, xEdgeVals)
    newq4 = consUpdate(q4, f4, dt, xEdgeVals)

    return newq0, newq1, newq2, newq3, newq4, dt


q0Vals = np.zeros((N+2*nGhost, Nt+1))
q0Vals[:, 0] = q0Init(xVals, simtype)

q1Vals = np.zeros((N+2*nGhost, Nt+1))
q1Vals[:, 0] = q1Init(xVals)

q2Vals = np.zeros((N+2*nGhost, Nt+1))
q2Vals[:, 0] = q2Init(xVals)

q3Vals = np.zeros((N+2*nGhost, Nt+1))
q3Vals[:, 0] = q3Init(xVals)

q4Vals = np.zeros((N+2*nGhost, Nt+1))
q4Vals[:, 0] = q4Init(xVals, simtype)

i = 0
tVals = np.zeros(Nt+1)
tVals[0] = ti

while i < Nt:
    q0 = q0Vals[:, i]
    q1 = q1Vals[:, i]
    q2 = q2Vals[:, i]
    q3 = q3Vals[:, i]
    q4 = q4Vals[:, i]

    q0_2, q1_2, q2_2, q3_2, q4_2, dt = hydrostep(q0, q1, q2, q3, q4, xVals, xEdgeVals, "superbee")

    q0_3, q1_3, q2_3, q3_3, q4_3 = boundaryConds(q0_2, q1_2, q2_2, q3_2, q4_2, boundaryType)

    q0Vals[:, i+1] = q0_3
    q1Vals[:, i+1] = q1_3
    q2Vals[:, i+1] = q2_3
    q3Vals[:, i+1] = q3_3
    q4Vals[:, i+1] = q4_3

    tVals[i+1] = tVals[i] + dt

    i += 1


fileName = "roe1D_" + simtype + "_data"

np.savez(fileName, q0=q0Vals, q1=q1Vals, q2=q2Vals, q3=q3Vals, q4=q4Vals, x=xVals, \
         a=a, b=b, Nx=N, Nt=Nt, gamma=gamma, t=tVals)