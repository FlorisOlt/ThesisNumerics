import numpy as np
import scipy.optimize as spo
import scipy.special as sps
import matplotlib.pyplot as plt
import math as m
import os
import warnings
import sys
warnings.filterwarnings("error")
os.chdir("C:\\Users\\Flori\\Documents\\Master Thesis Code")
import DataGeneratorNew as dgNew

## Code created by Floris Olthuis

## Choosing parameters to which the model must correspond and their relevant derived quantities for the model:
a0 = 1
b0 = 1
truebeta1 = 1
p = 150
truebeta2 = np.ones(p)
S2 = np.sqrt(np.sum(np.square(truebeta2))/p)
u_start = 0.1
v_start = 0.1
w_start = 0.99*S2
beta1_start = 0.999*truebeta1
beta2_start = 0.999*truebeta2
a_start = 0.999*a0
b_start = 0.999*b0
zeta_start = 0.1

## Amount of samples for Monte Carlo
M = 1000000

## We want to be able to generate samples from our Cox-model distribution p(t|...) for our Monte Carlo estimation of the integrals,
## this is done through inversion sampling
def time_generator(u, z1, xi2):
    return np.power(-(((b0+1)/a0)*np.exp(-truebeta1*z1 - S2*xi2)*np.log(1-u)), (1/(b0+1)))

def time_generator_null(u, xi2):
    return np.power(-(((b0+1)/a0)*np.exp(-S2*xi2)*np.log(1-u)), (1/(b0+1)))

## Functions which contain the label diffS refer to a consistency check applied during coding. They are not strictly relevant for the
## final result, but are left in as they were a vital part of the process performed.
def time_generator_null_diffS(u, xi2):
    return np.power(-(((b0+1)/a0)*np.exp(-np.sqrt(S2**2 + 1)*xi2)*np.log(1-u)), (1/(b0+1)))

## For the MC simulations we want to generate samples of xi2, z, and samples of the time integral corresponding to drawings belonging to z1 = 1 and
## z1 = -1 respectively. To achieve this we draw a pair (xi2, z) from a 2D normal random variable, and to this pair we also attach the drawn times
## for both instances of z1, with the time for z1 = 1 becoming the third element of the previously generated array and z1 = -1 the third of a copied array.

## Function for the real part of the Lambert W-function
def lambertW(z):
    return sps.lambertw(z).real

## This function calculates the value of zeta based on the obtained model parameters. It is constructed from the order parameter equation for u-tilde.
def zetafuncu(u, v, w, beta1, a, b, xi_vals, z_vals, timepos, timeneg):
    dummy = 0
    exppos = (u**2)*np.exp(u**2 + beta1 + w*xi_vals + v*z_vals)*(a/(b+1))
    expneg = (u**2)*np.exp(u**2 - beta1 + w*xi_vals + v*z_vals)*(a/(b+1))
    dummy += np.mean(np.power((u**2)*np.ones(M) - lambertW(np.power(timepos, b+1)*exppos), 2))
    dummy += np.mean(np.power((u**2)*np.ones(M) - lambertW(np.power(timeneg, b+1)*expneg), 2))
    return (dummy/(2*v*v))

## This function calculates the value of zeta based on the obtained model parameters. It is constructed from the order parameter equation for v.
def zetafuncv(u, v, w, beta1, a, b, xi_vals, z_vals, timepos, timeneg):
    dummy = 0
    exppos = (u**2)*np.exp(u**2 + beta1 + w*xi_vals + v*z_vals)*(a/(b+1))
    expneg = (u**2)*np.exp(u**2 - beta1 + w*xi_vals + v*z_vals)*(a/(b+1))
    dummy += np.mean(lambertW(np.power(timepos, b+1)*exppos)*z_vals)
    dummy += np.mean(lambertW(np.power(timeneg, b+1)*expneg)*z_vals)
    return (dummy/(2*v))

## The numerical form of the equation corresponding to the order parameter u-tilde.
def uoptimizer(u_old, v, w, beta1, a, b, zeta, xi_vals, z_vals, timepos, timeneg):
    dummy = 0
    exppos = (u_old**2)*np.exp(u_old**2 + beta1 + w*xi_vals + v*z_vals)*(a/(b+1))
    expneg = (u_old**2)*np.exp(u_old**2 - beta1 + w*xi_vals + v*z_vals)*(a/(b+1))
    dummy += np.mean(np.power((u_old**2)*np.ones(M) - lambertW(np.power(timepos, b+1)*exppos), 2))
    dummy += np.mean(np.power((u_old**2)*np.ones(M) - lambertW(np.power(timeneg, b+1)*expneg), 2))
    return (dummy/2 - zeta*(v**2))

## The numerical form of the equation corresponding to the order parameter v.
def voptimizer(v_old, u, w, beta1, a, b, zeta, xi_vals, z_vals, timepos, timeneg):
    dummy = 0
    exppos = (u**2)*np.exp(u**2 + beta1 + w*xi_vals + v_old*z_vals)*(a/(b+1))
    expneg = (u**2)*np.exp(u**2 - beta1 + w*xi_vals + v_old*z_vals)*(a/(b+1))
    dummy += np.mean(lambertW(np.power(timepos, b+1)*exppos)*z_vals)
    dummy += np.mean(lambertW(np.power(timeneg, b+1)*expneg)*z_vals)
    return (dummy/2 - zeta*v_old)

## The numerical form of the equation corresponding to the order parameter w.
def woptimizer(w_old, u, v, beta1, a, b, zeta, xi_vals, z_vals, timepos, timeneg):
    dummy = 0
    exppos = (u**2)*np.exp(u**2 + beta1 + w_old*xi_vals + v*z_vals)*(a/(b+1))
    expneg = (u**2)*np.exp(u**2 - beta1 + w_old*xi_vals + v*z_vals)*(a/(b+1))
    dummy += np.mean(lambertW(np.power(timepos, b+1)*exppos)*xi_vals)
    dummy += np.mean(lambertW(np.power(timeneg, b+1)*expneg)*xi_vals)
    return (dummy/2)

## The numerical form of the equation corresponding to the parameter vector beta1 (in this case, the scalar parameter beta1).
def beta1optimizer(beta1_old, u, v, w, a, b, zeta, xi_vals, z_vals, timepos, timeneg):
    dummy = 0
    exppos = (u**2)*np.exp(u**2 + beta1_old + w*xi_vals + v*z_vals)*(a/(b+1))
    expneg = (u**2)*np.exp(u**2 - beta1_old + w*xi_vals + v*z_vals)*(a/(b+1))
    dummy += np.mean(lambertW(np.power(timepos, b+1)*exppos))
    dummy += -1*np.mean(lambertW(np.power(timeneg, b+1)*expneg))
    return (dummy/2)

## The numerical form of the equation corresponding to the parameter a in the parametrized Cox model that is used.
def aoptimizer(a_old, u, v, w, beta1, b, zeta, xi_vals, z_vals, timepos, timeneg):
    dummy = 0
    exppos = (u**2)*np.exp(u**2 + beta1 + w*xi_vals + v*z_vals)*(a_old/(b+1))
    expneg = (u**2)*np.exp(u**2 - beta1 + w*xi_vals + v*z_vals)*(a_old/(b+1))
    dummy += np.mean(lambertW(np.power(timepos, b+1)*exppos))
    dummy += np.mean(lambertW(np.power(timeneg, b+1)*expneg))
    return (dummy/2 - u**2)

## The numerical form of the equation corresponding to the parameter b in the parametrized Cox model that is used.
def boptimizer(b_old, u, v, w, beta1, a, zeta, xi_vals, z_vals, timepos, timeneg):
    dummy = 0
    exppos = (u**2)*np.exp(u**2 + beta1 + w*xi_vals + v*z_vals)*(a/(b_old+1))
    expneg = (u**2)*np.exp(u**2 - beta1 + w*xi_vals + v*z_vals)*(a/(b_old+1))
    dummy += np.mean(np.log(timepos)*(lambertW(np.power(timepos, b_old+1)*exppos) - (u**2)*np.ones(M)))
    dummy += np.mean(np.log(timeneg)*(lambertW(np.power(timeneg, b_old+1)*expneg) - (u**2)*np.ones(M)))
    return (dummy*((b_old + 1)/2) - u**2)

## This is a test function which has been used during coding to optimize the code and check for errors and internal consistency. It is left in as
## a remnant of work performed, but is not a functioning part of the code in its final form.
def mainTest():
    xi_vals = np.random.normal(0,1,M)
    z_vals = np.random.normal(0,1,M)
    timepos = []
    timeneg = []
    for xi in xi_vals:
        unif = np.random.uniform(0,1)
        timepos.append(time_generator_null(unif, xi))
        timeneg.append(time_generator_null(unif, xi))
    timepos = np.asarray(timepos)
    timeneg = np.asarray(timeneg)
    itcounter = 1
    err = 1
    v_start = 0.0001
    u_old = u_start
    v_old = v_start
    w_old = w_start
    beta1_old = 0
    a_old = a_start
    b_old = b_start
    zeta_start = 0.1
    zeta_old = zeta_start
    while(err > 1.0e-5):
        print("This is iteration", itcounter)
        print(zeta_old, u_old, v_old, w_old, beta1_old, a_old, b_old)
        itcounter += 1
        print("The error is currently", err)
        u_new = spo.newton(uoptimizer, u_old, args=(v_old, w_old, beta1_old, a_old, b_old, zeta_old, xi_vals, z_vals, timepos, timeneg), tol=1.0e-5, maxiter=100)
        print("u has been updated!")
        #v_new = voptimizer(v_old, u_new, w_old, beta1_old, a_old, b_old, zeta)
        #v_new = spo.newton(voptimizer, v_old, args=(u_old, w_old, beta1_old, a_old, b_old, zeta_old), tol=1.0e-5, maxiter=100)
        #print("v has been updated!")
        w_new = spo.newton(woptimizer, w_old, args=(u_old, v_old, beta1_old, a_old, b_old, zeta_old, xi_vals, z_vals, timepos, timeneg), tol=1.0e-5, maxiter=100)
        print("w has been updated!")
        #beta1_new = spo.newton(beta1optimizer, beta1_old, args=(u_old, v_old, w_old, a_old, b_old, zeta_old, xi_vals, z_vals, timepos, timeneg), tol=1.0e-5, maxiter=100)
        #print("beta1 has been updated!")
        a_new = spo.newton(aoptimizer, a_old, args=(u_old, v_old, w_old, beta1_old, b_old, zeta_old, xi_vals, z_vals, timepos, timeneg), tol=1.0e-5, maxiter=100)
        print("a has been updated!")
        b_new = spo.newton(boptimizer, b_old, args=(u_old, v_old, w_old, beta1_old, a_old, zeta_old, xi_vals, z_vals, timepos, timeneg), tol=1.0e-5, maxiter=100)
        print("b has been updated!")
        zeta_new = zetafuncv(u_old, v_old, w_new, beta1_old, a_new, b_new, xi_vals, z_vals, timepos, timeneg)
        print("zeta has been updated!")

        err = np.sqrt((u_new - u_old)**2 + (zeta_new - zeta_old)**2 + (w_new - w_old)**2 + (a_new - a_old)**2 + (b_new - b_old)**2)
        u_old = u_new
        zeta_old = zeta_new
        #v_old = v_new
        w_old = w_new
        #beta1_old = beta1_new
        a_old = a_new
        b_old = b_new
    
    print("v:", v_old)
    print("The value of u is", u_old)
    #print("The value of v is", v_new)
    print("The value of w is", w_new, "and the deviation of w from S2 =", S2, "is", np.abs(w_new - S2))
    #print("The value of beta1 is", beta1_new, "and its deviation from the true value is", np.abs(beta1_new - truebeta1))
    print("The value of a is", a_new, "and its deviation from the true value is", np.abs(a_new - a0))
    print("The value of b is", b_new, "and its deviation from the true value is", np.abs(b_new - b0))
    #print("Check: The value of zeta for these values is given by zeta =", zetafuncu(u_new, v_new, w_new, beta1_new, a_new, b_new))
    print("Check: The value of zeta for these values is given by zeta =", zetafuncv(u_new, v_old, w_new, beta1_old, a_new, b_new, xi_vals, z_vals, timepos, timeneg))
    print("Second check: The value of zeta for these values is given by zeta=", zetafuncu(u_new, v_old, w_new, beta1_old, a_new, b_new, xi_vals, z_vals, timepos, timeneg))
    return 0

## The program that solves the obtained equations numerically in the presence of a non-Gaussian field.
def RunnerWithBeta1():
    xi_vals = np.random.normal(0,1,M)
    z_vals = np.random.normal(0,1,M)
    timepos = []
    timeneg = []
    for xi in xi_vals:
        unif = np.random.uniform(0,1)
        timepos.append(time_generator(unif, 1, xi))
        timeneg.append(time_generator(unif, -1, xi))
    timepos = np.asarray(timepos)
    timeneg = np.asarray(timeneg)

    results = [[0, 0, 0, 1, truebeta1, a0, b0]]
    u_old = u_start
    w_old = w_start
    beta1_old = beta1_start
    a_old = a_start
    b_old = b_start
    zeta_old = zeta_start
    v_values = np.append(np.array([0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0]), np.arange(10, 201, 2))
    try:
        for v in v_values:
            itcounter = 1
            err = 1
            while(err > 1.0e-5):
                print("This is iteration", itcounter)
                print(zeta_old, u_old, v, w_old, beta1_old, a_old, b_old)
                itcounter += 1
                print("The error is currently", err)
                u_new = spo.newton(uoptimizer, u_old, args=(v, w_old, beta1_old, a_old, b_old, zeta_old, xi_vals, z_vals, timepos, timeneg), tol=1.0e-5, maxiter=100)
                print("u has been updated!")
                w_new = spo.newton(woptimizer, w_old, args=(u_old, v, beta1_old, a_old, b_old, zeta_old, xi_vals, z_vals, timepos, timeneg), tol=1.0e-5, maxiter=100)
                print("w has been updated!")
                beta1_new = spo.newton(beta1optimizer, beta1_old, args=(u_old, v, w_old, a_old, b_old, zeta_old, xi_vals, z_vals, timepos, timeneg), tol=1.0e-5, maxiter=100)
                print("beta1 has been updated!")
                a_new = spo.newton(aoptimizer, a_old, args=(u_old, v, w_old, beta1_old, b_old, zeta_old, xi_vals, z_vals, timepos, timeneg), tol=1.0e-5, maxiter=100)
                print("a has been updated!")
                b_new = spo.newton(boptimizer, b_old, args=(u_old, v, w_old, beta1_old, a_old, zeta_old, xi_vals, z_vals, timepos, timeneg), tol=1.0e-5, maxiter=100)
                print("b has been updated!")
                zeta_new = zetafuncv(u_new, v, w_new, beta1_new, a_new, b_new, xi_vals, z_vals, timepos, timeneg)
                print("zeta has been updated!")
                err = np.max(np.array([(u_new - u_old)**2, (zeta_new - zeta_old)**2, (w_new - w_old)**2, (beta1_new - beta1_old)**2, (a_new - a_old)**2, (b_new - b_old)**2]))
                u_old = u_new
                w_old = w_new
                beta1_old = beta1_new
                a_old = a_new
                b_old = b_new
                zeta_old = zeta_new
            print("v:", v)
            print("The value of zeta is", zeta_new)
            print("The value of u is", u_new)
            print("The value of w is", w_new, "and the deviation of w from S2 =", S2, "is", np.abs(w_new - S2))
            print("The value of beta1 is", beta1_new, "and its deviation from the true value is", np.abs(beta1_new - truebeta1))
            print("The value of a is", a_new, "and its deviation from the true value is", np.abs(a_new - a0))
            print("The value of b is", b_new, "and its deviation from the true value is", np.abs(b_new - b0))
            print("Check: The value of zeta for these values is given by zeta =", zetafuncv(u_new, v, w_new, beta1_new, a_new, b_new, xi_vals, z_vals, timepos, timeneg))
            print("Second check: The value of zeta for these values is given by zeta=", zetafuncu(u_new, v, w_new, beta1_new, a_new, b_new, xi_vals, z_vals, timepos, timeneg))
            results.append([zeta_new, u_new, v, w_new/S2, beta1_new/truebeta1, a_new, b_new])
            if(zeta_new >= 0.5):
                return results
        return results
    except (RuntimeWarning, RuntimeError):
        return results

## The program that solves the obtained equations numerically without the non-Gaussian field.
def RunnerWithoutBeta1():
    xi_vals = np.random.normal(0,1,M)
    z_vals = np.random.normal(0,1,M)
    timepos = []
    timeneg = []
    for xi in xi_vals:
        unif = np.random.uniform(0,1)
        timepos.append(time_generator_null(unif, xi))
        timeneg.append(time_generator_null(unif, xi))
    results = [[0, 0, 0, 1, a0, b0]]
    u_old = u_start
    w_old = w_start
    beta1_old = 0
    a_old = a_start
    b_old = b_start
    zeta_old = zeta_start
    v_values = np.append(np.array([0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0]), np.arange(10, 201, 2))
    try:
        for v in v_values:
            itcounter = 1
            err = 1
            while(err > 1.0e-5):
                print("This is iteration", itcounter)
                print(zeta_old, u_old, v, w_old, beta1_old, a_old, b_old)
                itcounter += 1
                print("The error is currently", err)
                u_new = spo.newton(uoptimizer, u_old, args=(v, w_old, beta1_old, a_old, b_old, zeta_old, xi_vals, z_vals, timepos, timeneg), tol=1.0e-5, maxiter=100)
                print("u has been updated!")
                w_new = spo.newton(woptimizer, w_old, args=(u_old, v, beta1_old, a_old, b_old, zeta_old, xi_vals, z_vals, timepos, timeneg), tol=1.0e-5, maxiter=100)
                print("w has been updated!")
                a_new = spo.newton(aoptimizer, a_old, args=(u_old, v, w_old, beta1_old, b_old, zeta_old, xi_vals, z_vals, timepos, timeneg), tol=1.0e-5, maxiter=100)
                print("a has been updated!")
                b_new = spo.newton(boptimizer, b_old, args=(u_old, v, w_old, beta1_old, a_old, zeta_old, xi_vals, z_vals, timepos, timeneg), tol=1.0e-5, maxiter=100)
                print("b has been updated!")
                zeta_new = zetafuncv(u_new, v, w_new, beta1_old, a_new, b_new, xi_vals, z_vals, timepos, timeneg)
                print("zeta has been updated!")
                err = np.max(np.array([(u_new - u_old)**2, (zeta_new - zeta_old)**2, (w_new - w_old)**2, (a_new - a_old)**2, (b_new - b_old)**2]))
                #err = np.sqrt((u_new - u_old)**2 + (zeta_new - zeta_old)**2 + (w_new - w_old)**2 + (a_new - a_old)**2 + (b_new - b_old)**2)
                u_old = u_new
                w_old = w_new
                a_old = a_new
                b_old = b_new
                zeta_old = zeta_new
            print("v:", v)
            print("The value of zeta is", zeta_new)
            print("The value of u is", u_new)
            print("The value of w is", w_new, "and the deviation of w from S2 =", S2, "is", np.abs(w_new - S2))
            print("The value of a is", a_new, "and its deviation from the true value is", np.abs(a_new - a0))
            print("The value of b is", b_new, "and its deviation from the true value is", np.abs(b_new - b0))
            print("Check: The value of zeta for these values is given by zeta =", zetafuncv(u_new, v, w_new, beta1_old, a_new, b_new, xi_vals, z_vals, timepos, timeneg))
            print("Second check: The value of zeta for these values is given by zeta=", zetafuncu(u_new, v, w_new, beta1_old, a_new, b_new, xi_vals, z_vals, timepos, timeneg))
            results.append([zeta_new, u_new, v, w_new/S2, a_new, b_new])
            if(zeta_new >= 0.5):
                return results
        return results
    except (RuntimeWarning, RuntimeError):
        return results
    
## This is once again a program used to run consistency checks on the results of the code during the process in which results were obtained.
## It is left in as a part of the work performed, but is no longer an active part of the code in its final form.
def RunnerWithoutBeta1_diffS():
    xi_vals = np.random.normal(0,1,M)
    z_vals = np.random.normal(0,1,M)
    timepos = []
    timeneg = []
    for xi in xi_vals:
        unif = np.random.uniform(0,1)
        timepos.append(time_generator_null_diffS(unif, xi))
        timeneg.append(time_generator_null_diffS(unif, xi))
    results = [[0, 0, 0, 1, a0, b0]]
    u_old = u_start
    w_old = w_start
    beta1_old = 0
    a_old = a_start
    b_old = b_start
    zeta_old = zeta_start
    v_values = np.append(np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0]), np.arange(10, 201, 2))
    try:
        for v in v_values:
            itcounter = 1
            err = 1
            while(err > 1.0e-5):
                print("This is iteration", itcounter)
                print(zeta_old, u_old, v, w_old, beta1_old, a_old, b_old)
                itcounter += 1
                print("The error is currently", err)
                u_new = spo.newton(uoptimizer, u_old, args=(v, w_old, beta1_old, a_old, b_old, zeta_old, xi_vals, z_vals, timepos, timeneg), tol=1.0e-5, maxiter=100)
                print("u has been updated!")
                w_new = spo.newton(woptimizer, w_old, args=(u_old, v, beta1_old, a_old, b_old, zeta_old, xi_vals, z_vals, timepos, timeneg), tol=1.0e-5, maxiter=100)
                print("w has been updated!")
                a_new = spo.newton(aoptimizer, a_old, args=(u_old, v, w_old, beta1_old, b_old, zeta_old, xi_vals, z_vals, timepos, timeneg), tol=1.0e-5, maxiter=100)
                print("a has been updated!")
                b_new = spo.newton(boptimizer, b_old, args=(u_old, v, w_old, beta1_old, a_old, zeta_old, xi_vals, z_vals, timepos, timeneg), tol=1.0e-5, maxiter=100)
                print("b has been updated!")
                zeta_new = zetafuncv(u_new, v, w_new, beta1_old, a_new, b_new, xi_vals, z_vals, timepos, timeneg)
                print("zeta has been updated!")
                err = np.max(np.array([(u_new - u_old)**2, (zeta_new - zeta_old)**2, (w_new - w_old)**2, (a_new - a_old)**2, (b_new - b_old)**2]))
                #err = np.sqrt((u_new - u_old)**2 + (zeta_new - zeta_old)**2 + (w_new - w_old)**2 + (a_new - a_old)**2 + (b_new - b_old)**2)
                u_old = u_new
                w_old = w_new
                a_old = a_new
                b_old = b_new
                zeta_old = zeta_new
            print("v:", v)
            print("The value of zeta is", zeta_new)
            print("The value of u is", u_new)
            print("The value of w is", w_new, "and the deviation of w from S2 =", S2, "is", np.abs(w_new - S2))
            print("The value of a is", a_new, "and its deviation from the true value is", np.abs(a_new - a0))
            print("The value of b is", b_new, "and its deviation from the true value is", np.abs(b_new - b0))
            print("Check: The value of zeta for these values is given by zeta =", zetafuncv(u_new, v, w_new, beta1_old, a_new, b_new, xi_vals, z_vals, timepos, timeneg))
            print("Second check: The value of zeta for these values is given by zeta=", zetafuncu(u_new, v, w_new, beta1_old, a_new, b_new, xi_vals, z_vals, timepos, timeneg))
            results.append([zeta_new, u_new, v, w_new/np.sqrt(S2**2 + 1), a_new, b_new])
        return results
    except (RuntimeWarning, RuntimeError):
        return results

## The program generating plots for the relevant (order) parameters of the model from the outcomes obtained through the previously defined functions.
def plots():
    vdata, vdata_vars, wS2data, wS2data_vars, beta1data, beta1data_vars, adata, adata_vars, bdata, bdata_vars, zetadata = dgNew.Program(a_start, b_start, beta1_start, beta2_start)

    resultsWith = RunnerWithBeta1()
    zetavalsWith = []
    uvalsWith = []
    vvalsWith = []
    wS2valsWith = []
    beta1valsWith = []
    avalsWith = []
    bvalsWith = []
    for i in range(len(resultsWith)):
        zetavalsWith.append(resultsWith[i][0])
        uvalsWith.append(resultsWith[i][1])
        vvalsWith.append(resultsWith[i][2])
        wS2valsWith.append(resultsWith[i][3])
        beta1valsWith.append(resultsWith[i][4])
        avalsWith.append(resultsWith[i][5])
        bvalsWith.append(resultsWith[i][6])

    resultsWithout = RunnerWithoutBeta1()
    zetavalsWithout = []
    uvalsWithout = []
    vvalsWithout = []
    wS2valsWithout = []
    avalsWithout = []
    bvalsWithout = []
    for i in range(len(resultsWithout)):
        zetavalsWithout.append(resultsWithout[i][0])
        uvalsWithout.append(resultsWithout[i][1])
        vvalsWithout.append(resultsWithout[i][2])
        wS2valsWithout.append(resultsWithout[i][3])
        avalsWithout.append(resultsWithout[i][4])
        bvalsWithout.append(resultsWithout[i][5])


    plt.plot(zetavalsWith, uvalsWith, label='with Beta1')
    plt.plot(zetavalsWithout, uvalsWithout, label='without Beta1')
    #plt.plot(zetavalsDiffS2, uvalsDiffS2, label='S2 for all covariates')
    plt.xlabel(chr(950))
    plt.ylabel('u')
    plt.legend()
    plt.savefig('uzeta1.png')
    plt.ylim(bottom=0)
    plt.show()

    plt.plot(zetavalsWith, vvalsWith, label='with Beta1')
    plt.plot(zetavalsWithout, vvalsWithout, label='without Beta1')
    #plt.plot(zetavalsDiffS2, vvalsDiffS2, label='S2 for all covariates')
    plt.errorbar(zetadata, vdata, vdata_vars, linestyle='None', marker='.', label='Synthetic Data')
    plt.xlabel(chr(950))
    plt.ylabel('v')
    plt.legend()
    plt.savefig('vzeta1.png')
    plt.ylim(bottom=0)
    plt.show()

    plt.plot(zetavalsWith, wS2valsWith, label='with Beta1')
    plt.plot(zetavalsWithout, wS2valsWithout, label='without Beta1')
    #plt.plot(zetavalsDiffS2, wS2valsDiffS2, label='S2 for all covariates')
    plt.errorbar(zetadata, wS2data, wS2data_vars, linestyle='None', marker='.', label='Synthetic Data')
    plt.xlabel(chr(950))
    plt.ylabel('w/S2')
    plt.legend()
    plt.savefig('wS2zeta1.png')
    plt.ylim(bottom=1)
    plt.show()

    plt.plot(zetavalsWith, beta1valsWith)
    plt.errorbar(zetadata, beta1data, beta1data_vars, linestyle='None', marker='.', label='Synthetic Data')
    plt.xlabel(chr(950))
    plt.ylabel(chr(946))
    plt.savefig('beta1zeta1.png')
    plt.ylim(bottom=1)
    plt.show()

    plt.plot(zetavalsWith, avalsWith, label='with Beta1')
    plt.plot(zetavalsWithout, avalsWithout, label='without Beta1')
    #plt.plot(zetavalsDiffS2, avalsDiffS2, label='S2 for all covariates')
    plt.errorbar(zetadata, adata, adata_vars, linestyle='None', marker='.', label='Synthetic Data')
    plt.xlabel(chr(950))
    plt.ylabel('a')
    plt.legend()
    plt.savefig('azeta1.png')
    plt.ylim(bottom=1)
    plt.show()

    plt.plot(zetavalsWith, bvalsWith, label='with Beta1')
    plt.plot(zetavalsWithout, bvalsWithout, label='without Beta1')
    #plt.plot(zetavalsDiffS2, bvalsDiffS2, label='S2 for all covariates')
    plt.errorbar(zetadata, bdata, bdata_vars, linestyle='None', marker='.', label='Synthetic Data')
    plt.xlabel(chr(950))
    plt.ylabel('b')
    plt.legend()
    plt.savefig('bzeta1.png')
    plt.ylim(bottom=1)
    plt.show()

    plt.plot(zetavalsWith, beta1valsWith, label='Beta1')
    plt.plot(zetavalsWith, wS2valsWith, label='w/S2')
    plt.errorbar(zetadata, beta1data, beta1data_vars, linestyle='None', marker='.', label='Synthetic Data for beta1')
    plt.errorbar(zetadata, wS2data, wS2data_vars, linestyle='None', marker='.', label='Synthetic Data for w/S2')
    plt.xlabel(chr(950))
    plt.legend()
    plt.savefig('Beta1wS21.png')
    plt.ylim(bottom=1)
    plt.show()
    return 0

plots()
