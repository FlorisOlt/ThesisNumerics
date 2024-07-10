import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import warnings
import os
warnings.filterwarnings("error")
os.chdir("C:\\Users\\Flori\\Documents\\Master Thesis Code")
## Code created by Floris Olthuis

## Parameters
a0 = 1
b0 = 1
truebeta1 = 1
p = 150
truebeta2 = np.ones(p)
S2 = np.sqrt(np.sum(np.square(truebeta2))/p)

beta2_start = 0.9999*truebeta2
beta1_start = 0.9999*truebeta1
a_start = 0.9999
b_start = 0.9999

samplesDataAvg = 15

## Function used for generating time-samples through inversion sampling
def time_generator(u, z1, z2):
    return np.power(-(((b0+1)/a0)*np.exp(-truebeta1*z1 - (1/np.sqrt(p))*np.sum(truebeta2*z2, axis=1))*np.log(1-u)), (1/(b0+1)))

## Creating a log-likelihood, the negative of which is to be minimized
def minus_log_likelihood(initial, *args):
    beta2 = []
    a = initial[0]
    b = initial[1]
    beta1 = initial[2]
    for i in range(p):
        beta2.append(initial[i+3])
    beta2 = np.array(beta2)
    z1, z2, times, N = args
    dummy = 0
    dummy += N*np.log(a) + np.sum(b*np.log(times) + beta1*z1 + (1/np.sqrt(p))*np.sum(beta2*z2, axis=1) - np.exp(beta1*z1 + (1/np.sqrt(p))*np.sum(beta2*z2, axis=1))*(a/(b+1))*(times**(b+1)) )
    return -1*dummy

def Optimizer(a_in, b_in, beta1_in, beta2_in, z1, z2, times, N):
    beta2_old = beta2_in
    beta1_old = beta1_in
    a_old = a_in
    b_old = b_in
    minimum = spo.minimize(minus_log_likelihood, np.append(np.array([a_old, b_old, beta1_old]), beta2_old), args=(z1,z2,times,N), method='BFGS', options={'maxiter':1000}).x
    print("The optimization has been performed for zeta =", np.size(z2[0])/np.size(z1), "i.e. N =", np.size(z1), "and p =", np.size(z2[0]))
    print("a =", minimum[0])
    print("b =", minimum[1])
    print("beta1 =", minimum[2])
    beta2 = []
    for i in range(p):
        beta2.append(minimum[i+3])
    print("beta2 =", beta2)
    return minimum[0], minimum[1], minimum[2], np.array(beta2)

## Constructing values for the order parameter v from inferred samples of the vector beta2
def v_vals(beta2_vals):
    inprod = np.array([np.inner(beta2, beta2) for beta2 in beta2_vals])
    dummy = np.mean(inprod) - (np.inner(truebeta2, np.mean(beta2_vals, axis=0))/np.linalg.norm(truebeta2))**2
    return np.sqrt(dummy/p)

## Calculating the standard deviation for v. Note that this is done by calculating v for singular instances of the vector beta2 and then
## calculating the standard deviation for this array, thereby strictly speaking creating an upper bound for the standard deviation!
def v_vars(beta2_vals):
    v_dummy = []
    for beta2 in beta2_vals:
        v_single = np.sqrt(np.inner(beta2, beta2) - (np.inner(truebeta2, beta2)/np.linalg.norm(truebeta2)**2))/np.sqrt(p)
        v_dummy.append(v_single)
    return np.sqrt(np.var(np.array(v_dummy)))

## Constructing values for the order parameter w from inferred samples of the vector beta2
def w_vals(beta2_vals):
    dummy = np.inner(truebeta2, np.mean(beta2_vals, axis=0))/np.linalg.norm(truebeta2)
    return dummy/np.sqrt(p)

## Calculating the standard deviation for w.
def w_vars(beta2_vals):
    return np.sqrt( ((1/(p*samplesDataAvg**2))/np.linalg.norm(truebeta2)**2)*np.sum(np.square(truebeta2))*np.sum(np.var(beta2_vals,axis=0)) )

## Program to infer the desired model parameters for many instantiations of synthetic data and constructing the desired data-averaged values.
## Note that the program contains a method to deal with overflow, so as to return usable results even if at some point the machine struggles with the regression.
def Program(a_start, b_start, beta1_start, beta2_start):
    results_v = []
    results_wS2 = []
    results_beta1 = []
    results_a = []
    results_b = []
    zeta = []
    vars_v = []
    vars_wS2 = []
    vars_beta1 = []
    vars_a = []
    vars_b = []
    zeta_vals = np.arange(0.025, 0.525, 0.025)
    N_vals = (p/np.array(zeta_vals)).astype(int)
    try:
        a_in = a_start
        b_in = b_start
        beta1_in = beta1_start
        beta2_in = beta2_start
        for N in N_vals:
            results_at = []
            results_bt = []
            results_beta1t = []
            results_beta2t = []
            for i in range(samplesDataAvg): 
                print("N =", N, ", sample =", i)
                z1 = np.random.randint(2, size=N)*2 - np.ones(N)
                z2 = np.random.normal(0,1,(N,p))
                u = np.random.uniform(0, 1, N)
                times = time_generator(u, z1, z2)
                a, b, beta1, beta2 = Optimizer(a_in, b_in, beta1_in, beta2_in, z1, z2, times, N)
                results_at.append(a)
                results_bt.append(b)
                results_beta1t.append(beta1)
                results_beta2t.append(beta2)
            zeta.append(p/N)
            results_at = np.array(results_at)
            results_bt = np.array(results_bt)
            results_beta1t = np.array(results_beta1t)
            results_beta2t = np.array(results_beta2t)
            results_a.append(np.mean(results_at))
            vars_a.append(np.sqrt(np.var(results_at)))
            results_b.append(np.mean(results_bt))
            vars_b.append(np.sqrt(np.var(results_bt)))
            results_beta1.append(np.mean(results_beta1t)/truebeta1)
            vars_beta1.append(np.sqrt(np.var(results_beta1t)))
            results_v.append(v_vals(results_beta2t))
            vars_v.append(v_vars(results_beta2t))
            results_wS2.append(w_vals(results_beta2t)/S2)
            vars_wS2.append(w_vars(results_beta2t)/S2)

            

        return results_v, vars_v, results_wS2, vars_wS2, results_beta1, vars_beta1, results_a, vars_a, results_b, vars_b, zeta
    except (RuntimeWarning, RuntimeError):
        return results_v, vars_v, results_wS2, vars_wS2, results_beta1, vars_beta1, results_a, vars_a, results_b, vars_b, zeta
    
## Code for plotting the results for different regressions, so as to observe what the outcome is.
def Plotter():
    v, v_vars, wS2, wS2_vars, beta1, beta1_vars, a, a_vars, b, b_vars, zeta = Program(a_start, b_start, beta1_start, beta2_start)
    print(v_vars)
    plt.errorbar(zeta, v, v_vars, linestyle='None', marker='.')
    plt.ylabel("v")
    plt.xlabel(chr(950))
    plt.title("Results from synthetic data")
    plt.savefig('vzetasynth.png')
    plt.show()

    plt.errorbar(zeta, v, v_vars, linestyle='None', marker='.', label="Gaussian v")
    plt.errorbar(zeta, results_v1, vars_v1, linestyle='None', marker='.', label="non-Gaussian v")
    plt.xlabel(chr(950))
    plt.legend()
    plt.title("Results from synthetic data")
    plt.savefig('bothvzeta.png')
    plt.show()

    plt.errorbar(zeta, wS2, wS2_vars, linestyle='None', marker='.')
    plt.ylabel("w/S2")
    plt.xlabel(chr(950))
    plt.title("Results from synthetic data")
    plt.savefig('wS2zetasynth.png')
    plt.show()

    plt.errorbar(zeta, beta1, beta1_vars, linestyle='None', marker='.')
    plt.ylabel(chr(946))
    plt.xlabel(chr(950))
    plt.title("Results from synthetic data")
    plt.savefig('beta1zetasynth.png')
    plt.show()

    plt.errorbar(zeta, a, a_vars, linestyle='None', marker='.')
    plt.ylabel("a")
    plt.xlabel(chr(950))
    plt.title("Results from synthetic data")
    plt.savefig('azetasynth.png')
    plt.show()

    plt.errorbar(zeta, b, b_vars, linestyle='None', marker='.')
    plt.ylabel("b")
    plt.xlabel(chr(950))
    plt.title("Results from synthetic data")
    plt.savefig('bzetasynth.png')
    plt.show()

    return 0