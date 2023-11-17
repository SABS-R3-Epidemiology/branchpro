import stan
import arviz as az
import numpy as np
import branchpro
import scipy.stats


poisson_model = """
functions {
    real normalizing_const (real [] aSI) {
        return sum(aSI);
    }
    real effective_no_infectives (
        int N, int S, int t, real [] aI, real [] aSI) {
            real mean;
            if(t > S) {
                mean = (
                    dot_product(aI[(t-S):(t-1)], aSI) /
                    normalizing_const(aSI));
            }
            else {
                mean = (
                    dot_product(aI[:(t-1)], aSI[(S-t+2):]) /
                    normalizing_const(aSI));
            }
            return mean;
    }
}
data {
    int N; // number of days
    int S; // length serial interval
    int I[N]; // local incidences for N days
    int tau; // sliding window
    real revSI[S]; // reversed serial interval
}
parameters {
    real<lower=0> R[N-tau-1]; // vector of R numbers
}
model {
    for(t in (tau+1):(N-1)) {
        for(k in (t-tau+1):(t+1)) {
            if (effective_no_infectives(N, S, k, I, revSI) != 0) {
                I[k] ~ poisson (
                    R[t-tau] * effective_no_infectives(
                        N, S, k, I, revSI)); // likelihood
                    }
        }
    }
    for(t in 1:(N-tau-1)) {
        R[t] ~ gamma (1, 0.2); // prior of R
    }
}
"""

locimp_poisson_model = """
functions {
    real normalizing_const (real [] aSI) {
        return sum(aSI);
    }
    real effective_no_infectives (
        int N, int S, int t, real [] aI, real [] aSI) {
            real mean;
            if(t > S) {
                mean = (
                    dot_product(aI[(t-S):(t-1)], aSI) /
                    normalizing_const(aSI));
            }
            else {
                mean = (
                    dot_product(aI[:(t-1)], aSI[(S-t+2):]) /
                    normalizing_const(aSI));
            }
            return mean;
    }
}
data {
    int N; // number of days
    int S; // length serial interval
    int I[N]; // local incidences for N days
    int impI[N]; // imported incidences for N days
    int tau; // sliding window
    real revSI[S]; // reversed serial interval
    real epsilon; // epsilon
}
parameters {
    real<lower=0> R[N-tau-1]; // vector of R numbers
}
model {
    for(t in (tau+1):(N-1)) {
        for(k in (t-tau+1):(t+1)) {
            if (
                (effective_no_infectives(N, S, k, I, revSI) != 0) ||
                (effective_no_infectives(N, S, k, impI, revSI) != 0)) {
                    I[k] ~ poisson (
                        R[t-tau] * (effective_no_infectives(
                            N, S, k, I, revSI) + epsilon *
                            effective_no_infectives(
                                N, S, k, impI, revSI))); // likelihood
            }
        }
    }
    for(t in 1:(N-tau-1)) {
        R[t] ~ gamma (1, 5); // prior of R
    }
}
"""

num_timepoints = 30  # number of days for incidence data

# Build the imported cases
ic_mean = 70
imported_times = np.arange(1, (num_timepoints+1))
imported_cases = scipy.stats.poisson.rvs(ic_mean, size=num_timepoints)

# Build the serial interval w_s
ws_mean = 2.6
ws_var = 1.5**2
theta = ws_var / ws_mean
k = ws_mean / theta
w_dist = scipy.stats.gamma(k, scale=theta)
disc_w = w_dist.pdf(np.arange(num_timepoints))

# Simulate incidence data
epsilon = 1

initial_r = 3
serial_interval = disc_w

m = branchpro.BranchProModel(initial_r, serial_interval)
locimp_m = branchpro.LocImpBranchProModel(initial_r, serial_interval, epsilon)

new_rs = [3, 0.5]          # sequence of R_0 numbers
start_times = [0, 15]      # days at which each R_0 period begins

m.set_r_profile(new_rs, start_times)
locimp_m.set_r_profile(new_rs, start_times)

parameters = 100  # initial number of cases
times = np.arange(num_timepoints)

cases = m.simulate(parameters, times)

print(cases)

locimp_m.set_imported_cases(imported_times, imported_cases)
locally_infected_cases = locimp_m.simulate(parameters, times)

tau = 6
R_t_start = tau+1

poisson_data = {
    'N': num_timepoints,
    'S': len(serial_interval),
    'I': cases.astype(np.integer).tolist(),
    'tau': tau,
    'revSI': serial_interval.tolist()[::-1]}

locimp_poisson_data = {
    'N': num_timepoints,
    'S': len(serial_interval),
    'I': locally_infected_cases.astype(np.integer).tolist(),
    'impI': imported_cases.astype(np.integer).tolist(),
    'tau': tau,
    'revSI': serial_interval.tolist()[::-1],
    'epsilon': epsilon}

posterior = stan.build(
    poisson_model, data=poisson_data, random_seed=10)
locimp_posterior = stan.build(
    locimp_poisson_model, data=locimp_poisson_data, random_seed=10)

fit = posterior.sample(num_chains=4, num_samples=1000)
locimp_fit = locimp_posterior.sample(num_chains=4, num_samples=1000)

samples = az.from_pystan(
    fit,
    observed_data='I',
    coords={'observation': list(range(num_timepoints)),
            'covariate': [
                '{}'.format(_)
                for _ in range(num_timepoints - R_t_start)]
            },
    dims={
        'I': ['observation'],
        'R': ['covariate']
        })

locimp_samples = az.from_pystan(
    locimp_fit,
    observed_data='I',
    coords={'observation': list(range(num_timepoints)),
            'covariate': [
                '{}'.format(_)
                for _ in range(num_timepoints - R_t_start)]
            },
    dims={
        'I': ['observation'],
        'R': ['covariate']
        })

az.rcParams['plot.max_subplots'] = 2*(num_timepoints - R_t_start)

print(az.summary(samples))
az.plot_trace(
    samples,
    var_names=('R'),
    filter_vars='like',
    compact=False,
    show=True)

print(az.summary(locimp_samples))
az.plot_trace(
    locimp_samples,
    var_names=('R'),
    filter_vars='like',
    compact=False,
    show=True)
