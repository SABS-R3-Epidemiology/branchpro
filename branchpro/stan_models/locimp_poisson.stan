functions {
    real normalizing_const (array [] real aSI) {
        return sum(aSI);
    }
    real effective_no_infectives (
        int N, int S, int t, array [] real aI, array [] real aSI) {
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
    array [N] int I; // local incidences for N days
    array [N] int impI; // imported incidences for N days
    int tau; // sliding window
    array [S] real revSI; // reversed serial interval
    real epsilon; // epsilon
}
parameters {
    array [N-tau-1] real<lower=0> R; // vector of R numbers
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