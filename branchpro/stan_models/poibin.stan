functions {
    // Return the Poisson-binomial log probability mass for the specified
    // count y and vector of probabilities theta.  The result is the log
    // probability of y successes in N = size(theta) independent
    // trials with probabilities of success theta[1], ..., theta[N].
    //
    // See:  https://en.wikipedia.org/wiki/Poisson_binomial_distribution
    //
    // @param y number of successes
    // @param theta vector of probabilities
    // @return Poisson-binomial log probability mass
    //
    real poisson_binomial_lpmf(int y, vector theta) {
        int N = rows(theta);
        matrix[N + 1, N + 1] alpha;
        vector[N] log_theta = log(theta);
        vector[N] log1m_theta = log1m(theta);

        if (y < 0 || y > N)
        reject("poisson binomial variate y out of range, y = ", y,
                " N = ", N);
        for (n in 1:N)
        if (theta[n] < 0 || theta[n] > 1)
            reject("poisson binomial parameter out of range,",
                " theta[", n, "] =", theta[n]);

        if (N == 0)
        return y == 0 ? 0 : negative_infinity();

        // dynamic programming with cells
        // alpha[n + 1, tot + 1] = log prob of tot successes in first n trials
        alpha[1, 1] = 0;
        for (n in 1:N) {
        // tot = 0
        alpha[n + 1, 1] = alpha[n, 1] + log1m_theta[n];

        // 0 < tot < n
        for (tot in 1:min(y, n - 1))
            alpha[n + 1, tot + 1]
                = log_sum_exp(alpha[n, tot] + log_theta[n],
                            alpha[n, tot  + 1] + log1m_theta[n]);

        // tot = n
        if (n > y) continue;
        alpha[n + 1, n + 1] = alpha[n, n] + log_theta[n];
        }
        return alpha[N + 1, y + 1];
    }
    int [] evaluate_previous_cases (int t, array [] int aI, array [] real aMu) {
        array [t-1] int rev_prev_cases;
        int counter;

        for (k in 1:(t-1)) {
            rev_prev_cases[k] = 0;
            counter = 0;
            while (counter < floor(aMu[t])) {
                rev_prev_cases[k] += aI[t-k];
                counter += 1;
            }
        }
        return rev_prev_cases;
    }
    vector evaluate_poissonbinomial_probabilities (
        int N, int S, int t, array [] int aI, array [] real aTheta, array [] real aMu) {
        array [t-1] int rev_prev_cases;
        int ci;
        
        rev_prev_cases = evaluate_previous_cases(t, aI, aMu);
        
        vector [sum(rev_prev_cases)] pp;

        pp[:rev_prev_cases[1]] = rep_vector(aTheta[1], rev_prev_cases[1]);
        ci = rev_prev_cases[1];
        for(k in 2:(t-1)) {
            if(k <= S) {
                pp[(ci + 1):(ci + rev_prev_cases[k])] = rep_vector(
                    aTheta[k], rev_prev_cases[k]);
            }
            else {
                pp[(ci + 1):(ci + rev_prev_cases[k])] = rep_vector(
                    0, rev_prev_cases[k]);
            }
        }
        return pp;
    }
}
data {
    int N; // number of days
    int S; // length probability distribution
    array [N] int I; // local incidences for N days
    array [S] real Theta; // probability distribution of
    // that the infector infects a contact s days after they
    // first displays symptoms.
}
parameters {
    array [N] real<lower=0, upper=50> Mu;
    // vector of time-dependent mean number of contacts
}
model {
    for(t in 2:N) {
        if (sum(evaluate_previous_cases(t, I, Mu)) != 0){
            target += poisson_binomial_lpmf(
                I[t] |
                evaluate_poissonbinomial_probabilities(
                    N, S, t, I, Theta, Mu)
                ); // likelihood
        }
    }
    for(t in 1:N){
        Mu[t] ~ gamma(14, 0.5);
    }
}