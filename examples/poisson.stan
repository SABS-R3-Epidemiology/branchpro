functions {
    // need a function to compute poissonbinomial log-probability
    //   log-prob of k successes from n items 
    //   where prob of success of the n items is given by vector pp
    // this mainly requires looping over combinations of n-choose-k
    // for each combination, log-prob is straight-forward, given vector of probs pp

    // function for log-prob of each term, given combination of size k
    // called by lpmf below
    real lppoibin( int k, int n, array [] int elements, vector pp) {
        real lp;
        lp = 0;
        //print(elements);

        // for each possible element, accumulate probability
        for (i in 1:n) {
            int bang;
            bang = 0;
            for (j in 1:k) {
                if (elements[j]==i) {
                    bang = 1;
                }
            }
            if ( bang==1 ) {
                // term in elements
                lp = lp + log(pp[i]);
            } 
            else {
                // term not in elements
                lp = lp + log1m(pp[i]);
            }
        }
        return lp;
    }
    real poissonbinomial_lpmf(int k, int n, vector pp) {
        array [n] int x;
        array [k] int a;
        int e;
        int h;
        array [k] int r;
        vector[choose(n, k)] terms; // holds each log-prob

        for (i in 1:n) {
            x[i] = i;
        }
        for (i in 1:k) {
            a[i] = i;
        }

        // first sequence
        r = a;
        terms[1] = lppoibin(k, n, r, pp);

        // loop to lexicographically generate remaining
        if (k > 0) {
            int i;
            int j;
            int nmmp1;
            i = 2;
            nmmp1 = n - k + 1;
            while (a[1] != nmmp1) {
                if (e < (n - h)) {
                    h = 1;
                    e = a[k];
                    j = 1;
                    a[k - h + j] = e + j;
                }
                else {
                    e = a[k - h];
                    h = h + 1;
                    j = h;
                    for (ii in 1:j) {
                        a[k - h + ii] = e + ii;
                    }
                }
                for ( ii in 1:k ) {
                    r[ii] = x[a[ii]];
                }
                terms[i] = lppoibin(k, n, r, pp);
                i += 1;
            }
        }
        return log_sum_exp(terms);
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
    array [N] real<lower=0> Mu;
    // vector of time-dependent mean number of contacts
}
model {
    for(t in 2:N) {
        target += poissonbinomial_lpmf(
            I[t] |
            sum(evaluate_previous_cases(t, I, Mu)),
            evaluate_poissonbinomial_probabilities(
                N, S, t, I, Theta, Mu)
            ); // likelihood
    }
}
generated quantities{
    array [N] real R; // vector of R numbers
    for(t in 1:N) {
        R[t] = 0;
    }
}