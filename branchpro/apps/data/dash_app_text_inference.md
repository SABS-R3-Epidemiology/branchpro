##### BranchPro: inference of $R\_t$

In order to make predictions about how case numbers will evolve in the future, we require an estimate of the reproduction number $R\_t$. One common method of estimating $R\_t$ is to infer it (using Bayesian inference) from recent incidence data. Below, we do just that: given the incidence data for 30 days, we can infer the value of $R\_t$ over that time period. In a sense, this is the opposite (a.k.a. the inverse) of the *forward model*, where we used a known/assumed value for the reproduction number to predict future case numbers; now, we are using past case numbers to estimate the reproduction number. This is called an *inverse probem*.

The model also calculates a 95% *credible interval* for the reproduction number; this simply tells us that we are 95% sure that the value for $R$ (the reproduction number value assumed to be constant over the last $\tau$ days) lies between a lower and an upper bound. So, for example, for day 16, our model might calculate that $R\_{16}$ was most likely equal to 1.3 and that we are 95% sure that it lay between 1.0 and 1.5. This credible interval is an important output of the model because it highlights that we can only *estimate* values for $R\_t$ and also *quantifies the uncertainty* of the model.

Another important feature of this inference model is its ability to distinguish between the transmission risks of imported cases and local cases. If we assume that ε is the relative transmissibility risk between imported cases and local cases, i.e. the ratio of expected secondary cases from an imported infector to that from a local infector, then the inference of the local reproduction number depends on ε. For example, if it is thought that people entering the country who are carrying the disease tend to infect more people than carriers already within the country, i.e. ε>1, but we ignore this fact in our inference model, then the local reproduction number $R\_t$ will be overestimated. Our model corrects for such effects by incorporating ε.

You can upload your own data to this app, including imported case data if desired.

When imported cases are uploaded, a slider will appear allowing you to choose the value of ε, the relative transmission risk of an imported case compared to a local one.

All of the code can be found on [our GitHub page](https://github.com/SABS-R3-Epidemiology/branchpro).

*Disclaimer: The modelling framework adopted by this software tool involves a number of simplifying assumptions. Furthermore, the validity of any outputs is contingent on appropriate choices for the parameter values and the accuracy of user-uploaded data. The developers urge cautious interpretation of all results and renounce any responsibility for decisions made on the basis of this software. Please refer to the paper above for full details.*

Click [here](https://sabs-r3-epidemiology.github.io/branchpro/) to return to the home page.
