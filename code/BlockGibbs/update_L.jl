function update_L(dat, cur, hyper)

    survival = dat["survival"]
    nu = dat["nu"]
    z = dat["tz"]

    N = hyper["N"]
    U_beta = cur["U_beta"]
    U_phi = cur["U_phi"]
    logp = cur["logp"]

    n = length(survival)
    L = zeros(Int64, n) 

    for i in 1:n
        logLprob = zeros(N)

        for l in 1:N

            theta = exp(z[i,:]' * U_beta[l,:])
            phi = sqrt(U_phi[l])


            dist = LogLogistic(theta, phi)

            if nu[i] == 1
                logLprob[l] = logp[l] + logpdf(dist, survival[i])
            else
                logLprob[l] = logp[l] + logccdf(dist, survival[i])
            end

            if logLprob[l] == -Inf
                logLprob[l] = log(eps(0.0))
            end
        end 

        Lprob = exp.(logLprob .- Base.maximum(logLprob)) / sum( exp.(logLprob .- Base.maximum(logLprob)) )

        L[i] = sample([1:1:N;], Weights(Lprob))

    end 

    return L
end