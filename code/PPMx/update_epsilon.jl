using PolyaGammaDistribution

function update_epsilon(dat, cur) 

    survival = dat["survival"]
    n = length(survival)
    z = dat["tz"]
    nu = dat["nu"]
    L = cur["L"]
    
    phi, beta = cur["phi"], cur["beta"]

    epsilon = zeros(n)

    for i in 1:n
        Li = L[i]
        a = 1 + nu[i]
        b = phi[Li] * (log(survival[i]) - (z[i,:]' * beta[Li]))
        epsilon[i] = rand(PolyaGamma(a, b), 1)[1]
    end
    return epsilon 
end