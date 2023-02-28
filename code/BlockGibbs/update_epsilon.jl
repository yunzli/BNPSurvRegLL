using PolyaGammaDistribution

function update_epsilon(dat, cur) 

    survival = dat["survival"]
    n = length(survival)
    z = dat["tz"]
    nu = dat["nu"]
    L = cur["L"]
    
    U_phi, U_beta = cur["U_phi"], cur["U_beta"]

    epsilon = zeros(n)

    for i in 1:n
        Li = L[i]
        a = 1 + nu[i]
        b = sqrt(U_phi[Li]) * (log(survival[i]) - z[i,:]'*U_beta[Li,:])
        epsilon[i] = rand(PolyaGamma(a, b), 1)[1]
    end

    return epsilon 
end