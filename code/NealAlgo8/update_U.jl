function update_beta(dat, cur, hyper)

    L = cur["L"]
    
    phi = cur["phi"]
    epsilon = cur["epsilon"]
    J = hyper["J"]
    z = dat["tz"]

    survival = dat["survival"]
    nu = dat["nu"]

    k = length(phi) 
	beta_new = []

    sigmaBeta = cur["sigmaBeta"]
    invSigmaBeta = svd2inv(sigmaBeta)
    muBeta = cur["muBeta"]

    for l in 1:k 
        ind = findall(L .== l)

        ezz = zeros(J, J)
        Asum = zeros(J) 
        Bsum = zeros(J) 
        for i in ind 
            ezz  += epsilon[i] * (z[i,:] * z[i,:]')
            Asum += epsilon[i] * log(survival[i]) * z[i,:]
            Bsum += (1 - nu[i]) * z[i,:]            
        end 
        invΣnew = invSigmaBeta + phi[l]^2 * ezz
        Σnew = svd2inv(invΣnew)
        μnew = Σnew * (invSigmaBeta * muBeta + phi[l]^2 * Asum + 0.5 * phi[l] * Bsum)
        new = rand(MvNormal(vec(μnew), Σnew), 1)[:,1]

        push!(beta_new, new)
    end

    return beta_new 
end

function MH_phi_sampler(phi2, nu, c, aPhi, bPhi)

	phi2prop = exp(log(phi2) + rand(Normal(0, 1),1)[1])

	logcur = logpdf(Gamma(aPhi,bPhi),phi2) + 0.5 * sum((nu .- 1) .* c) * sqrt(phi2) + log(phi2) 
	logpro = logpdf(Gamma(aPhi,bPhi),phi2prop) + 0.5 * sum((nu .- 1) .* c) * sqrt(phi2prop) + log(phi2prop) 

	if rand(Uniform(0,1),1)[1] < (logpro - logcur)
		return phi2prop
	else
		return phi2
	end 
end 

function update_phi(dat, cur, hyper)

	nu = dat["nu"] 
    survival = dat["survival"]

	aPhi = hyper["aPhi"]
	bPhi = cur["bPhi"]
	epsilon = cur["epsilon"]
	beta = cur["beta"]
	phi = cur["phi"]

	L = cur["L"]
	z = dat["tz"] 

    k = length(phi)
	phi_new = zeros(k)
	
	for l in 1:k

        ind = findall(L .== l)

        anew = sum(nu[ind])/2 + aPhi 
        c = log.(survival[ind]) .- (z[ind,:] * beta[l])  # log(t) - Z'Uᵦₗ

        bnew_inv = 1 / bPhi + 0.5 * sum(epsilon[ind] .* (c .^ 2)) 
        bnew = 1 / bnew_inv 

        if length(ind) == sum(nu[ind]) 
            new = rand(Gamma(anew, bnew), 1)[1]
        else
            new = MH_phi_sampler(phi[l]^2, nu[ind], c, anew, bnew)
        end

		phi_new[l] = sqrt(new)
	end

	return phi_new 
end 