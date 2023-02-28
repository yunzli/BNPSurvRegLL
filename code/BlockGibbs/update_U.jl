function update_U_beta(dat, cur, hyper)

	survival = dat["survival"]
	nu = dat["nu"]

	L = cur["L"]
	Lstar = unique(L) 

	sigmaBeta = cur["sigmaBeta"]
	invSigmaBeta = svd2inv(sigmaBeta) 
	muBeta = cur["muBeta"]

	epsilon = cur["epsilon"]
	U_phi = cur["U_phi"]

	z = dat["tz"]
	N = hyper["N"]
	J = hyper["J"]

	U_beta_new = zeros(N, J)

	for l in 1:N
		if l in Lstar
			ind = findall(L .== l)

            ezz = zeros(size(z)[2], size(z)[2])
			Asum = zeros(J)
			Bsum = zeros(J)
			for i in ind
                ezz  += epsilon[i] * (z[i,:] * z[i,:]')
				Asum += epsilon[i] * log(survival[i]) * z[i,:]
				Bsum += (1 - nu[i]) * z[i,:] 
			end

			invΣnew = invSigmaBeta + U_phi[l] * ezz
			Σnew = svd2inv(invΣnew)
			μnew = Σnew * (invSigmaBeta * muBeta + U_phi[l] * Asum + 0.5 * sqrt(U_phi[l]) * Bsum)
			new = rand(MvNormal(vec(μnew), Σnew), 1) 
		else
			new = rand(MvNormal(vec(muBeta), sigmaBeta), 1) 
		end

		@assert(length(findall(new .== -Inf)) == 0, "-Inf beta") 
		U_beta_new[l,:] = new 
	end 

	return U_beta_new  
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


function update_U_phi(dat, cur, hyper) 
	Lstar = unique(cur["L"]) 

	nu = dat["nu"] 
	aPhi = hyper["aPhi"]
	bPhi = cur["bPhi"]
	epsilon = cur["epsilon"]
	U_beta = cur["U_beta"]
	U_phi = cur["U_phi"]
	L = cur["L"]
	z = dat["tz"] 

	survival = dat["survival"]
	
	N = hyper["N"]
	U_phi_new = zeros(N)
	
	for l in 1:N 

		if l in Lstar
			ind = findall(L .== l)

			anew = sum(nu[ind])/2 + aPhi 
			c = log.(survival[ind]) .- (z[ind,:] * U_beta[l,:])  # log(t) - Z'Uᵦₗ

			bnew_inv = 1 / bPhi + 0.5 * sum(epsilon[ind] .* (c .^ 2)) 
			bnew = 1 / bnew_inv 

			if length(ind) == sum(nu[ind]) 
				new = rand(Gamma(anew, bnew), 1)[1]
			else
				new = MH_phi_sampler(U_phi[l], nu[ind], c, anew, bnew)
			end
		else
			new = rand(Gamma(aPhi, bPhi), 1)[1]
		end

		U_phi_new[l] = new
	end

	return U_phi_new 
end 
