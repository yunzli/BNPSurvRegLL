function update_alpha(dat, cur, hyper)

	n = length(dat["survival"])
	aAlpha = hyper["aAlpha"]
	bAlpha = hyper["bAlpha"]

    L = cur["L"]
    nstar = length(unique(L))

	alpha = cur["alpha"]
	eta = rand(Beta(alpha+1, n),1)[1]

	weight = (aAlpha + nstar - 1) / (n * (1 / bAlpha - log(eta)) + aAlpha + nstar - 1) 

	u = rand(Uniform(0,1), 1)[1] 
	if u < weight
		alpha = rand(Gamma(aAlpha+nstar, 1 / (1 / bAlpha - log(eta))), 1)[1]
	else
		alpha = rand(Gamma(aAlpha+nstar-1, 1 / (1 / bAlpha - log(eta))), 1)[1]
	end

	return alpha 

end
