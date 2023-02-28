function update_sigmaBeta(dat, cur, hyper)

	n = length(dat["nu"])
	J = hyper["J"]

	sigmaBeta = zeros(J,J)
	for j in 1:J 
		tmp = 0
		for i in 1:n
			Li = cur["L"][i]
			tmp += (cur["U_beta"][Li,j] - cur["muBeta"][j])^2 
		end 
		sigmaBeta[j,j] = rand(InverseGamma(hyper["cBeta"] + n/2, hyper["CBeta"] + tmp),1)[1]
	end

	return sigmaBeta 
end 

function update_muBeta(dat, cur, hyper)
	n = length(dat["nu"])
	J = hyper["J"]

	invΣβ = svd2inv(cur["sigmaBeta"]) 
	invSβ = (1/hyper["SBeta"]) * I 

	invSnew = invSβ + n*invΣβ
	Snew = svd2inv(invSnew) 

	sumVec = zeros(hyper["J"])
	for i in 1:n
		Li = cur["L"][i]
		sumVec += cur["U_beta"][Li,:]
	end

	snew = Snew * (invSβ * ones(J) * hyper["sBeta"] + invΣβ*sumVec)

	muBeta = rand(MvNormal(vec(snew), Snew), 1)

	return muBeta
end


function update_bPhi(dat, cur, hyper)
	n = length(dat["nu"])

	rnew = hyper["rPhi"] + hyper["aPhi"] * n 
	Rnew = hyper["RPhi"] 
	for i in 1:n
		Li = cur["L"][i]
		Rnew += cur["U_phi"][Li]
	end 

	bPhi = rand(InverseGamma(rnew, Rnew), 1)[1]

	return bPhi 
end 
