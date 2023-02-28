function update_L(dat, cur, hyper)

    L = cur["L"]
    nu = dat["nu"]
    survival = dat["survival"]
    n = length(survival)
    m = hyper["m"]
    alpha = cur["alpha"]
    z = dat["tz"]
    J = hyper["J"]

    beta = cur["beta"]
    phi = cur["phi"]

    sigmaBeta = cur["sigmaBeta"]
    muBeta = cur["muBeta"]
    bPhi = cur["bPhi"]
    aPhi = hyper["aPhi"]

    for i in 1:n

        phi_i = phi[L[i]]
        beta_i = beta[L[i]]

        _L = L[1:end .!= i]
        _Lstar = sort(unique(_L))

        _k = length(_Lstar)  
        h = _k + m 
        
        is_singleton = false 
        if _k < length(phi) # phi[i] is a singleton 
            is_singleton = true 
            for j in 1:n # relabel L
                if (j != i) & (L[j] > L[i])
                    L[j] = L[j] - 1 
                end
            end
        end 

        phi = phi[_Lstar]
        beta = beta[_Lstar]

        _L = L[1:end .!= i]
        h = _k + m

        _phi = deepcopy(phi)
        _beta = deepcopy(beta)
        for l in 1:m
            if l == 1 
                if is_singleton 
                    push!(_beta, beta_i)
                    push!(_phi, phi_i)
                else 
                    beta_tmp = zeros(J)
                    for j in 1:J
                        beta_tmp[j] = rand(Normal(muBeta[j], sqrt(sigmaBeta[j,j])),1)[1]
                    end 
                    phi_tmp = sqrt(rand(Gamma(aPhi, bPhi), 1)[1])
                    push!(_beta, beta_tmp)
                    push!(_phi, phi_tmp)
                end 
            else
                beta_tmp = zeros(J)
                for j in 1:J
                    beta_tmp[j] = rand(Normal(muBeta[j], sqrt(sigmaBeta[j,j])),1)[1]
                end 
                phi_tmp = sqrt(rand(Gamma(aPhi, bPhi), 1)[1])
                push!(_beta, beta_tmp)
                push!(_phi, phi_tmp)
            end 
        end 

        logLprob = zeros(h) 
        for c in 1:h
            if c < _k + 1
                _nc = length(findall(_L .== c))
                logLprob[c] = log(_nc)
                theta = exp(z[i,:]' * _beta[c])
            else
                logLprob[c] = log(alpha/m) 
                theta = exp(z[i,:]' * _beta[c])
            end

            if nu[i] == 1
                logLprob[c] += logpdf(LogLogistic(theta, _phi[c]), survival[i])
            else
                logLprob[c] += logccdf(LogLogistic(theta, _phi[c]), survival[i])
            end
        end


        Lprob = exp.(logLprob .- Base.maximum(logLprob)) / sum( exp.(logLprob .- Base.maximum(logLprob)) )
        Li = sample([1:1:h;], Weights(Lprob))

        if Li > _k 
            push!(phi, _phi[Li])
            push!(beta, _beta[Li])
            Li = _k + 1
        end 
        L[i] = Li
            
    end

    return Dict("L" => L, "beta" => beta, "phi" => phi)
end