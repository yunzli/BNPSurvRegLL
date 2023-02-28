function update_p(cur, hyper)

    N = hyper["N"]

    L = cur["L"]
    alpha = cur["alpha"]

    V = zeros(N-1)
    logp = zeros(N)
    M = zeros(Int64, N)

    for l in 1:N
        M[l] = length(findall(L .== l))
    end 

    V[1] = rand(Beta(1+M[1], alpha+sum(M[2:N])), 1)[1]
    logp[1] = log(V[1])

    for l in 2:(N-1)
        V[l] = rand(Beta(1+M[l], alpha+sum(M[(l+1):N])), 1)[1]
        logp[l] = log(V[l]) + sum(log.(1 .- V[1:(l-1)]))
    end

    logp[N] = sum(log.(1 .- V))

    for l in 1:N
        if logp[l] == -Inf
            logp[l] = log(eps(0.0))
        end 
    end

	@assert sum(exp.(logp)) ≈ 1.0

	return logp 
end