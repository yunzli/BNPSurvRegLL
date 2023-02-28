function update_alpha(cur, hyper)

    anew = hyper["N"] + hyper["aAlpha"] - 1
    bnew = 1 / (1 / hyper["bAlpha"] - cur["logp"][end])

    alpha = rand(Gamma(anew, bnew), 1)[1]

    return alpha 
end