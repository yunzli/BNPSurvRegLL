using JLD2
using RCall 
using Random
using TOML 
using StatsBase 
using LinearAlgebra
using Distributions 
using ProgressMeter
using ProximalOperators

Random.seed!(20221001)

include("../utils.jl")
include("../loglogistic.jl")
include("./update_U.jl") 
include("./update_L.jl")
include("./update_epsilon.jl") 
include("./update_baseline.jl") 



function MCMC(config_file)
	"""
	config_file: a TOML configuration file in the folder ./configs
	"""
	
	config = TOML.parsefile(config_file)

	nsam = config["nsam"]
	dat = load(config["data_file"])
	hyper = config["hyper"]
    
	n = length(dat["survival"]) 

	J = hyper["J"]

	cur = Dict(
		# initialization of all parameters 
		"phi" => [0.5, 2],
		"beta" => [zeros(J), ones(J)],
		"L" => fill(1, n),
		"epsilon" => fill(0.1, n), 
		"muBeta" => hyper["sBeta"] .* ones(J),
		"sigmaBeta" => hyper["CBeta"] .* Matrix(Diagonal(ones(J))), 
		"bPhi" => 1.0,
		"alpha" => hyper["alpha"],
		)

    # posterior values
    pos = Dict(
		"phi" => [], # Matrix{Float64}(undef, nsam, N),
		"beta" => [], # Array{Float64}(undef, nsam, N, J),
		"L" => Matrix{Int64}(undef, nsam, n),
		"epsilon" => Matrix{Float64}(undef, nsam, n),
		"sigmaBeta" => Array{Float64}(undef, nsam, J, J), 
		"muBeta" => Matrix{Float64}(undef, nsam, J),
		"bPhi" => Vector{Float64}(undef, nsam),
	)

	@showprogress for i in (1:nsam)
		# Gibbs sampler 
		pos["epsilon"][i,:] = cur["epsilon"] = update_epsilon(dat, cur)
		cur["beta"] = update_beta(dat, cur, hyper)
        push!(pos["beta"], cur["beta"])
        cur["phi"] = update_phi(dat, cur, hyper)
		push!(pos["phi"], cur["phi"]) 
		tmp = update_L(dat, cur, hyper)
        pos["L"][i,:] = cur["L"] = tmp["L"]
        pos["beta"][i] = cur["beta"] = tmp["beta"]
        pos["phi"][i] = cur["phi"] = tmp["phi"]
		pos["sigmaBeta"][i,:,:] = cur["sigmaBeta"] = update_sigmaBeta(dat, cur, hyper)
		pos["muBeta"][i,:] = cur["muBeta"] = update_muBeta(dat, cur, hyper)
		pos["bPhi"][i] = cur["bPhi"] = update_bPhi(dat, cur, hyper)
	end

	result = Dict("pos" => pos,
				  "hyper" => hyper)

	savefile = config["save_path"] 
	save(savefile, result) 
	
end;
