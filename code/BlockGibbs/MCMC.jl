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
include("./update_p.jl")
include("./update_epsilon.jl") 
include("./update_baseline.jl") 
include("./update_alpha.jl") 



function MCMC(config_file)
	"""
	config_file: a TOML configuration file in the folder ./configs
	"""
	
	config = TOML.parsefile(config_file)

	nsam = config["nsam"]
	dat = load(config["data_file"])
	hyper = config["hyper"]
    
	n = length(dat["survival"]) 

	J, N = hyper["J"], hyper["N"] 

	cur = Dict(
		# initialization of all parameters 
		"U_phi" => fill(5.0, N),
		"U_beta" => fill(0.0, N, J),
		"L" => fill(1, n),
		"logp" => fill(log(1/N), N),
		"epsilon" => fill(0.1, n), 
		"muBeta" => hyper["sBeta"] .* ones(J),
		"sigmaBeta" => hyper["CBeta"] .* Matrix(Diagonal(ones(J))), 
		"bPhi" => 1.0,
		"alpha" => 5.0,
		)

    # posterior values
    pos = Dict(
		"U_phi" => Matrix{Float64}(undef, nsam, N),
		"U_beta" => Array{Float64}(undef, nsam, N, J),
		"L" => Matrix{Int64}(undef, nsam, n),
		"logp" => Matrix{Float64}(undef, nsam, N),
		"epsilon" => Matrix{Float64}(undef, nsam, n),
		"sigmaBeta" => Array{Float64}(undef, nsam, J, J), 
		"muBeta" => Matrix{Float64}(undef, nsam, J),
		"bPhi" => Vector{Float64}(undef, nsam),
		"alpha" => Vector{Float64}(undef, nsam),
	)

	@showprogress for i in (1:nsam)
		# Gibbs sampler 
		pos["epsilon"][i,:] = cur["epsilon"] = update_epsilon(dat, cur)
		pos["U_beta"][i,:,:] = cur["U_beta"] = update_U_beta(dat, cur, hyper)
		pos["U_phi"][i,:] = cur["U_phi"] = update_U_phi(dat, cur, hyper)
		pos["L"][i,:] = cur["L"] = update_L(dat, cur, hyper)
		pos["logp"][i,:] = cur["logp"] = update_p(cur, hyper)
        pos["alpha"][i] = cur["alpha"] = update_alpha(cur, hyper)
		pos["sigmaBeta"][i,:,:] = cur["sigmaBeta"] = update_sigmaBeta(dat, cur, hyper)
		pos["muBeta"][i,:] = cur["muBeta"] = update_muBeta(dat, cur, hyper)
		pos["bPhi"][i] = cur["bPhi"] = update_bPhi(dat, cur, hyper)
	end

	result = Dict("pos" => pos,
				  "hyper" => hyper)

	savefile = config["save_path"] # * "surv_only_fit_" * config["datafile"]
	save(savefile, result) 
	
end;

