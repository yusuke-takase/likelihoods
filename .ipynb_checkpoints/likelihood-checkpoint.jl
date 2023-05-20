function forecasting(lmax, cl_obs,;
        path="/home/cmb/yusuket/program/MapData/CAMB/ClBB_PTEPini.npz", 
        rmin=1e-8,
        rmax=1e-1,
        rresol=1e5 |> Int,
        iter=0,
        verbose=false,
        test=false,
        bias=1e-5
    )
    #= `Cl_obs requires B-mode power spectrum due to the systematics`=#
    gridOfr = range(rmin, rmax, length=rresol)
    cl_models = npzread(path)
    cl_lens = @views cl_models["lens"]
    cl_tens = @views cl_models["tensor"]
    if test == true
        cl_obs = cl_tens*bias
    end
    ℓ = 2:lmax
    Nₗ = length(ℓ)
    delta_r = 0.0
    likelihood = 0
    gridOfr_old = 0
    @views @inbounds for j in 0:iter
        Nᵣ = length(gridOfr)
        likelihood = zeros(Nᵣ)
        @views @inbounds for i in eachindex(gridOfr)
            Cl_hat = @. cl_obs[3:lmax+1] + cl_lens[3:lmax+1]
            Cl = @. gridOfr[i] * cl_tens[3:lmax+1] + cl_lens[3:lmax+1]
            likelihood[i] = sum(@.((-0.5)*(2*ℓ + 1)*((Cl_hat / Cl) + log(Cl) - ((2*ℓ - 1)/(2*ℓ + 1))*log(Cl_hat))))
        end
        likelihood = @views exp.(likelihood .- maximum(likelihood))
        maxid = findmax(likelihood)[2]
        delta_r = gridOfr[maxid]
        survey_range = @views [delta_r - delta_r*(0.5/(j+1)), delta_r + delta_r*(0.5/(j+1))]
        gridOfr_old = gridOfr
        gridOfr = range(survey_range[1], survey_range[2], length=Int(1e4))
        if verbose == true
            println("*--------------------------- iter = $j ---------------------------*")
            println(" Δr                : $delta_r")
            println(" Next survey range : $survey_range")
        end
    end
    gridOfr_old = range(delta_r*1e-3, delta_r*3, length=Int(1e4))
    Nᵣ = length(gridOfr_old)
    likelihood = zeros(Nᵣ)
    @views @inbounds for i in eachindex(gridOfr_old)
        Cl_hat = @. cl_obs[3:lmax+1] + cl_lens[3:lmax+1]
        Cl = @. gridOfr_old[i] * cl_tens[3:lmax+1] + cl_lens[3:lmax+1]
        likelihood[i] = sum(@.((-0.5)*(2*ℓ + 1)*((Cl_hat / Cl) + log(Cl) - ((2*ℓ - 1)/(2*ℓ + 1))*log(Cl_hat))))
    end
    likelihood = @views exp.(likelihood .- maximum(likelihood))
    return delta_r, gridOfr_old, likelihood
end