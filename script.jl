using StateSpaceModels
using CSV
using DataFrames
using Plots
using Statistics
plotly()

# Local level
nile = CSV.read("nile.csv", DataFrame)
plt = plot(nile.year, nile.flow, label = "Nile river annual flow", xlabel = "year", ylabel = "10⁸ m³")

model = LocalLevel(nile.flow)
fit!(model)
nile.flow[[collect(21:40); collect(61:80)]] .= NaN

plt = plot(nile.year, nile.flow, label = "Annual nile river flow", xlabel = "year", ylabel = "10⁸ m³");

model = ARIMA(nile.flow, (1, 0 , 1))
fit!(model)
filter_output = kalman_filter(model)
plot!(plt, nile.year, get_filtered_state(filter_output), label = "Filtered level");

# GAS(1, 1) com inverse scaling Normal e média variando no tempo e assumindo score 0
using ScoreDrivenModels

# sobrescrever o método que assume que se a observação é NaN então o score é 0
function scaling_missing!(score_til::Matrix{T}, t::Int) where T
    for p in axes(score_til, 2)
        score_til[t, p] = zero(T)
    end
    return
end
function ScoreDrivenModels.score_tilde!(score_til::Matrix{T}, y::T, D::Type{<:ScoreDrivenModels.Distribution}, 
                                        param::Matrix{T}, aux::ScoreDrivenModels.AuxiliaryLinAlg{T}, scaling::T, t::Int) where T

    if isnan(y)
        scaling_missing!(score_til, t)
        return
    end
    if scaling == ScoreDrivenModels.SCALINGS[1] # 0.0
        ScoreDrivenModels.scaling_identity!(score_til, y, D, aux, param, t)
    elseif scaling == ScoreDrivenModels.SCALINGS[2] # 1/2
        ScoreDrivenModels.scaling_invsqrt!(score_til, y, D, aux, param, t)
    elseif scaling == ScoreDrivenModels.SCALINGS[3] # 1
        ScoreDrivenModels.scaling_inv!(score_til, y, D, aux, param, t)
    end
    ScoreDrivenModels.NaN2zero!(score_til, t)
    ScoreDrivenModels.big_threshold!(score_til, ScoreDrivenModels.SCORE_BIG_NUM, t)
    ScoreDrivenModels.small_threshold!(score_til, ScoreDrivenModels.SMALL_NUM, t)
    return
end
function ScoreDrivenModels.log_likelihood(::Type{ScoreDrivenModels.Normal}, y::Vector{T}, param::Matrix{T}, n::Int) where T
    loglik = -0.5 * n * log(2 * pi)
    for t in 1:n
        if !isnan(y[t])
            loglik -= 0.5 * (log(param[t, 2]) + (1 / param[t, 2]) * (y[t] - param[t, 1]) ^ 2)
        end
    end
    return -loglik
end

gas = Model(1, 1, Normal, 1.0, time_varying_params=[1])
initial_params = [nile.flow[1] var(filter(!isnan, nile.flow))]
gas.ω[1] = 0
# Fit specified model to historical data
f = ScoreDrivenModels.fit!(gas, nile.flow; opt_method=NelderMead(gas, 10), initial_params=initial_params)
fit_stats(f)
gas_mean_0 = fitted_mean(gas, nile.flow; initial_params=initial_params)
plt = plot(nile.year, nile.flow, label = "Annual nile river flow");
plot!(plt, nile.year, gas_mean_0, label = "time varying mean gas considering s=0 when missing");
plot!(plt, nile.year, get_predictive_state(filter_output)[1:end-1, 1], label = "Predictive state for y (state-space approach)")

function mean_log_lik_simulate_observation(gas, S::Int, T::Int, y::Vector)
    log_liks = zeros(S)
    idx = findall(isnan, y)
    for s in 1:S
        obs, params = simulate_recursion(gas, T)
        obs[idx] .= NaN
        log_liks[s] = -ScoreDrivenModels.log_likelihood(ScoreDrivenModels.Normal, obs, params, T)
    end
    return mean(log_liks)
end
S = 100
T = 100
mean_log_lik_simulate_observation(gas, S, T, nile.flow)
