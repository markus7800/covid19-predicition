using Statistics
using Distributions


# f(x) = L / (1 + exp(-k(x-x0)))
# L: max val
# k: growth rate
# x0: midpoint
f(x,k,L,x0) = L / (1 + exp(-k*(x-x0)))

function loss(x,k,L,x0,y)
    return sum((f.(x,k,L,x0) .- y).^2)
end


function loss_L(L, α, β, x, y)
    y_L = @. log(y / (L-y))
    y_hat = @. α + β * x

    return sum((y_hat .- y_L).^2)
end

function ∇loss_L(L, α, β, x, y)
    y_L = @. log(y / (L-y))
    y_hat = @. α + β * x

    w = @. 1 / (y - L) # = ∇y_L
    w_bar = mean(w)
    ww = (w .- w_bar)

    x_bar  = mean(x)
    xx = (x .- x_bar)

    ∇β = (xx'ww) / (xx'xx)
    ∇α = w_bar - ∇β * x_bar

    return sum(@. 2*(y_hat - y_L) * (∇α + ∇β * x - w))
end

function ∇∇loss_L(L, α, β, x, y)
    y_L = @. log(y / (L-y))
    y_hat = @. α + β * x

    w = @. 1 / (y - L) # = ∇y_L
    w_bar = mean(w)
    ww = (w .- w_bar)

    w2 = w.^2 # = ∇∇y_L
    w2_bar = mean(w2)
    w2w2 = (w2 .- w2_bar)

    x_bar  = mean(x)
    xx = (x .- x_bar)
    c = (xx'xx)

    ∇β = (xx'ww) / c
    ∇α = w_bar - ∇β * x_bar

    ∇∇β = (xx'w2w2) / c
    ∇∇α = w2_bar - ∇∇β * x_bar

    return sum(@. ( 2*(y_hat - y_L) * (∇∇α + ∇∇β * x - w2) ) + ( 2*(∇α + ∇β * x - w)^2 ) )
end

function loss_L(L, x, y)
    α, β = line_fit(L, x, y)
    return loss_L(L, α, β, x, y)
end

function ∇loss_L(L, x, y)
    α, β = line_fit(L, x, y)
    return ∇∇loss_L(L, α, β, x, y)
end

function ∇∇loss_L(L, x, y)
    α, β = line_fit(L, x, y)
    return ∇∇loss_L(L, α, β, x, y)
end

# fits a linear model for a given L
function line_fit(L, x, y)
    @assert all(L .> y)

    y_L = @. log(y / (L-y))

    x_bar  = mean(x)
    y_bar = mean(y_L)
    xx = (x .- x_bar)
    yy = (y_L .- y_bar)
    β = (xx'yy) / (xx'xx)
    α = y_bar - β * x_bar

    return α, β
end

function fit_logistic(x, y)
    L = maximum(y) + mean(y)

    # newton
    last = 0
    n = 0
    η = 1
    while true
        n += 1
        if n % 100 == 0
            η *= 0.5
        end

        α, β = line_fit(L, x, y)
        ∇l = ∇loss_L(L, α, β, x, y)
        ∇∇l = ∇∇loss_L(L, α, β, x, y)

        if abs((∇l - last) / last) < 1e-6 || ∇l == last
            k = β
            x0 = -α/k
            return L, k, x0
        end
        last = ∇l
        L -= η * ∇l / ∇∇l
    end
end

function get_L(α, β, x, y)
    x_bar = mean(x)

    function f(L)
        y_L = @. log(y / (L-y))
        return mean(y_L) - β * x_bar - α
    end

    function ∇f(L)
        w = @. 1 / (y - L)
        return mean(w)
    end

    # newton
    last = 0
    L = maximum(y) + mean(y)
    n = 0
    η = 1
    while true
        n += 1
        if n % 100 == 0
            η *= 0.5
        end

        fx = f(L)
        dfx = ∇f(L)

        if abs((fx - last) / last) < 1e-6 || fx == last
            return L
        end

        last = fx
        L -= η * fx / dfx
    end
end

function confidence_intervals(L, k, x0, x, y, γ=0.05)
    # https://en.wikipedia.org/wiki/Simple_linear_regression#Confidence_intervals

    β = k
    α = -x0 * k

    y_L = @. log(y / (L-y))
    y_hat = @. α + β * x

    x_bar  = mean(x)
    xx = (x .- x_bar)
    c = (xx'xx)

    n = length(y)

    sβ = sqrt(1/(n-2)*sum((y_L .- y_hat).^2) / c)
    sα = sβ * sqrt(mean(x.^2))

    t_star = quantile(TDist(n-2), 1 - γ/2)

    α_max = α + sα * t_star
    α_min = α - sα * t_star

    β_max = β + sβ * t_star
    β_min = β - sβ * t_star

    k_max = β_max # > 0
    k_min = β_min # > 0

    x0_max = -α_min/k_min
    x0_min = -α_max/k_max

    L11 = get_L(α_min, β_min, x, y)
    L12 = get_L(α_min, β_max, x, y)
    L21 = get_L(α_max, β_min, x, y)
    L22 = get_L(α_max, β_max, x, y)

    @assert all(L11 .>= [L12, L21, L22])
    @assert all(L22 .<= [L11, L12, L21])

    return L22, k_max, x0_max, L11, k_min, x0_min
end
