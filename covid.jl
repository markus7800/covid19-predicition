using Plots
using Statistics
using Dates
using Distributions

# f(x) = L / (1 + exp(-k(x-x0)))
# L: max val
# k: growth rate
# x0: midpoint


f(x,k,L,x0) = L / (1 + exp(-k*(x-x0)))

function loss(x,k,L,x0,y)
    return sum((f.(x,k,L,x0) .- y).^2)
end

∇f_x(x,k,L,x0) = (L*k*exp(-k*(x-x0))) / ((1 + exp(-k*(x-x0)))^2)
∇f_L(x,k,L,x0) =  1 / (1 + exp(-k*(x-x0)))
∇f_k(x,k,L,x0) = (L*(x-x0)*exp(-k*(x-x0))) / ((1 + exp(-k*(x-x0)))^2)
∇f_x0(x,k,L,x0) =  (-L*k*exp(-k*(x-x0))) / ((1 + exp(-k*(x-x0)))^2)

∇f_xL(x,k,L,x0) = (k*exp(-k*(x-x0))) / ((1 + exp(-k*(x-x0)))^2)
function ∇f_xk(x,k,L,x0)
    v = exp(-k*(x-x0))
    return (2*L*k*(x-x0) * v^2 / (1+v)^3) + ( L*v / (1+v)^2) - (L*k*(x-x0)*v / (1+v)^2)
end
function ∇f_xx0(x,k,L,x0)
    v = exp(-k*(x-x0))
    return -2*L*k^2*v^2 / (1+v)^3 + (L*k^2*v / (1+v)^2)
end

∇dloss_L(x,k,L,x0,y) = sum(2*(∇f_x.(x,k,L,x0) .- y) .* ∇f_xL.(x,k,L,x0))
∇dloss_k(x,k,L,x0,y) = sum(2*(∇f_x.(x,k,L,x0) .- y) .* ∇f_xk.(x,k,L,x0))
∇dloss_x0(x,k,L,x0,y) = sum(2*(∇f_x.(x,k,L,x0) .- y) .* ∇f_xx0.(x,k,L,x0))

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

function loss_L(L, x, y)
    α, β = line_fit(L, x, y)
    return loss_L(L, α, β, x, y)
end

function ∇loss_L(L, x, y)
    α, β = line_fit(L, x, y)
    return ∇∇loss_L(L, α, β, x, y)
end

function ∇loss_L(L, x, y)
    α, β = line_fit(L, x, y)
    return ∇∇loss_L(L, α, β, x, y)
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

function fit_logistic(x, y)
    L = maximum(y) + mean(y)

    # newton
    last = 0
    while true
        α, β = line_fit(L, x, y)
        ∇l = ∇loss_L(L, α, β, x, y)
        ∇∇l = ∇∇loss_L(L, α, β, x, y)

        if abs((∇l - last) / last) < 1e-6 || ∇l == last
            k = β
            x0 = -α/k
            return L, k, x0
        end
        last = ∇l
        L -= ∇l / ∇∇l
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

function total_infected_plot(start_date, x,y,L,k,x0,L_l,k_l,x0_l,L_u,k_u,x0_u,prob)
    x_max = 2 * max(x0, x0_l, x0_u)

    start = Dates.format(start_date, "YY-mm-d")
    current = Dates.format(start_date + Day(length(y)), "YY-mm-d")

    p1 = scatter(x, y, label="Infected Persons", legend=:topleft,
            xlabel="Days since $start", ylabel="Total Infected",
            title="Austria-Covid-19 Prediction (as of $current)", size=(1000, 700))

    best(t) = f(t, k, L, x0)
    lower(t) = f(t, k_l, L_l, x0_l)
    upper(t) = f(t, k_u, L_u, x0_u)

    range = 0:0.1:Int(ceil(x_max))
    plot!(range, best.(range),
        ribbon=(best.(range) .- lower.(range) , upper.(range).- best.(range)),
        fillalpha=.3, label="Logistic Fit with $(prob * 100)% bounds",
        xlims=(0,x_max+20))

    # hline!([L], label="Maximum infizierte Personen Prognose")
    infl = Dates.format(start_date + Day(ceil(x0)), "d.mm.YY")
    vline!([x0], label="Predicted Inflection Point")
    s = Dates.format(start_date + Day(ceil(2*x0)), "d.mm.YY")
    p = vline!([ceil(2*x0)], label="Predicted End of Crisis")

    annotate!(x_max+12, L, text("Max. Infected Prediction: $(Int(ceil(L)))", 10))
    annotate!(x_max+12, L_l, text("Max. Infected Best Case: $(Int(ceil(L_l)))", 10))
    annotate!(x_max+12, L_u, text("Max. Infected Worst Case: $(Int(ceil(L_u)))", 10))

    annotate!(x0+12, 100, text("Predicted Inflection Point: $infl", 10))
    annotate!(2*x0+12, 100, text("Predicted End of Crisis: $s", 10))

    return p
end

function new_infected_plot(start_date,x,y,L,k,x0,L_l,k_l,x0_l,L_u,k_u,x0_u,prob)
    x_max = 2 * max(x0, x0_l, x0_u)
    y_new = diff(vcat([0],y))

    start = Dates.format(start_date, "YY-mm-d")
    current = Dates.format(start_date + Day(length(y)), "YY-mm-d")

    p1 = scatter(x, y_new, label="New Infected Persons", legend=:topleft,
            xlabel="Days since $start", ylabel="Total Infected",
            title="Austria-Covid-19 Prediction (as of $current)", size=(1000, 700))
    range = 0:0.1:Int(ceil(x_max))
    plot!(range, ∇f_x.(range,k,L,x0))
end

function daily_prediction(y, start_date=DateTime(2020,2,26), prob=0.95)
    x = Array{Int}(0:length(y)-1)
    L,k,x0 = fit_logistic(x, y)

    L_l, k_l, x0_l, L_u, k_u, x0_u = confidence_intervals(L, k, x0, x, y, 1-prob)

    p1 = total_infected_plot(start_date,x,y,L,k,x0,L_l,k_l,x0_l,L_u,k_u,x0_u,prob)

    p2 = new_infected_plot(start_date,x,y,L,k,x0,L_l,k_l,x0_l,L_u,k_u,x0_u,prob)
    # current = Dates.format(start_date + Day(length(y)), "YY_mm_d")
    # savefig("Prediction_$(current).png")

    return p2
end

# data_24_3 = [2, 5, 10, 10, 13, 18, 26, 38, 53, 74, 91, 122, 147, 213, 316, 441, 628, 809, 1016, 1306, 1646, 2057, 2503, 3010, 3405, 3973, 4632]
# daily_prediction(data_24_3)

data_25_3 = [2, 5, 10, 10, 13, 18, 26, 38, 53, 74, 91, 122, 147, 213, 316, 441, 629, 810, 1017, 1307, 1648, 2060, 2510, 3019, 3419, 3991, 4678, 5246]
daily_prediction(data_25_3)
