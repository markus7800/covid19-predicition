using Plots
using Dates

include("logistic_fit.jl")
include("logistic_derivative_fit.jl")

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

function new_infected_plot(start_date,x,y,L,k,x0)
    x_max = 2 * x0
    y_new = diff(vcat([0],y))

    start = Dates.format(start_date, "YY-mm-d")
    current = Dates.format(start_date + Day(length(y)), "YY-mm-d")

    p1 = scatter(x, y_new, label="New Infected Persons", legend=:topleft,
            xlabel="Days since $start", ylabel="New Infected Per Day",
            title="Austria-Covid-19 Prediction (as of $current)", size=(1000, 700))
    range = 0:0.1:Int(ceil(x_max))
    plot!(range, ∇f_x.(range,k,L,x0))
end

function daily_prediction(y, start_date=DateTime(2020,2,26), prob=0.95)
    x = Array{Int}(0:length(y)-1)
    L,k,x0 = fit_logistic(x, y)
    dL,dk, dx0 = fit_d(x,y,L,k,x0, 1000)

    # L_l, k_l, x0_l, L_u, k_u, x0_u = confidence_intervals(L, k, x0, x, y, 1-prob)
    #
    # p1 = total_infected_plot(start_date,x,y,L,k,x0,L_l,k_l,x0_l,L_u,k_u,x0_u,prob)
    #
    # p2 = new_infected_plot(start_date,x,y,dL,dk,dx0)
    # current = Dates.format(start_date + Day(length(y)), "YY_mm_d")
    # savefig("Predictions\Prediction_$(current).png")

    return p2
end

# data_24_3 = [2, 5, 10, 10, 13, 18, 26, 38, 53, 74, 91, 122, 147, 213, 316, 441, 628, 809, 1016, 1306, 1646, 2057, 2503, 3010, 3405, 3973, 4632]
# daily_prediction(data_24_3)

data_25_3 = [2, 5, 10, 10, 13, 18, 26, 38, 53, 74, 91, 122, 147, 213, 316, 441, 629, 810, 1017, 1307, 1648, 2060, 2510, 3019, 3419, 3991, 4678, 5246]
daily_prediction(data_25_3)
