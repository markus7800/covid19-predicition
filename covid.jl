using Plots
using Dates
using Measures

include("logistic_fit.jl")
include("logistic_derivative_fit.jl")

function annotate_fit!(start_date,n,L,x0,x_max; show_text=true)
    current = Dates.format(start_date + Day(n), "YY-mm-d")
    vline!([n-1], ls=:dot, lc=:black, label="Today")

    infl = Dates.format(start_date + Day(ceil(x0)), "YY-mm-d")
    vline!([x0], label="Predicted Inflection Point", lc=:blue)
    s = Dates.format(start_date + Day(ceil(2*x0)), "YY-mm-d")
    vline!([ceil(2*x0)], label="Predicted End of Crisis", lc=:green)
    if show_text
        annotate!(x_max+2, L, text("Max. Infected Prediction: $(Int(ceil(L)))", 10, halign=:left))
        annotate!(x0+2, 100, text("Predicted Inflection Point: $infl", 10, halign=:left))
        annotate!(2*x0+2, 100, text("Predicted End of Crisis: $s", 10, halign=:left))
    end
end

function total_infected_plot(start_date,x,y,L,k,x0,prob)
    L_l, k_l, x0_l, L_u, k_u, x0_u = confidence_intervals(L, k, x0, x, y, 1-prob)

    x_max = 2 * max(x0, x0_l, x0_u)

    start = Dates.format(start_date, "YY-mm-d")
    current = Dates.format(start_date + Day(length(y)), "YY-mm-d")
    best(t) = f(t, k, L, x0)
    lower(t) = f(t, k_l, L_l, x0_l)
    upper(t) = f(t, k_u, L_u, x0_u)

    range = 0:0.1:Int(ceil(x_max))
    p1 = plot(range, best.(range),
        ribbon=(best.(range) .- lower.(range) , upper.(range).- best.(range)),
        fillalpha=.3, label="Logistic Fit with $(prob * 100)% bounds",
        xlims=(0,x_max+20), lc=2, fc=2)

    scatter!(x, y, label="Infected Persons", legend=:bottomright,
            xlabel="Days since $start", ylabel="Total Infected",
            title="Austria-Covid-19 Total Prediction (as of $current)", size=(1000, 700), mc=1)

    annotate_fit!(start_date,length(y),L,x0,x_max)

    annotate!(x_max+2, L_l, text("Max. Infected Best Case: $(Int(ceil(L_l)))", 10, halign=:left))
    annotate!(x_max+2, L_u, text("Max. Infected Worst Case: $(Int(ceil(L_u)))", 10, halign=:left))

    p2 = plot(t -> k*(t-x0) + log(L/2), label="Inflection line", xlims=(0,x_max), lc=2)
    scatter!(x, log.(y), label="Infected Persons", legend=:bottomright,
            xlabel="Days since $start", ylabel="Log(Total Infected)",
            title="Austria-Covid-19 Growth Rate (as of $current)", size=(1000, 700), mc=1)

    annotate_fit!(start_date,length(y),L,x0,x_max)

    return p1, p2
end

function new_infected_plot(start_date,x,y,L,k,x0)
    x_max = 2 * x0
    y_new = diff(vcat([0],y))

    start = Dates.format(start_date, "YY-mm-d")
    current = Dates.format(start_date + Day(length(y)), "YY-mm-d")

    range = 0:0.1:Int(ceil(x_max))
    p1 = plot(range, ∇f_x.(range,k,L,x0), label="Logistic Derivative Fit", lc=2)
    scatter!(x, y_new, label="New Infected Persons", legend=:bottomright,
            xlabel="Days since $start", ylabel="New Infected Per Day",
            title="Austria-Covid-19 New Prediction (as of $current)", size=(1000, 700), mc=1)
    annotate_fit!(start_date,length(y),L,x0,x_max,show_text=false)

    p2 = plot(t->f(t,k,L,x0), 0, x_max, label="Logistic Derivative Fit",  xlims=(0,x_max+20), lc=2)
    scatter!(x, y, label="Infected Persons", legend=:bottomright,
            xlabel="Days since $start", ylabel="Total Infected",
            title="Austria-Covid-19 Total Prediction (as of $current)", size=(1000, 700), mc=1)

    annotate_fit!(start_date,length(y),L,x0,x_max)

    return p1, p2
end

function daily_prediction(y, start_date=DateTime(2020,2,26), prob=0.95)
    x = Array{Int}(0:length(y)-1)
    L,k,x0 = fit_logistic(x, y)
    dL,dk, dx0 = fit_logistic_derivative(x,y,L,k,x0)

    p11, p12 = total_infected_plot(start_date,x,y,L,k,x0,prob)

    # p21, p22 = new_infected_plot(start_date,x,y,dL,dk,dx0)

    # p = plot(p11,p12,p21,p22, layout=(2,2), size=(3000,1800), margin=20mm)

    p = plot(p11,p12, layout=(1,2), size=(3000,900), margin=20mm)

    current = Dates.format(start_date + Day(length(y)), "YY_mm_d")
    savefig("Predictions/Prediction_$(current).png")

    return p
end

# data_24_3 = [2, 5, 10, 10, 13, 18, 26, 38, 53, 74, 91, 122, 147, 213, 316, 441, 628, 809, 1016, 1306, 1646, 2057, 2503, 3010, 3405, 3973, 4632]
# daily_prediction(data_24_3)

data_25_3 = [2, 5, 11, 11, 14, 19, 27, 40, 56, 77, 94, 125, 150, 216, 319, 444, 632, 814, 1021, 1306, 1646, 2063, 2514, 3025, 3427, 4010, 4725, 5341]
daily_prediction(data_25_3)

y = data_25_3
x = Array{Int}(0:length(y)-1)
L,k,x0 = fit_logistic(x, y)

f(length(y)-1 + 1, k, L, x0)


# 25.03. 15:00
Infected = [2,2,3,6,10,14,18,21,27,39,53,77,100,138,178,242,356,497,648,853,1007,1320,1633,1998,2373,2798,3219,3894,4892,5549]
Recovered = [0,0,0,0, 0, 0, 0, 0, 2, 2, 2, 2,  2,  2,  4,  4,  4,  6,  6,  6,   6,   9,   9,   9,   9,   9,   9,   9,   9,   9]
Dead = [0,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  1,  1,  1,  1,   3,   3,   4,   6,   6,   7,  16,  21,  25,  30]

y1 = Infected
y2 = Recovered .+ Dead

fit_SEIR(y1,y2)
