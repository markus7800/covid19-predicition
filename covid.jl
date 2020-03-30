using Plots
using Dates
using Measures

include("logistic_fit.jl")
include("logistic_derivative_fit.jl")
include("SIR.jl")

function annotate_fit!(start_date,n,L,x0,x_max; show_text=true)
    current = Dates.format(start_date + Day(n), "YY-mm-d")
    vline!([n-1], ls=:dot, lc=:black, label="Today")

    infl = Dates.format(start_date + Day(ceil(x0)), "YY-mm-d")
    vline!([x0], label="Predicted Inflection Point", lc=:blue)
    s = Dates.format(start_date + Day(ceil(2*x0)), "YY-mm-d")
    vline!([ceil(2*x0)], label="Predicted End of Crisis", lc=:green)
    if show_text
        annotate!(x_max+2, L, text("Max. Affected Prediction: $(Int(ceil(L)))", 10, halign=:left))
        annotate!(x0+2, 100, text("Predicted Inflection Point: $infl", 10, halign=:left))
        annotate!(2*x0+2, 100, text("Predicted End of Crisis: $s", 10, halign=:left))
    end
end

function total_affected_plot(start_date,x,y,L,k,x0,prob)
    L_l, k_l, x0_l, L_u, k_u, x0_u = confidence_intervals(L, k, x0, x, y, 1-prob)

    x_max = 2 * max(x0, x0_l, x0_u)

    start = Dates.format(start_date, "YY-mm-d")
    current = Dates.format(start_date + Day(length(y)-1), "YY-mm-d")
    best(t) = f(t, k, L, x0)
    lower(t) = f(t, k_l, L_l, x0_l)
    upper(t) = f(t, k_u, L_u, x0_u)

    range = 0:0.1:Int(ceil(x_max))
    p1 = plot(range, best.(range),
        ribbon=(best.(range) .- lower.(range) , upper.(range).- best.(range)),
        fillalpha=.3, label="Logistic Fit with $(prob * 100)% bounds",
        xlims=(0,x_max+20), lc=2, fc=2)

    scatter!(x, y, label="Affected Persons", legend=:bottomright,
            xlabel="Days since $start", ylabel="Total Affected",
            title="Austria-Covid-19 Total Prediction (as of $current)", size=(1000, 700), mc=1)

    annotate_fit!(start_date,length(y),L,x0,x_max)

    annotate!(x_max+2, L_l, text("Max. Affected Best Case: $(Int(ceil(L_l)))", 10, halign=:left))
    annotate!(x_max+2, L_u, text("Max. Affected Worst Case: $(Int(ceil(L_u)))", 10, halign=:left))

    p2 = plot(t -> k*(t-x0) + log(L/2), label="Inflection line", xlims=(0,x_max), lc=2)
    scatter!(x, log.(y), label="Affected Persons", legend=:bottomright,
            xlabel="Days since $start", ylabel="Log(Total Affected)",
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

function SIR_prediction(start_date,I,R, μ, months=6)
    x = Array(0:length(I)-1)
    r = fit_exact_IR_1(x,I,R)
    i0 = I[1]
    (s0, t0, γ, R0) = r.minimizer

    b = γ * R0 / (R0 - 1)
    c = γ / (R0 - 1)

    pred_I(t) = exact_I(t, t0, b, c, s0, i0)
    pred_R(t) = exact_R(t, t0, b, c, s0, i0)
    pred_D(t) = μ * exact_R(t, t0, b, c, s0, i0)

    # x_max = length(x)
    # while pred_I(x_max) > 1000
    #     x_max += 1
    # end
    x_max = length(x) + 30 * months
    current = Dates.format(start_date + Day(length(I)-1), "d-mm-YY")

    p1 = plot(pred_I, 0, x_max, label="Predicted Infected")
    title!("SIR prediction as of $current")
    plot!(pred_R, label="Predicted Removed")
    plot!(pred_D, label="Predicted Dead")
    scatter!(x, I, label="Infected")
    scatter!(x, R, label="Removed")

    is = []
    for i in 0:x_max
        d = start_date + Day(i)
        if day(d) == 1
            push!(is, i-0.5)
            m = Dates.format(d, "U")
            t = text(m, font(5, rotation=90.0), halign=:left)
            annotate!(i+5, 0, t)
        end
    end
    vline!(is, ls=:dot, lc=:black, label="")

    p2 = scatter(x, I, label="Infected", mc=4, legend=:topleft)
    title!("Goodness of fit for Infected")
    plot!(pred_I, label="Predicted Infected",lc=1)

    p3 = scatter(x, R, label="Removed",mc=5, legend=:topleft)
    title!("Goodness of fit for Removed")
    plot!(pred_R, label="Predicted Removed",lc=2)

    l = @layout [a ; b c]
    p = plot(p1,p2,p3, layout=l, size=(1000,1000))

    current = Dates.format(start_date + Day(length(I)-1), "YY_mm_d")
    savefig("Predictions/SIR_Prediction_$(current).png")

    return p
end

function daily_prediction(Infected, Recovered, Dead, start_date=DateTime(2020,2,25), prob=0.95)
    y = Infected .+ Recovered .+ Dead
    x = Array{Int}(0:length(y)-1)
    L,k,x0 = fit_logistic(x, y)

    p11, p12 = total_affected_plot(start_date,x,y,L,k,x0,prob)

    p = plot(p11,p12, layout=(2,1), size=(1800,1800), margin=20mm)

    current = Dates.format(start_date + Day(length(y)-1), "YY_mm_d")
    savefig("Predictions/Prediction_$(current).png")

    return p, p11, t -> f(t, k,L,x0)
end


# 25.03. 15:00
Infected =  [2,2,3,6,10,14,18,21,27,39,53,77,100,138,178,242,356,497,648,853,1007,1320,1633,1998,2373,2798,3219,3894,4892,5549]
Recovered = [0,0,0,0, 0, 0, 0, 0, 2, 2, 2, 2,  2,  2,  4,  4,  4,  6,  6,  6,   6,   9,   9,   9,   9,   9,   9,   9,   9,   9]
Dead =      [0,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  1,  1,  1,  1,   3,   3,   4,   6,   6,   7,  16,  21,  25,  30]
p, p1, base = daily_prediction(Infected, Recovered, Dead)
display(p1)

# 26.03 15:00
#Dates =    [25.2.26.2,27.2,28.2,29.2,01.3,02.3,03.3,04.3,05.3,06.3,07.3,08.3,09.3,10.3,11.3,12.3,13.3,14.3,15.3,16.3,17.3,18.3,19.3,20.3,21.3,22.3,23.3,24.3,25.3.26.3,27.3,28.3,29.3]
Infected =  [   2,   2,   3,   6,  10,  14,  18,  21,  27,  39,  53,  77, 100, 138, 178, 242, 356, 497, 648, 853,1007,1320,1633,1998,2373,2793,3214,3889,4887,5544,6335,7158,7552,7971]
Recovered = [   0,   0,   0,   0,   0,   0,   0,   0,   2,   2,   2,   2,   2,   2,   4,   4,   4,   6,   6,   6,   6,   9,   9,   9,   9,   14,   14,  14,  14,   14,  14, 225, 410, 479]
Dead =      [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   1,   3,   3,   4,   6,   6,   7,  16,  21,  25,  30,  49,  58,  68,  86]
p, p1, base = daily_prediction(Infected, Recovered, Dead)
display(p1)

function smooth_recovered(Recovered, Dead)
    weights = Dead ./ sum(Dead)

    return sum(Recovered) .* weights
end

function estimate_μ(Recovered,Dead)
    return sum(Dead) / (sum(Dead) + sum(Recovered))
end

I = Infected
R = smooth_recovered(Recovered, Dead) .+ Dead
μ = estimate_μ(Recovered, Dead)

SIR_prediction(Date(2020,2,25),I,R,μ)


x = Array(0:length(I)-1)
r = fit_exact_IR_1(x,I,R)
i0 = I[1]
(s0, t0, γ, R0) = r.minimizer

b = γ * R0 / (R0 - 1)
c = γ / (R0 - 1)

scatter(x, I)
plot!(t -> exact_I(t, t0, b, c, s0, i0))

scatter(x, R)
plot!(t -> exact_R(t, t0, b, c, s0, i0))


plot(t -> exact_I(t, t0, b, c, s0, i0),0,200)
plot!(t -> exact_R(t, t0, b, c, s0, i0),0,200)
scatter!(x, I)
scatter!(x, R)
