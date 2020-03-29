using ProgressMeter
using Optim

function exact_SIR(t, b, c, s0, i0)
    # https://arxiv.org/pdf/1812.09759.pdf
    κ = i0 / s0
    c = c
    γ = b / (b-c)

    return i0*(1+κ)^γ * exp((b-c)*t) / (1+κ*exp((b-c)*t))^γ
end

function ∇exact_SIR_s0(t, b, c, s0, i0)
    κ = i0 / s0
    c = c
    γ = b / (b-c)

    e = exp((b-c)*t)
    u = (1 + κ)^γ
    v = 1/ (1 + κ * e)^γ

    a1 = ( i0^2 * u * b * e * v ) / ( (b-c) * s0^2 * (1+κ) )
    a2 = ( i0^2 * u * e^2 * v * b) / ( (b-c) * s0^2 * (1+κ*e) )

    return -a1+a2
end

function ∇exact_SIR_b(t, b, c, s0, i0)
    if b <= c
        return 1
    end

    κ = i0 / s0
    c = c
    γ = b / (b-c)

    e = exp((b-c)*t)
    u = (1 + κ)^γ
    v = 1/ (1 + κ * e)^γ

    a1 = i0 * u * (1/(b-c) - b/((b-c)*(b-c))) * log(1+κ) * e * v
    a2 = i0 * u * e * v
    a3 = -(1/(b-c) - b/((b-c)*(b-c)))*log(1+κ*e) - (b*i0*t*e) / ((b-c)*s0*(1+κ*e))

    return a1 + t * a2 + a2 * a3
end

function ∇exact_SIR_c(t, b, c, s0, i0)
    if b <= c
        return -1
    end

    κ = i0 / s0
    c = c
    γ = b / (b-c)

    e = exp((b-c)*t)
    u = (1 + κ)^γ
    v = 1/ (1 + κ * e)^γ

    a1 = (i0 * u * b * log(1+κ) * e * v) / (b-c)^2
    a2 = i0 * u * t * e * v
    a3 = i0 * u * e * v
    a4 = -(b * log(1+κ*e)) / ((b-c)*(b-c)) + (b*i0*t*e) / ((b-c)*s0*(1+κ*e))

    return a1 - a2 + a3*a4
end

exact_SIR_loss(x,y,b,c,s0,i0) = 1/2 * sum((exact_SIR.(x,b,c,s0,i0) .- y).^2)

function ∇exact_SIR_loss(x,y,b,c,s0,i0)
    w = exact_SIR.(x,b,c,s0,i0) .- y
    ∇s0 = sum(w .* ∇exact_SIR_s0.(x,b,c,s0,i0))
    ∇b = sum(w .* ∇exact_SIR_b.(x,b,c,s0,i0))
    ∇c = sum(w .* ∇exact_SIR_c.(x,b,c,s0,i0))

    return ∇s0, ∇b, ∇c
end

function fit_exact_SIR_1(x,y)
    i0 = y[1]
    F((s0,b,c)) = exact_SIR_loss(x,y,b,c,s0,i0)
    optimize(F, [10000, 0.5, 1])
end
F((s0,b,c)) = exact_SIR_loss(x,y,b,c,s0,i0)
function fit_exact_SIR_3(x,y)
    i0 = y[1]
    F((s0,b,c)) = exact_SIR_loss(x,y,b,c,s0,i0)

    lower = Float64[maximum(y),0,0]
    upper = Float64[9*1e6, 1, 0.1]

    # optimize(F, [10000, 0.5, 0.1], ParticleSwarm(lower=lower,upper=upper,n_particles=100))

    optimize(F, lower, upper, [50000, 0.5, 0.1], SAMIN(), Optim.Options(iterations=10^6))
end

function fit_exact_SIR_4(x,y,R0)
    i0 = y[1]
    # γ = b-c
    # R0 = b/c
    # b = γ * R0/(1-R0)
    # c = γ /(R0 - 1)
    F((s0,γ)) = exact_SIR_loss(x,y,γ*R0/(R0-1),γ/(R0-1),s0,i0)

    lower = Float64[maximum(y),0]
    upper = Float64[9*1e6, Inf]

    # optimize(F, [10000, 0.5, 0.1], ParticleSwarm(lower=lower,upper=upper,n_particles=100))
    optimize(F, [50000, 0.1])
    # optimize(F, lower, upper, [50000, 0.1], SAMIN(), Optim.Options(iterations=10^6))
end

function fit_exact_SIR_5(x,y)
    i0 = y[1]
    # γ = b-c
    # R0 = b/c
    # b = γ * R0/(1-R0)
    # c = γ /(R0 - 1)
    F((s0,R0,γ)) = exact_SIR_loss(x,y,γ*R0/(R0-1),γ/(R0-1),s0,i0)

    lower = Float64[maximum(y),0]
    upper = Float64[9*1e6, Inf]

    # optimize(F, [10000, 0.5, 0.1], ParticleSwarm(lower=lower,upper=upper,n_particles=100))
    optimize(F, [50000, 2, 0.1])
    # optimize(F, lower, upper, [50000, 0.1], SAMIN(), Optim.Options(iterations=10^6))
end

function fit_exact_SIR_2(x,y)
    i0 = y[1]
    F((s0,b,c)) = exact_SIR_loss(x,y,b,c,s0,i0)
    function g!(G, (s0,b,c))
        ∇s0, ∇b, ∇c = ∇exact_SIR_loss(x,y,b,c,s0,i0)
        G[1] = ∇s0
        G[2] = ∇b
        G[3] = ∇c
    end

    optimize(F, g!, [10000, 0.5, 0.2], BFGS())
end

function fit_exact_SIR(x,y)
    i0 = y[1]
    F((s0,b,c)) = exact_SIR_loss(x,y,b,c,s0,i0)
    lower = [maximum(y),0,0]
    upper = [Inf, 1, Inf]
    inner_optimizer = GradientDescent()
    optimize(F, lower, upper, [10000, 0.5, 0.1], Fminbox(inner_optimizer))
end
""
# function fit_exact_SIR(x,y,frac=5,N=3)
#     b0 = 0
#     b1 = 1
#     c0 = 0
#     c1 = 1
#     s00 = maximum(y)
#     s01 = 100000
#
#     i0 = y[1]
#
#     function search_min(s00,s01,b0,b1,c0,c1,n)
#         println("Search in ($s00, $s01)x($b0, $b1)x($c0, $c1)")
#         Δs0 = (s01 - s00) / 50
#         Δb = (b1 - b0) / 1000
#         Δc = (c1 - c0) / 1000
#
#         arg_min = (0, 0, 0)
#         min_val = Inf
#         @showprogress for s0 in s00:Δs0:s01, b in b0:Δb:b1, c in c0:Δc:c1
#             v = exact_SIR_loss(x, y, b, c, s0, i0)
#             if  v < min_val
#                 min_val = v
#                 arg_min = (s0, b, c)
#             end
#         end
#         return arg_min, min_val
#     end
#
#     # (s0,b,c), min_val = search_min(s00,s01,b0,b1,c0,c1,1000)
#
#     (s0,b,c) = (0,0,0)
#     n = 0
#     while n <= N
#         n += 1
#
#         (s0,b,c), min_val = search_min(s00,s01,b0,b1,c0,c1,100)
#
#         s00 = max(maximum(y),s0 - (s01-s00)/frac)
#         s01 = s0 + (s01-s00)/frac
#
#         # b0 = max(0, b - (b1-b0)/frac)
#         # b1 = min(1, b + (b1-b0)/frac)
#         #
#         # c0 = max(0, c - (c1-c0)/frac)
#         # c1 = min(10, c + (c1-c0)/frac)
#
#         last = min_val
#
#         println("$s0, $b, $c: $min_val")
#     end
#
#     return b, c, i0, s0
# end

function SEIR(T, β, γ, σ, I0=1)
    N = 8.822 * 1e6
    S = zeros(T)
    E = zeros(T)
    I = zeros(T)
    R = zeros(T)

    S[1] = N
    I[1] = I0

    for t in 2:T
        S[t] = S[t-1] - (β*S[t-1]*I[t-1]/N) #* 1e6
        E[t] = E[t-1] + (β*S[t-1]*I[t-1]/N - σ*E[t-1]) #* 1e6
        I[t] = I[t-1] + (σ*E[t-1] - γ*I[t-1]) #* 1e6
        R[t] = R[t-1] + (γ*I[t-1]) #* 1e6

        # println(S[t]+E[t]+I[t]+R[t])
    end

    # println(S)
    # println(E)
    # println(I)
    # println(R)

    return I, R
end

# a, b = SEIR(30, 4.3436420293955645e-7, 4.068150510289786e-9, 1.9727396235421875e-6)
a, b = SEIR(1000, 0.1, 0.01, 1.0)
plot(a, label="Infiziert")
plot!(b,label="Nicht mehr Infiziert")
# scatter!(y1)
# scatter!(y2)

function SEIRloss(y1,y2,β,γ,σ)
    I, R = SEIR(length(y1),β,γ,σ)
    return sum((I.-y1).^2) + sum((R.-y2).^2)
end

function fit_SEIR(y1,y2,prec=1e-10,frac=2)
    β0 = 0
    β1 = 1
    γ0 = 0
    γ1 = 1
    σ0 = 0
    σ1 = 1

    function search_min(β0,β1,γ0,γ1,σ0,σ1,n)
        println("Search in ($β0, $β1)x($γ0, $γ1)x($σ0, $σ1)")
        Δβ = (β1 - β0) / n
        Δγ = (γ1 - γ0) / n
        Δσ = (σ1 - σ0) / n

        arg_min = (0, 0, 0)
        min_val = Inf
        for β in β0:Δβ:β1, γ in γ0:Δγ:γ1, σ in σ0:Δσ:σ1
            v = SEIRloss(y1, y2, β, γ, σ)
            if  v < min_val
                min_val = v
                arg_min = (β, γ, σ)
            end
        end
        return arg_min, min_val
    end

    (β,γ,σ) = (0.5,0.5,0.5)

    N = ceil((log(1 - 0) - log(prec)) / log(frac))
    n = 0
    while n <= N
        n += 1

        (β,γ,σ) , min_val = search_min(β0,β1,γ0,γ1,σ0,σ1,100)

        β0 = max(0, β - (β1-β0)/frac)
        β1 = min(1, β + (β1-β0)/frac)

        γ0 = max(0, γ - (γ1-γ0)/frac)
        γ1 = min(1, γ + (γ1-γ0)/frac)

        σ0 = max(0, σ - (σ1-σ0)/frac)
        σ1 = min(1, σ + (σ1-σ0)/frac)

        last = min_val

        println("$β, $γ, $σ: $min_val")
    end
end
