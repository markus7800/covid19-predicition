using ProgressMeter
using Optim

# N = s0 +i0
function exact_I(t, t0, b, c, s0, i0)
    # https://arxiv.org/pdf/1812.09759.pdf
    κ = i0 / s0
    c = c
    γ = b / (b-c)

    return i0*(1+κ)^γ * exp((b-c)*(t-t0)) / (1+κ*exp((b-c)*(t-t0)))^γ
end

function exact_S(t, t0, b, c, s0, i0)
    κ = i0 / s0
    c = c
    γ = b / (b-c)

    return s0*(1+κ)^γ / (1+κ*exp((b-c)*(t-t0)))^γ
end

function exact_R(t, t0, b, c, s0, i0)
    N = s0 + i0
    # println("$b, $c, $(b-c) $(s0 + i0)")
    return N - (s0 + i0)^(b/(b-c)) / (s0 + i0*exp((b-c)*(t-t0)))^(c/(b-c))
end

function peak_I(t0, b, c, s0, i0, upper=100)
    F((t,)) = -exact_I(t, t0, b, c, s0, i0)

    lower = [t0]
    upper = [upper]

    optimize(F, lower, upper, [Float64(t0)], SAMIN(),Optim.Options(iterations=10^6))
end

using Plots

plot(t -> exact_S(t, 0, 0.5, 0.1, 100, 1),0,50)
plot!(t -> exact_I(t, 0, 0.5, 0.1, 100, 1))
plot!(t -> exact_R(t, 0, 0.5, 0.1, 100, 1))
plot!(t -> exact_S(t, 0, 0.5, 0.1, 100, 1)+exact_I(t, 0, 0.5, 0.1, 100, 1)+exact_R(t, 0, 0.5, 0.1, 100, 1))


exact_I_loss(x,I,t0,b,c,s0,i0) = 1/2 * sum((exact_I.(x,t0,b,c,s0,i0) .- I).^2)
exact_R_loss(x,R,t0,b,c,s0,i0) = 1/2 * sum((exact_R.(x,t0,b,c,s0,i0) .- R).^2)
exact_IR_loss(x,I,R,t0,b,c,s0,i0) = exact_I_loss(x,I,t0,b,c,s0,i0) + exact_R_loss(x,R,t0,b,c,s0,i0)


function fit_exact_IR_1(x,I,R)
    i0 = I[1]
    # γ = b-c
    # R0 = b/c
    # b = γ * R0/(1-R0)
    # c = γ /(R0 - 1)
    F((s0,t0,γ,R0)) = exact_IR_loss(x,I,R,t0,γ*R0/(R0-1),γ/(R0-1),s0,i0)

    lower = Float64[maximum(I .+ R), -10,    0,  1]
    upper = Float64[9*1e6,       10,    1,  100]

    optimize(F, lower, upper, [1e6, 0, 0.1, 2], SAMIN(),Optim.Options(iterations=10^6))
end

function fit_exact_SIR_3(x,y)
    i0 = y[1]
    F((s0,b,c)) = exact_I_loss(x,y,b,c,s0,i0)

    lower = Float64[maximum(y),0,0]
    upper = Float64[9*1e6, 1, 0.1]

    # optimize(F, [10000, 0.5, 0.1], ParticleSwarm(lower=lower,upper=upper,n_particles=100))

    optimize(F, lower, upper, [50000, 0.5, 0.1], SAMIN(), Optim.Options(iterations=10^6))
end

function fit_exact_SIR_4(x,I,R,R0)
    i0 = y[1]
    # γ = b-c
    # R0 = b/c
    # b = γ * R0/(1-R0)
    # c = γ /(R0 - 1)
    F((s0,t0,γ)) = exact_I_loss(x,y,t0,γ*R0/(R0-1),γ/(R0-1),s0,i0)

    optimize(F, [1e4, 0, 0.1])
end

function fit_exact_SIR_5(x,y)
    i0 = y[1]
    # γ = b-c
    # R0 = b/c
    # b = γ * R0/(1-R0)
    # c = γ /(R0 - 1)
    F((t0,R0,γ)) = exact_I_loss(x,y,t0,γ*R0/(R0-1),γ/(R0-1),1e6,i0)

    optimize(F, [0, 2, 0.1])
end

function fit_exact_SIR_2(x,y)
    i0 = y[1]
    F((s0,b,c)) = exact_I_loss(x,y,b,c,s0,i0)
    function g!(G, (s0,b,c))
        ∇s0, ∇b, ∇c = ∇exact_I_loss(x,y,b,c,s0,i0)
        G[1] = ∇s0
        G[2] = ∇b
        G[3] = ∇c
    end

    optimize(F, g!, [10000, 0.5, 0.2], BFGS())
end

function fit_exact_SIR(x,y)
    i0 = y[1]
    F((s0,b,c)) = exact_I_loss(x,y,b,c,s0,i0)
    lower = [maximum(y),0,0]
    upper = [Inf, 1, Inf]
    inner_optimizer = GradientDescent()
    optimize(F, lower, upper, [10000, 0.5, 0.1], Fminbox(inner_optimizer))
end


# grads

function ∇exact_I_s0(t, b, c, s0, i0)
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

function ∇exact_I_b(t, b, c, s0, i0)
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

function ∇exact_I_c(t, b, c, s0, i0)
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

function ∇exact_SIR_loss(x,y,b,c,s0,i0)
    w = exact_SIR.(x,b,c,s0,i0) .- y
    ∇s0 = sum(w .* ∇exact_I_s0.(x,b,c,s0,i0))
    ∇b = sum(w .* ∇exact_I_b.(x,b,c,s0,i0))
    ∇c = sum(w .* ∇exact_I_c.(x,b,c,s0,i0))

    return ∇s0, ∇b, ∇c
end
