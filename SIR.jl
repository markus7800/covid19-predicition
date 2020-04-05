using ProgressMeter
using Optim

#==============================================================================#
#=
    SIR MODEL

three functions:

    S ... Susceptible
    I ... Infected
    R ... Removed (Dead and Recovered)

exact solutions to following system of differential equations:

    dS = -b * S * I / (S + I)
    dI = b * S * I / (S + I) - c * I
    dR = c * I

under the constant population constraint:

    S + I + R = s0 + i0; s0 = S(t0), i0 = I(t0)

# https://arxiv.org/pdf/1812.09759.pdf
=#

# N = s0 +i0
function exact_S(t, t0, b, c, s0, i0)
    κ = i0 / s0
    γ = b / (b-c)

    return s0*(1+κ)^γ / (1+κ*exp((b-c)*(t-t0)))^γ
end

function exact_I(t, t0, b, c, s0, i0)
    κ = i0 / s0
    γ = b / (b-c)

    return i0*(1+κ)^γ * exp((b-c)*(t-t0)) / (1+κ*exp((b-c)*(t-t0)))^γ
end

function exact_R(t, t0, b, c, s0, i0)
    N = s0 + i0
    # println("$b, $c, $(b-c) $(s0 + i0)")
    return N - (s0 + i0)^(b/(b-c)) / (s0 + i0*exp((b-c)*(t-t0)))^(c/(b-c))
end

#==============================================================================#
#=
    SINIR MODEL

five functions:

    S  ... Susceptible
    I  ... Infectious
    NI ... Not Infectious
    Q  ... Quarantined (Not Infectious and Not Removed)
    R  ... Removed (Recovered or Dead)

exact solutions to following system of (differential) equations:

    dS = -b * S * I / (S + I)
    dI = b * S * I / (S + I) - c * I
    dNI = c * I
    R(t) = NI(t - t1), where t1 is a recovery/succumbency time
    Q = NI - R

under the constant population constraint:

    S(t) + I(t) + NI(t) = s0 + i0; s0 = S(t0), i0 = I(t0)
    S(t) + I(t) + R(t + t1) = s0 + i0; delay in recovery/succumbency

=#

function exact_NI(t, t0, b, c, s0, i0)
    N = s0 + i0
    # println("$b, $c, $(b-c) $(s0 + i0)")
    return N - (s0 + i0)^(b/(b-c)) / (s0 + i0*exp((b-c)*(t-t0)))^(c/(b-c))
end

function exact_R(t, t0, t1, b, c, s0, i0)
    return exact_NI(t-t1,t0,b,c,s0,i0)
end

function exact_Q(t, t0, t1, b, c, s0, i0)
    return exact_NI(t,t0,b,c,s0,i0) - exact_R(t,t0,t1,b,c,s0,i0)
end

#==============================================================================#

function peak_I(t0, b, c, s0, i0, upper=100)
    F((t,)) = -exact_I(t, t0, b, c, s0, i0)

    lower = [t0]
    upper = [upper]

    optimize(F, lower, upper, [Float64(t0)], SAMIN(),Optim.Options(iterations=10^6))
end

# using Plots
#
# plot(t -> exact_S(t, 0, 0.5, 0.1, 100, 1),0,50)
# plot!(t -> exact_I(t, 0, 0.5, 0.1, 100, 1))
# plot!(t -> exact_R(t, 0, 0.5, 0.1, 100, 1))
# plot!(t -> exact_S(t, 0, 0.5, 0.1, 100, 1)+exact_I(t, 0, 0.5, 0.1, 100, 1)+exact_R(t, 0, 0.5, 0.1, 100, 1))


exact_I_loss(x,I,t0,b,c,s0,i0) = 1/2 * sum((exact_I.(x,t0,b,c,s0,i0) .- I).^2)
exact_R_loss(x,R,t0,b,c,s0,i0) = 1/2 * sum((exact_R.(x,t0,b,c,s0,i0) .- R).^2)
exact_SIR_loss(x,I,R,t0,b,c,s0,i0) = exact_I_loss(x,I,t0,b,c,s0,i0) + exact_R_loss(x,R,t0,b,c,s0,i0)
exact_SINIR_loss(x,I,R,t0,t1,b,c,s0,i0) = exact_I_loss(x,I,t0,b,c,s0,i0) + exact_R_loss(x,R,t0+t1,b,c,s0,i0)

γ(b,c) = b - c
R0(b,c) = b / c
b(γ,R0) = γ * R0 / (1 - R0)
c(γ,R0) = γ / (R0 - 1)

function fit_exact_SIR(x,I,R)
    i0 = I[1]
    # γ = b-c
    # R0 = b/c
    # b = γ * R0/(1-R0)
    # c = γ /(R0 - 1)
    F((s0,t0,γ,R0)) = exact_SIR_loss(x,I,R,t0,γ*R0/(R0-1),γ/(R0-1),s0,i0)

    lower = Float64[maximum(I .+ R), -10,    0,  1]
    upper = Float64[9*1e6,       10,    1,  100]

    optimize(F, lower, upper, [1e6, 0, 0.1, 2], SAMIN(),Optim.Options(iterations=10^6))
end

function fit_exact_SINIR(x,I,R)
    i0 = I[1]
    # γ = b-c
    # R0 = b/c
    # b = γ * R0/(1-R0)
    # c = γ /(R0 - 1)
    F((s0,t0,t1,γ,R0)) = exact_SINIR_loss(x,I,R,t0,t1,γ*R0/(R0-1),γ/(R0-1),s0,i0)

    lower = Float64[maximum(I .+ R),   -10,    0,   0,    1]
    upper = Float64[9*1e6,              10,    20,  1,  100]

    optimize(F, lower, upper, [1e6, 0, 0, 0.1, 2], SAMIN(),Optim.Options(iterations=10^6))
end
