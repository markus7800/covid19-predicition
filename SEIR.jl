
function estimate_SIR(y1,y2)

    N = 8.822 * 1e6

    I = y1
    R = y2

    dI = diff(vcat([0], y1))
    dR = diff(vcat([0], y2))

    γ = mean(dR ./ I)

    S = @. N - I - R
    dS = diff(vcat([N], S))

    β = -mean(@. (dS * N) / (S * I))

    return β, γ
end

β, γ = estimate_SIR(y1,y2)
a, b = SEIR(35, 0.4, γ, 0.5, 2)
plot(a)
scatter!(y1)
plot!(b)
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
