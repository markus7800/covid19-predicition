
using Statistics
using Distributions
using ProgressMeter

∇f_x(x,k,L,x0) = (L*k*exp(-k*(x-x0))) / ((1 + exp(-k*(x-x0)))^2)
# ∇f_L(x,k,L,x0) =  1 / (1 + exp(-k*(x-x0)))
# ∇f_k(x,k,L,x0) = (L*(x-x0)*exp(-k*(x-x0))) / ((1 + exp(-k*(x-x0)))^2)
# ∇f_x0(x,k,L,x0) =  (-L*k*exp(-k*(x-x0))) / ((1 + exp(-k*(x-x0)))^2)

∇f_xL(x,k,L,x0) = (k*exp(-k*(x-x0))) / ((1 + exp(-k*(x-x0)))^2)
function ∇f_xk(x,k,L,x0)
    v = exp(-k*(x-x0))
    return (2*L*k*(x-x0) * v^2 / (1+v)^3) + ( L*v / (1+v)^2) - (L*k*(x-x0)*v / (1+v)^2)
end
function ∇f_xx0(x,k,L,x0)
    v = exp(-k*(x-x0))
    return -2*L*k^2*v^2 / (1+v)^3 + (L*k^2*v / (1+v)^2)
end

dloss(x,y,k,L,x0) =  sum((∇f_x.(x,k,L,x0) .- y).^2)
∇dloss_L(x,y,k,L,x0) = sum(2*(∇f_x.(x,k,L,x0) .- y) .* ∇f_xL.(x,k,L,x0))
∇dloss_k(x,y,k,L,x0) = sum(2*(∇f_x.(x,k,L,x0) .- y) .* ∇f_xk.(x,k,L,x0))
∇dloss_x0(x,y,k,L,x0) = sum(2*(∇f_x.(x,k,L,x0) .- y) .* ∇f_xx0.(x,k,L,x0))

function fit_logistic_derivative(x,y,init_L,init_k,init_x0, L_prec=10, frac=5)
    y = y_new = diff(vcat([0],y))

    L0 = 0.5*init_L
    L1 = 2*init_L

    k0 = 0.5 * init_k
    k1 = 2.0 * init_k

    x00 = 0.5 * init_x0
    x01 = 2.0 * init_x0

    # linesearch

    function search_min(L0,L1,k0,k1,x00,x01,n)
        # println("Search in ($L0, $L1)x($k0, $k1)x($x00, $x01)")
        ΔL = (L1 - L0) / n
        Δk = (k1 - k0) / n
        Δx0 = (x01 - x00) / n

        arg_min = (0, 0, 0)
        min_val = Inf
        for L in L0:ΔL:L1, k in k0:Δk:k1, x0 in x00:Δx0:x01
            v = dloss(x, y, k, L, x0)
            if  v < min_val
                min_val = v
                arg_min = (L, k, x0)
            end
        end
        return arg_min, min_val
    end

    (L,k,x0) = (init_L,init_k,init_x0)


    N = ceil((log(L1 - L0) - log(L_prec)) / log(frac))
    n = 0
    while n <= N
        n += 1

        (L,k,x0), min_val = search_min(L0,L1,k0,k1,x00,x01,100)

        L0 = L - (L1-L0)/frac
        L1 = L + (L1-L0)/frac

        k0 = k - (k1-k0)/frac
        k1 = k + (k1-k0)/frac

        x00 = x0 - (x01-x00)/frac
        x01 = x0 + (x01-x00)/frac

        last = min_val

        # println("$L, $k, $x0: $min_val")
    end

    return (L,k,x0)
end
