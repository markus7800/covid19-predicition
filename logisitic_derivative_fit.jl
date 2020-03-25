
using Statistics
using Distributions
using NLsolve

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

∇dloss_L(x,y,k,L,x0) = sum(2*(∇f_x.(x,k,L,x0) .- y) .* ∇f_xL.(x,k,L,x0))
∇dloss_k(x,y,k,L,x0) = sum(2*(∇f_x.(x,k,L,x0) .- y) .* ∇f_xk.(x,k,L,x0))
∇dloss_x0(x,y,k,L,x0) = sum(2*(∇f_x.(x,k,L,x0) .- y) .* ∇f_xx0.(x,k,L,x0))

function fit_d(x,y,init_L,init_k,init_x0)
    function f!(F, t)
        F[1] = ∇dloss_L(x,y,t[1],t[2],t[3])
        F[2] = ∇dloss_k(x,y,t[1],t[2],t[3])
        F[3] = ∇dloss_x0(x,y,t[1],t[2],t[3])
    end

    init_t = [init_k,init_L,init_x0]

    println(nlsolve(f!, init_t))
end
