include("tools.jl")

# BASELINE 30.3.

# February
# Dates =   [25.2.26.2,27.2,28.2,29.2]]
Infected_base =  [   2,   2,   3,   6,  10]
Recovered_base = [   0,   0,   0,   0,   0]
Dead_base =      [   0,   0,   0,   0,   0]

# March
# Dates =            [01.3,02.3,03.3,04.3,05.3,06.3,07.3,08.3,09.3,10.3,11.3,12.3,13.3,14.3,15.3,16.3,17.3,18.3,19.3,20.3,21.3,22.3,23.3,24.3,25.3.26.3,27.3,28.3,29.3,30.3,31.3]
push!(Infected_base, [  14,  18,  21,  27,  39,  53,  77, 100, 138, 178, 242, 356, 497, 648, 853,1007,1320,1633,1998,2373,2793,3212,3875,4833,5490,6237,7158,7552,7971,8633,8751]...)
push!(Recovered_base,[   0,   0,   0,   2,   2,   2,   2,   2,   2,   4,   4,   4,   6,   6,   6,   6,   9,   9,   9,   9,  14,  16,  28,  68,  68, 112, 225, 410, 479, 636,1095]...)
push!(Dead_base,     [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   1,   3,   3,   4,   6,   6,   7,  16,  21,  25,  30,  49,  58,  68,  86, 108, 128]...)



l = length(Infected_base)

p_base, pred_I_base, pred_R_base = SIR_prediction(Date(2020,2,25),Infected_base,Recovered_base,Dead_base,save=false)
display(p_base)

# p1, = daily_prediction(Infected_base, Recovered_base, Dead_base)
# display(p1)

# COMPARE

# Dates =   [25.2.26.2,27.2,28.2,29.2]]
Infected =  [   2,   2,   3,   6,  10]
Recovered = [   0,   0,   0,   0,   0]
Dead =      [   0,   0,   0,   0,   0]

# March
# Dates =       [01.3,02.3,03.3,04.3,05.3,06.3,07.3,08.3,09.3,10.3,11.3,12.3,13.3,14.3,15.3,16.3,17.3,18.3,19.3,20.3,21.3,22.3,23.3,24.3,25.3.26.3,27.3,28.3,29.3,30.3,31.3]
push!(Infected, [  14,  18,  21,  27,  39,  53,  77, 100, 138, 178, 242, 356, 497, 648, 853,1007,1320,1633,1998,2373,2793,3212,3875,4833,5490,6237,7158,7552,7971,8633,8751]...)
push!(Recovered,[   0,   0,   0,   2,   2,   2,   2,   2,   2,   4,   4,   4,   6,   6,   6,   6,   9,   9,   9,   9,  14,  16,  28,  68,  68, 112, 225, 410, 479, 636,1095]...)
push!(Dead,     [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   1,   3,   3,   4,   6,   6,   7,  16,  21,  25,  30,  49,  58,  68,  86, 108, 128]...)

# April
# Dates =       [01.4,02.4,03.4,04.4]
push!(Infected, [8900,9060,9193,8990]...)
push!(Recovered,[1436,1749,2022,2507]...)
push!(Dead,     [ 146, 158, 168, 168]...)

scatter!(l+1:length(Infected), Infected[l+1:length(Infected)], label="new Infected", mc=:red)
scatter!(l+1:length(Recovered), Recovered[l+1:length(Recovered)], label="new Recovered", mc=:red)
p = scatter!(l+1:length(Dead), Dead[l+1:length(Dead)], label="new Dead", mc=:red)

display(p)

p_now, pred_I, pred_R = SIR_prediction(Date(2020,2,25),Infected,Recovered,Dead,save=true)
display(p_now)

p_tot, = daily_prediction(Infected, Recovered, Dead, save=true)
display(p_tot)

scatter(Infected .+ Recovered .+ Dead, legend=false)

y = Infected .+ Recovered .+ Dead
x = collect(0:length(y)-1)

plot(t -> ∇loss_L(t, x, y), 12000, 15000)
