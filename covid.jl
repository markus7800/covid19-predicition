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


# April
# Dates =       [01.4,02.4,03.4,04.4]
# push!(Infected_base, [8900,9060,9193,8990]...)
# push!(Recovered_base,[1436,1749,2022,2507]...)
# push!(Dead_base,     [ 146, 158, 168, 168]...)

l = length(Infected_base)

p_base, pred_I_base, pred_R_base = SINIR_prediction(Date(2020,2,25),Infected_base[1:33],Recovered_base[1:33],Dead_base[1:33],save=false)
display(p_base)

# p1, = daily_prediction(Infected_base, Recovered_base, Dead_base)
# display(p1)

# COMPARE

# Dates =   [25.2.26.2,27.2,28.2,29.2]
Infected =  [   2,   2,   3,   6,  10]
Recovered = [   0,   0,   0,   0,   0]
Dead =      [   0,   0,   0,   0,   0]

# March
# Dates =       [01.3,02.3,03.3,04.3,05.3,06.3,07.3,08.3,09.3,10.3,11.3,12.3,13.3,14.3,15.3,16.3,17.3,18.3,19.3,20.3,21.3,22.3,23.3,24.3,25.3.26.3,27.3,28.3,29.3,30.3,31.3]
push!(Infected, [  14,  18,  21,  27,  39,  53,  77, 100, 138, 178, 242, 356, 497, 648, 853,1007,1320,1633,1998,2373,2793,3212,3875,4833,5490,6237,7158,7552,7971,8633,8751]...)
push!(Recovered,[   0,   0,   0,   2,   2,   2,   2,   2,   2,   4,   4,   4,   6,   6,   6,   6,   9,   9,   9,   9,  14,  16,  28,  68,  68, 112, 225, 410, 479, 636,1095]...)
push!(Dead,     [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   1,   3,   3,   4,   6,   6,   7,  16,  21,  25,  30,  49,  58,  68,  86, 108, 128]...)

# April
# Dates =       [01.4,02.4,03.4,04.4,05.4,06.4,07.4,08.4,09.4,10.4,11.4,12.4,13.4,14.4,15.4,16.4,17.4, 18.4 19.4, 20.4, 21.4, 22.4, 23.4, 24.4, 25.4, 26.4, 27.4, 28.4, 29.4, 30.4]
push!(Infected, [8900,9060,9193,8990,8705,8523,8230,8067,7603,7070,6835,6557,6288,6142,5830,5055,4418, 3980, 3743, 3654, 3348, 3051, 2747, 2636, 2478, 2351, 2328, 2137, 1993, 1933]...)
push!(Recovered,[1436,1749,2022,2507,2998,3463,4046,4512,5240,6064,6604,6987,7343,7633,8098,8986,9704,10214,10501,10631,10971,11328,11694,11872,12103,12282,12362,12580,12779,12907]...)
push!(Dead,     [ 146, 158, 168, 168, 204, 220, 243, 273, 295, 319, 337, 350, 368, 384, 393, 419, 431,  443,  452,  470,  491,  510,  522,  530,  536,  542,  549,  569,  580,  584]...)

scatter!(l+1:length(Infected), Infected[l+1:length(Infected)], label="new Infected", mc=:red)
scatter!(l+1:length(Recovered), Recovered[l+1:length(Recovered)], label="new Recovered", mc=:red)
scatter!(l+1:length(Dead), Dead[l+1:length(Dead)], label="new Dead", mc=:red)
p = scatter!(l+1:length(Dead), Infected[l+1:length(Infected)] .+ Recovered[l+1:length(Recovered)] .+ Dead[l+1:length(Dead)], label="new Dead", mc=:red)

display(p)

p_now, pred_I, pred_R = SIR_prediction(Date(2020,2,25),Infected,Recovered,Dead,save=true)
display(p_now)

p_now, pred_I, pred_R, p1 = SINIR_prediction(Date(2020,2,25),Infected,Recovered,Dead,save=true,months=3)
display(p1)

p_tot, = Logisitic_prediction(Infected, Recovered, Dead, save=true)
display(p_tot)

SINIR_animation(Date(2020,2,25),Infected,Recovered,Dead,40)


p, = SINIR_prediction(Date(2020,2,25),Infected,Recovered,Dead,save=false,up_to=length(Infected)-20,months=3)

display(p)
