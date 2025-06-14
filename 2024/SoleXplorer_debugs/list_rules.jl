using Printf
format_float(x) = replace(x, r"(\d+\.\d+)" => s -> @sprintf("%.2f", parse(Float64, s)))

dt = prop_sole_dt
min_lift=1.0
min_ninstances=0
min_coverage=0.07
min_ncovered=1
normalize=true

rules = listrules(
            dt,
            min_lift=min_lift,
            min_ninstances=min_ninstances,
            min_coverage=min_coverage,
            min_ncovered=min_ncovered,
            normalize=normalize,
)


# for i in irules
    
#     test=printrules(i)
#     # println(test)
#     # antecedent, consequent = a_c[1], a_c[3]
#     # antecedent = reduce((s, r) -> replace(s, r => ""), r_p_ant, init=antecedent)
#     # antecedent = format_float(antecedent)
#     # println(antecedent, consequent, readmetrics(i[1])...)
# end

a,c=rules2string(rules[1])
format_float(a)