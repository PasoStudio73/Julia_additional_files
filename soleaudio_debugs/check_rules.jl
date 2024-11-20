using SoleModels

rules = listrules(
    irules,
    min_lift = 1.0,
    # min_lift = 2.0,
    min_ninstances = 0,
    # min_coverage = 0.10,
    # min_ncovered = 5,
    normalize = true,
);
map(r->(consequent(r), readmetrics(r)), rules)
p_irules = sort(rules, by=readmetrics)