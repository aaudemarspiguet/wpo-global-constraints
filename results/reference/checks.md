# Algorithm checks

## Greedy vs brute force (Section 4.3)
30 random users, 9 candidate widgets, n = 4, objective = Eq. (9) with xi from the last day.
- min ratio greedy/optimal: 1.0000
- mean ratio: 1.0000
- theoretical floor (1 - 1/e): 0.6321

## n = 1, linear f: primal-dual vs exact LP (Section 4.2 special case)
- LP optimum (fractional, an upper bound): 0.17533 per page
- primal-dual (integral pages): 0.17533 per page, gap -0.00%
- primal-dual miss@1: 0.00%
- LP duals for the constraints (marginals): [0.0212, 0.0038]
- primal-dual xi on constrained categories: [0.0212, 0.0028]

## Algorithm 1 on day 6: iterations to reach eps = 0.02: 34
