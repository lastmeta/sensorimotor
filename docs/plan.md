optional paths: 
1. autoencoders - searching for the smallest unit of intelligence
2. abstract algebra map reduction - compress the map to a set of rules that always remain true
    a. define best representation of memory (map of environment (dags?))
    b. detect symetrical patterns
    c. represent symetrical patterns (concepts)
    d. compress map through recursion of this process


State, action, next state
0,       +1,     1
1,       +2,     3

Element
[-3,0,+3,...]


group elements 0: 
        0   1   -1  10  -9
---------------------------
0   |   0   1   -1  10  -9
1   |   1   2   0   11  -8
-1  |   -1  0   -2  9   -10
10  |   10  11  9   20  1
-9  |   -9  -8  -10 1   -18

group elements 1: 
%3      0   1   -1  10  -9  2   ...
---------------------------
0   |   0   1   -1  10  -9
1   |   1   2   0   11  -8
-1  |   -1  0   -2  9   -10
10  |   10  11  9   20  1
-9  |   -9  -8  -10 1   -18
2   |
...


step 1: 
create hierarchical group elemets (label all combinations of actions).
(keep a running list of every combination of actions I've performed)

    Actions
    0 (0)
    1 (1)
    2 (2)
    3 (3)
    4 (4)
    5 (0,0)
    6 (0,1)
    7 (0,2)
    ...


    id  State, action, next state
        0,       1,     1
        1,       3,     10
        v,      4,     1
        1,       2,     -1
        -1,      0,     -1
        -1,      2,     -2

    higher abstraction (every 2 actions)
    -1,      7,     -2



step 2: 
detect transition patterns (cycles, for instance)

step 3: 
marry elements to patterns

step 4:
higher order abstractions (how do you move from cycle to cycle)
