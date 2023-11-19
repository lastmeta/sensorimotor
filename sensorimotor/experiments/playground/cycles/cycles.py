'''
simple idea.

1. predict the future of the data. present -> future nn.
2. decompose the data into it's cycles, using dft3.
3. those cycles will be changing over time so add observation to histories (angle & amplitude).
4. predict angle, and amplitude in the same way as the nn.
5. repeat 2-4 with each cycle dataset: 
    a. decompose those changing cycles into their cycles.
    b. add to their histories.
    c. predict the future of those cycles.
6. make a model that takes the predictions of the cycles on all levels and predict the future of the data.
'''
