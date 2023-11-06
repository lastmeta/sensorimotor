'''
I had this idea about maybe how you could build a feed-forward only network,
a network which receives feed forward input from it's boundary (from both sides)
and efficiently learns the mapping between without backprop. (or with one layer
'backprop' only). Here are my notes on how that might be acheived.

take a feed forward nn, number of neurons in middle are same as beginning
(because they at least have to be number of pairs +1), number of layers is
the number of input neurons.

activation curve is a normalized s-curve. you could probably use anything, but
why not base everything around a normal distribution. either that or optimize 
with straight up relu, but the important thing is that it goes from y=-1 to y=1
with y=0 x=0 at center. that way negative numbers serve as inhibitory values.

now, what about 0? what you could do there is say, modify the activation 
function to be modified by a parameter around 0 so that it can be -1 up to 1.
however. that seems complex, so lets use a modification on the activation after
the fact: normal curve streched small in x or y dimensions. or just a 2 straight
lines if we're going the relu route. this value would be added to the
activation.

So it's a lot like bias, but on a weight basis, not a neuron basis. Perhaps that
could merely be replaced by a bias on the neuron but I don't see how.

Now, when the neurons get activated you do as we've discussed in the FF; you 
take the error right way for every layer and apply it back to the layer/neurons
that were input to it. However, this means we have to make the smallest 
modification possible, (I think), in order to make it have created the right
prediction.

I don't think this does what I've said is probably ideal - automatically 
maintain it's status as a safe model that wont violate any previous predictions,
but I think it'll approximate that somewhat. idk. in order to do that you'd have
to have smarter, memory driven training that said, here are the combinations of
changes I could make, and of those, I can't make these changes because those
would violate my model space. but with everything changing all the time, how are
you going to deduce that let alone track it? I wish there were a way that you
could train to automatically stay in the zone but I don't know how.

I think nature stays in that zone as much as it can by blowing up the spaces
to be so massive that whatever change they make is likely still safe. I don't
know if there's a way to guarantee it automatically, generally speaking.

ok so that's my idea so far. basically it means we add inhibition, and then we
can train incrementally. that's the theory.

one last thing. this probably isn't ideal for long range connections. To handle
long range connections it seems the brain puts cells in a predictive state (by
the firing of long range connections) which then allows that cell to fire first,
optimizing that only the most informed cells fire, not all of them to varying
degrees as we are doing. so I do believe long range connections are critical,
but I don't know if this is sufficient to incorperate, generate or use them
correctly.
'''
