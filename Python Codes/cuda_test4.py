import numba
import random
import time
from numba import vectorize
start_time = time.time()


#@numba.jit #same as immediate below. Very fast around 13 or 14 s
#@vectorize(['double(int32)'], target='cpu') #cuda cannnot, only cpu. Need to check function.
def monte_carlo_pi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x**2 + y**2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples

print(monte_carlo_pi(10000000000))
e = int(time.time() - start_time)
print('{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))