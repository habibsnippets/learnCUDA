Contains my learnings about CUDA Programming.
At the end I will try to optimize the edge detection algorithm so that it runs faster and I am able to reduce the time for which the program executes.
Currently it ~ 400 ms


# Introduction
* Flop
    Floating point operation

* LINPACK Benchmark - [LINPACK](https://www.top500.org/project/linpack/)

*  Measuring Performance
    Speedup S(p) = seq. exec time / parallel exec time
    Efficiency E(p) = speedup / no. of processors 
                    = S(p)/p * 100%
    

* Amdahl's and Gustafson's Law:
    Amdhal's Law : max speedup of a program is fundamentally limited by the fraction of program that cannot be parallelized regardless of how many processors are used.

    S : speedup
    N : number of processors
    1 - P : fraction of program that is always serial

    $S = \frac{1}{(1 - P) + \frac{P}{N} }$

* Gustafson's Law:
    s: fraction of serail operations
    p : processors
    $S = p + (1-p)*s$

    > Different from Amdahl's view as it consideres that the problem size can scale with the num of processors. Serial operations remain fixed in time but the parallel portion scales with more processors.

    