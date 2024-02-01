---
layout: single
---

# Parameter Constraints in Auto-Tuning

Traditionally, a tuning parameter `tp` is defined as a tuple consisting of the parameter's name and its range of possible values:

~~~ C
tp = ( name , range )
~~~

ATF extends this traditional definition of a tuning parameters by adding to it a *parameter constraint*:

~~~ C
tp = ( name , range , constraint )
~~~

ATF uses parameter constraints to express arbitrary interdependencies among tuning parameters, e.g., that the value of one tuning parameter has to evenly divide the value of another tuning parameter (such as tile size tuning parameters on different memory layers).

Expressing parameters constraints for the auto-tuner is important, because it allows the auto-tuner to internally use optimized processes for *generating*, *storing*, and *exploring* the search spaces of interdependent tuning parameters, thereby contributing to a significantly more efficient overall auto-tuning process, as discussed in detail [here](https://dl.acm.org/doi/abs/10.1145/3427093) and outlined in the following.

## Generating Constrained Search Spaces

ATF introduces a novel search space generation algorithm for constrained tuning parameters. Our novel algorithm exploits ATF's constraint design to substantially speedup the generation of constrained search spaces, thereby enabling auto-tuning complex parallel applications with large search spaces.

## Storing Constrained Search Spaces

ATF introduces the novel *Chain-of-Trees (CoT)* search space structure for constrained tuning parameters. In contrast to existing approaches, ATF reduces the memory consumption of constrained parameters' spaces, often by multiple orders of magnitude, thereby enabling auto-tuning within the memory limitations of state-of-the-art computer systems.

## Exploring Constrained Search Spaces

ATF introduces a novel exploration strategies for constrained search space, based on its CoT search space representation. Unlike other approaches which flatten the originally multi-dimensional search space representation to one dimension only, ATF retains the multi-dimensional search space structure, thereby contributing to the efficiency of search techniques (e.g., from numerical optimization).