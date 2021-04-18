# Paper Review: Efficient Mining of Multiple Fuzzy Frequent Itemsets

## Introduction

Association Rules:
- Sequential Pattern (SP)
- Classification
- Clustering
- High-utility itemset (HUI)

Cons (Appriori):
- Time-consuming computation
- Numerous candidate itemsets

*Note:* Han et al. designed FP-tree structure to not generate candidates. FP-growth algorithm can mine FIs effciently.

**BUT!** Crisp sets are not good to handle quantitative databases which usually provide more imformation for decision making than that of binary databases.

The reviewed paper proposes:
- MFFI-Miner algorithm
  - No candidate generation
  - Based on fuzzy-list structure
  - Reduces complexity of generate-and-test approach in a level-wise manner
  - With 2 pruning strategies to reduce the search space of tree based on the above-mentioned structure
  - Computation can be greatly reduced
  - Better performance than Apriori-based and pattern-growth algorithms

### Fuzzy sets

Fuzzy set = Set + Membership

Membership function indicates the degree of inclusion of a member in a set.
Membership in range [0,1]

## Preliminaries and Problem Definition

- D = { T1, T2, ..., Tn } - quantitative database
- I = { i1, i2, ..., im } - finite set of m distinct items
- Tq is included in D
  - subset of I
  - format: items and purchase qunatities v[iq]
  - represented by a unique identifier TID
- X is an itemset of k distinct items { i1, i2, ..., ik } and called k-itemset
- minsup - minimum support threshold
- Memb - a set of user-specified membership functions

supp(X) = { X in R[il] | ...

### Problem Statement

Goals:
- Speed up the mining process
- Discover the complete set of MFFIs

MFFI <- { X | supp(X) is greater than or equal to minsup*|D| }

## Proposed MFFI-Miner Algorithm

Phases:
1. Transformation
2. Fuzzy-list construction
3. Search space of enumeration tree

### Transformation Phase

- Transformation of quantitave value of each linguistic variable(item) into several fuzzy linguistic terms(fuzzy itemsets)
- Introducing membership degrees(fuzzy values) of the fuzzy itemsets
- Support of a fuzzy itemset is the summation of all fuzzy values of the same fuzzy itemset
- If support is no less than the minsup count then the fuzzy itemset is FFI (kept transformed)
- Sorting remaining fuzzy itemsets with their fuzzy values in support-ascending order
  - To perform intersection operation

### Fuzzy-list construction Phase

Definitions:
- Tq/R[il] is to indicate the set of fuzzy itemsets after R[il] in Tq
- The fuzzy value of R[il] is evaluated as if(R[il], Tq) ("if" stands for internal fuzzy [value])
- The resting fuzzy value except R[il] in Tq is evaluated as rf(R[il], Tq) = max(if(z,Tq) | z in (T1/R[il]) ("rf" stands for resting fuzzy [value])

Algorithm(Construct):
- Input: Fuzzy lists of Px and Py
- Output: Fuzzy list of x and y
- if Px and Py belong to the same item then return NULL
- P[xy] is initialized as NULL
- for each Ex in Px do
  - if there is a Ey such that Ex.tid == Ey.tid then
    - E[xy].tid <- Ex.tid
    - E[xy].if <- min(Ex.if, Ey.if)
    - E[xy].rf <- Ey.rf
    - E[xy] <- <E[xy].tid, E[xy].if, E[xy].rf>
    - P[xy].append(E[xy])

Note:
- SUM.R[il].if = sum of all if(R[il], Tq] over the transformed database D'
- SUM.R[il].ef = sum of all rf(R[il], Tq] over the transformed database D'

### Search space of enumeration tree

Algorithm(MFFI-Miner):
- Input: fuzzy-list of 1-itemsets(FLs), minsup
- Output: MFFIs
- for each fuzzy-list X in FLs do
  - if SUM.X.if >= minsup then
    - MFFIs <- X + MFFIs
  - if SUM.X.rf >= minsup and SUM.X.if >= minsup then
    - exFLs <- NULL
    - for each fuzzy-list Y after X in FLs do
      - exFL <- exFLs + Construct(X, Y)
  - MFFI-Miner(exFLs, minsup)

## Expreimental Results

Metrics;
- Runtime
- Memory usage
- Node analysis

Used algorithms:
- GDF (state-of-the-art)
- UBMFFP
- MFFI-Miner[PI,PR]

-Datasets:
- Four real-life chess
- Mushroom
- Connect
- Retail
- T10I10D100k & T10I4D100k (synthetic)

Process:
- Quantities of items were randomly assigned in the range of [1,5]
- The quantitative datasets were transformed into several fuzzy linguistic terms (based on predefined membership functions)
- The algorithm was terminated if the execution time exceeded 10000s

### Runtime

- 3 fuzzy linguistic terms
- GDF and UBMFFP-tree algorithms did not have any results when the support threshold was set lower

<images>

### Memory usage

- Designed fuzzy-list structre could be used to compress the dataset as a dense structure
- Designed pruning strategies could be used to reduce the generation of candidates during the mining process
- Very dense dataset required more memory usage to retain more fuzzy frequent itemsets (Fig. 5a)

<images>

### Node analysis

- GDF needed to generate the number of candidate itemsets in a level-wise manner
- UBMFFP-tree needed to generate even more candidates while the upper-bound strategy was used
- Designed algorithm(MFFI-Miner) did not need to generate candidates, but rather had to traverse the nodes in the enumeration tree

<images>

## Conclusion

- Two pruning strategies were designed to reduce the search space
- The proposed MFFI-Miner algorithm outperformed the GDF & UBMFFP-tree algorithms in terms of
  - runtime
  - memory usage
  - number of determining candidates in both real-world and synthetic datasets
