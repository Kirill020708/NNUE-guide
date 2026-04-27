There's also a very good [article](https://cosmo.tardis.ac/files/2024-06-01-nnue.html) about different optimizations.

## Lazy updates
Because we use a transposition table to store positions' evaluations, we don't need to evaluate some positions. That means that we also don't have to know the accumulators' values. So we can skip some accumulators' updates.

When we want to update our accumulators (when making a move), we can just push the update information (e.g. changed squares) into a stack (or pop when unmaking a move) and mark the current state of the accumulators as "dirty" ("dirty" means that the accumulators in this stack entry are unknown; "clean" means the opposite). When we want to evaluate our position, we need to calculate the accumulators. To do it, we find the last entry with "clean" accumulators and then go through the stack and apply all of the updates we stored (while also saving all the intermediate states because we might need them).  If you have some sort of net refreshing (e.g. mirroring/king buckets), you can also save this info instead of actually recalculating the net, and calculate it only when you actually need it (tip: you can mark the state right before the current one as "clean", because we don't need its accumulators, because we do a full net refresh on current entry anyway).
###### Pseudocode:
```python
def make_move():
	...
	stack.push(updated_squares)
	
def unmake_move():
	...
	stack.pop()

...

accumStack[256][2N]

def update(i, updated_squares):
	accumStack[i + 1] = add(accumStack[i], updated_squares)

def clean_accumulators():
	for i in lastCleanAccNumber..currentAccNumber:
		update(i, stack[i])
	
def evaluate():
	clean_accumulators()
	...
```

## Sparse matrix multiplication
There is another good [article](https://cosmo.tardis.ac/files/2024-08-17-multilayer.html#sparse-matrix-multiplication) on the same topic.

This technique only works for a multilayer network.
Because we're using SCReLU for the accumulators' activation, all negative values clamp to $0$, so after the activation a lot of values are $0$. These zeros don't affect the values of the second hidden layer, so we can just skip them when computing it, which gives us a speedup. However, if you just write
```c++
if (act[i] == 0)
	continue;
```
you may even slow down the inference because it's quite computationally expensive to compute a condition. Instead of this, we do a clever algorithm, which lets us compute all non-zero activation indexes.
##### Algorithm
When computing the matmul, we process 4 activations of the first hidden layer at once (let's call these 4 activations a "block"). We want to compute an array `int16_t nonzeroActivations[]`, which stores the **indices** of all nonzero blocks of 4 activations.

First, we write a `int8_t nonzeroMask(__m256i x)` function, which gives us a mask of all nonzero blocks in m256 vector:
```c++
inline int8_t nonzeroMask(__m256i x) {
        return _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(x, _mm256_setzero_si256())));
    }
```
which is basically comparing each 32-bit block with $0$, and using `_mm256_movemask_ps` to get the mask.

We have a byte, and we want to quickly store all non-zero bit numbers in `nonzeroActivations`. Turns out, we can just precompute an array `uint16_t byteIndex[256][8]`, which stores the indices (e.g. for 53 = 0b110101 `byteIndex[53] = {0, 1, 3, 5}`). Then the whole algorithm is:
```c++
uint16_t nonzeroIndexes[K];
int nonzeroCount;
__m128i baseIdx;

void update(__m256i x) {
	int8_t mask = nonzeroMask(x);
	__m128i indexes = load(byteIndex[mask]);
	store(nonzeroIndexes[nonzeroCount], add(indexes, baseIdx));
	nonzeroCount += popcount(mask);
	baseIdx = add(baseIdx, m128_set(8));
}
```
We need the `baseIdx` vector because we compute 8 blocks at once. So when we call update() the second time and we get indexes {0, 2, 3}, actually their indexes are {8, 10, 11}:

![](images/sparse.png)

Then when evaluating instead of doing
```c++
for (int i = 0; i < K; i++) {
	compute_matmul(i);
}
```
we do
```c++
nonzeroCount = 0;
baseIdx = m128_set(0);

for (int i = 0; i < K; i++) {
	update(load(act[i]));
}

for (int i = 0; i < nonzeroCount; i++) {
	compute_matmul(nonzeroIndexes[i]);
}
```