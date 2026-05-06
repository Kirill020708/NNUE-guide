Instead of using only one hidden layer in our net, we use more.
The most common architecture for a multilayer network is $(768 \rightarrow N)\text{x}2 \rightarrow 16 \rightarrow 32 \rightarrow 1$.
The quality of evaluation increases a lot, but we get a big performance hit.
To propagate the first hidden layer further, we now need $16\times$ more time because the number of neurons on the next layer increases $16\times$.
However, there is a clever algorithm which allows us to evaluate the net without a considerable performance hit.
Every hidden layer's activation function is still SCReLU except the last one (size $32$); its activation function is CReLU (Clipped ReLU), which is just $\text{clamp}(x, 0, 1)$.

![](../images/multilayer.png)
Just like in output buckets, number of weights doesn't increase a lot, so we don't need more data to train a multilayer net.

## The algorithm

It's described very well in [this](https://aletheiaaaaa.github.io/posts/2025-07-14-dpbusd-explained/) article.

## Quantization

You only need to quantize the first hidden layer (with the reasons described in Basic NNUE).
You can do floats for later layers, but you can get some floating-point inconsistencies (e.g. different bench on different platforms), so I prefer to quantize the whole net.
The constants are: $Q0=255, Q1=128, Q=64$. The quantization of each parameter is:

| Parameter    | Quantization |
| ------------ | ------------ |
| $w0,b0$      | $Q0$         |
| $w1$         | $Q1$         |
| $b1, w2, w3$ | $Q$          |
| $b2$         | $Q^3$        |
| $b3$         | $Q^4$        |

Just like in the basic NNUE, the accumulators end up fitting into 16 bits (quantized as $Q0$). $Q1$ needs to be $128$ instead of $255$, because in early versions of ARM the SIMD instruction uses signed int8 arithmetic, so you need it to fit into $2^7$.
After activation they get quantized as $Q0^2$.
However, when we compute next layer, we need the values of the accumulators' activations values to fit into 8-bit numbers.
To do it, we divide the activations by the least possible power of $2$: $\frac{Q0^2}{2^9}<2^{7}$; (we do it with `_mm256_mulhi_epi16` SIMD instruction: $\text{act} =\text{mulHigh16}( \text{shiftLeft16(acc, 7)}, acc)$, where `mulHigh16` is multiplication in 16-bit numbers and keeping the high part, which is effectively doing /= $2^{16}$, so it ends up in $\text{act} =\frac{\text{acc}^2 \cdot 2^7}{2^{16}} = \frac{\text{acc}^2}{2^{9}}$).
After this we end up with $\frac{Q0^2 \cdot Q1}{2^9}$ quantization; however, we want $Q$ quantization.
To do that, we must multiply our values by $Q \cdot \frac{2^9}{Q0^2 \cdot Q1} \approx \frac{1}{2^8}$, so we just shift the values right by $8$.
The rest is easy: we propagate the values further (without any fancy algorithms, just plain matmul with SIMD because $16 \cdot 32$ is small compared to $2N \cdot 16$), ending up with $Q^4$ quantization.
Then we multiply the result by $\frac{S}{Q^4}$, and the evaluation is done!
A good rule of thumb is that when you're activating a layer, it's clamped in the range $[0; X_b]$ where $X_b$ is the quantization constant of that layer's bias.

#### Output buckets

If you're doing multilayer, you've probably done output buckets, so remember that all weights and biases from $w1$ and $b1$ onward have output buckets.

## Permutations

To enable efficient inference, you will have to permute the weights and biases of different layers so that they are laid out contiguously in memory for the SIMD registers.
Some of these permutations can be done beforehand, while others will have to be done at build time depending on the target architecture.

### L0

If you are doing pairwise multiplication, chances are you are using packus to store two i16 registers into one u8 register.

- on AVX2, this is done with `_mm256_packus_epi16(a, b)`
- on AVX512, this is done with `_mm512_packus_epi16(a, b)`

At first glance, it is easy to assume that these instructions simply concatenate the results from `a` and `b`.
However, in reality, packus takes chunks of 8x i16 values from each register and interleaves them.
As a result, the output layout differs from a simple `[a, b]` arrangement.

On AVX2, the output is:

```cpp
[a[0:8] b[0:8] a[8:16] b[8:16]]
```

While on AVX512, the output is

```cpp
[a[0:8] b[0:8] a[8:16] b[8:16] a[16:24] b[16:24] a[24:32] b[24:32]]
```

To counteract this interleaving, you will want to permute the weights and bias of layer 0 based on the target architecture so that the output order is always `[a, b]`.

The easiest way to do this is to write a script that gets called during built-time, that looks at the current permutation of the l0 weights and biases, and applies an appropriate permutation for the target architecture.
You will want to store the current permutation as a flag somewhere in the network if you are modifying the network file in place.

The exact permutation depends on the SIMD width.
You want to permute $w0$ and $b0$ along the output dimension.

AVX2:

- permute in chunks of $4\times8=32$ values
- order: `[0, 2, 1, 3]`

AVX512:

- permute in chunks of $8\times8=64$ values
- order: `[0, 4, 1, 5, 2, 6, 3, 7]`

### L1

Again, the details are in [this](https://aletheiaaaaa.github.io/posts/2025-07-14-dpbusd-explained/) article, but to summarize, we want to go from:

```cpp
W1[L1_SIZE][NUM_OUTPUT_BUCKETS][L2_SIZE]
```

to

```cpp
W1[NUM_OUTPUT_BUCKETS][L1_SIZE / 4][L2_SIZE * 4]
```

assuming the input comes from an un-transposed bullet export, and parallelization along the output dimension (i.e., your dpbusd computes multiple output values at once, without using a horizontal add).
You can think of it as the first axis being the bucket index, the second axis being the tile index, and the third axis containing the data for that tile.

Note that you can do this permutation (and the permutation for the later layers) when you're exporting the model, as it does not depend on the target architecture.

### L2 onwards

There aren't any special permutation rules for the later layers.
You just want to make sure they are indexed in order of `bucket`, `input`, `output` so that the outputs are layed out contiguously in memory.
