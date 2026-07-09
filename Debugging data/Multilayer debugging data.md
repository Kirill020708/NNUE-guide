Example net can be found in `net_examples/multibeans.bin`

### Test positions

`startpos`: `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1`

`kiwipete`: `r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1`

### Network parameters
- Architecture: `(768hm->64)x2->(16->32->1)`
  - No input/output buckets
  - Mirroring enabled
  - No pairwise multiplication
- $Q0=255, Q1=128, Q=64, S=400$
- SCReLU activation for first two layers, CReLU for last layer

### File data
Here are the layers and their memory layout.
Note that the layout is the same as what you would get from bullet without transpose.

```cpp
int16_t w0[768][64];
int16_t b0[64];
int8_t  w1[2 * 64][16];
int32_t b1[16];
int32_t w2[16][32];
int32_t b2[32];
int32_t w3[32];
int32_t b3;
```

First $8$ values:

| Parameter             | Values                                             |
| --------------------- | -------------------------------------------------- |
| `w0[:8][32]`          | `-3 2 -9 3 -3 -2 19 -8`                            |
| `b0[:8]`              | `-48 88 69 12 26 36 81 127`                        |
| `w1[:8][8]`           | `4 -9 21 40 11 -5 11 -21`                          |
| `b1[:8]`              | `2 -3 6 11 -11 7 22 3`                             |
| `w2[:8][8]`           | `11 -6 7 -10 -25 -2 -17 31`                        |
| `b2[:8]`              | `-12382 59442 10176 34484 20167 47785 10864 81807` |
| `w3[:8]`              | `54 33 -22 -16 -20 -127 20 82`                     |
| `b3`                  | `525860`                                           |

Note that after permuting `w1` according to the [Multilayer guide](<Basic articles/Multilayer.md>), you should get this instead:

```cpp
int8_t w1[2 * 64 / 4][16 * 4];  // w1[:8][8] = 14 8 34 -4 -30 -6 26 47
```

### Intermediate values and activations

$L_i$ act/val are the first few activations/values of hidden layer number $i$ (assuming that the 1st hidden layer is the layer right after the input layer). `Sum w\o bias` is the value of the output neuron before adding output bias. L2 values are given before and after requantizing from $\frac{Q0^2 \cdot Q1}{2^9}$ to $Q$.

###### Startpos

$L_1$ act: `34 2 2 0 2 72 0 0 8 0 11 1 0 17 58 0`

$L_2$ val before requantizing (without bias): `5379 -14524 1587 8194 -3663 773 20793 -18305 2517 -1907 -402 97 8823 10048 3859 1937`

$L_2$ val after requantizing (with bias): `23 -60 12 43 -26 10 103 -69 8 0 5 8 40 54 23 4`

$L_2$ act: `529 0 144 1849 0 100 4096 0 64 0 25 64 1600 2916 529 16`

$L_3$ val: `-170472 318627 51166 84534 185813 -277798 131251 -45139 11540 -415313 186079 50542 21729 -241406 -141165 63983 32242 43604 -66495 -127265 175788 88921 9674 199213 -31799 46973 143275 -21347 -275227 94366 -51841 -156536`

$L_3$ act: `0 262144 51166 84534 185813 0 131251 0`

Sum w\o bias: `3843096`

###### Kiwipete

$L_1$ act: `3 1 34 0 0 51 16 0 2 6 55 0 0 34 30 0`

$L_2$ val before requantizing (without bias): `7672 -13317 2688 9139 -3786 4150 17748 -22667 4429 -1665 557 -570 9310 11303 3613 6034`

$L_2$ val after requantizing (with bias): `31 -56 16 46 -26 23 91 -86 16 1 9 5 42 59 22 20`

$L_2$ act: `961 0 256 2116 0 529 4096 0 256 1 81 25 1764 3481 484 400`

$L_3$ val: `-172382 334874 9820 64419 228549 -390075 126248 -90987 33250 -483002 215208 29934 -5994 -262386 -196672 40611 44524 32657 -46830 -167771 197904 117013 -29170 277772 -54705 43444 169679 11243 -286980 81488 -48648 -185513`

$L_3$ act: `0 262144 9820 64419 228549 0 126248 0`

Sum w\o bias: `-199538`


### Final evaluation
`startpos`: 104 cp
`kiwipete`: 7 cp