Example net can be found in `net_examples/beans.bin`

### Test positions

`startpos`: `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1`

`kiwipete`: `r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1`

### Network parameters
- Hidden layer size = 64 (therefore $N=32$)
- activation is SCReLU
- $Q_a=255, Q_b=64, S=400$

### File data
The first 16 weights of the hidden layer's 32nd neuron connected from the input layer (assuming numbering from zero) ($w0_{32}$): `0 0 0 -1 0 0 0 0 10 8 5 12 8 6 -11 4`

First 16 weights of output neuron weights ($w1$): `122 70 -70 -126 -30 -22 -58 -66 37 32 -90 -17 -111 -25 -78 -27`

First 16 biases of hidden layer ($b0$): `176 33 18 47 9 64 104 -24 161 85 58 180 23 57 6 36`

Output neuron bias($b1$): `825`

### Active indices
`startpos` (both perspectives): `192 65 130 259 324 133 70 199 8 9 10 11 12 13 14 15 432 433 434 435 436 437 438 439 632 505 570 699 764 573 510 639`

`kiwipete` (white): `192 324 199 8 9 10 139 140 13 14 15 82 277 407 409 28 35 100 552 489 428 493 430 432 434 435 692 437 566 632 764 639`

`kiwipete` (black): `632 764 639 432 433 434 563 564 437 438 439 490 685 47 33 420 411 476 144 81 20 85 22 8 10 11 268 13 142 192 324 199`

### First 16 accumulator values (pre-activation)
`startpos` (both perspectives): `-1233 106 168 -515 401 268 5 134 565 564 -26 233 -346 253 131 237`

`kiwipete` (white): `-1326 140 57 -500 539 265 -180 81 574 576 42 271 -260 286 -52 287`

`kiwipete` (black): `-1296 138 83 -485 511 229 7 97 575 565 2 -174 -279 285 153 303`

### Unscaled eval without output bias
`startpos`: `608404`
`kiwipete`: `-1423747`

### Final evaluation
`startpos`: `78` `kiwipete`: `-116`

Credits to [Ciekce](https://github.com/Ciekce) for training the net