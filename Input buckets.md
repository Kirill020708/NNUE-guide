The king is a very important piece on the board. Because of it, the evaluation of a position can depend heavily on king's location. To help the network understand this concept, we do **input buckets**: we choose hidden layer weights and biases based on where the king is located. Generally, you have an array which tells you the bucket number for every square. Example for 4 buckets:
```python
buckets = [
	3,3,3,3,3,3,3,3,
	3,3,3,3,3,3,3,3,
	3,3,3,3,3,3,3,3,
	3,3,3,3,3,3,3,3,
	3,3,3,3,3,3,3,3,
	3,3,3,3,3,3,3,3,
	2,2,2,2,2,2,2,2,
	1,1,0,0,0,0,1,1
]
```
The implementation is very similar to Mirroring: every time the king changes its bucket, you recalculate the network from scratch. You also calculate the accumulator for a color based on that color's king position. Because you need a separate $w0$ matrix for every bucket, the number of weights increases almost proportionally with the number of buckets, so you need much more data for them to work. Generally, you start with 4 buckets and gradually increase their number, testing if it gains every time.