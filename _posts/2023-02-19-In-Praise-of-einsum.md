---
layout: post
title: In Praise of Einsum
date: 2023-02-19 09:10:17
description: A tutorial on einsum, Einstein summation notation.
tags: jax numpy machine-learning
categories: blog
related_posts: false
toc: true
math: true
---

This is a short note about the `einsum` functionality that is present in numpy, jax, etc. Understanding what it does is a bit tricky --naturally, because it can do the job of many other functions-- but it is also very rewarding as it can help a lot with linear algebraic computations. I will use numpy's `np.einsum()` notation, but the underlying concepts are the same regardless of slight syntactic differences in libraries.

Put simply, `einsum` allows you to convert simple sums over product of elements of matrices from math into code. It can also change the order of axes of a matrix (computing transposes). For instance, let's say we want to compute the product of two matrices $A, B$ and store them in a new matrix $C$. Mathematically, we can specify $C$ as
$$
C\_{ik} = \sum\_{j} A\_{ij}B\_{jk}.
$$
In code, this could be written as `C = np.einsum('ij, jk -> ik', A, B)`.
First, I will discuss exactly where we can use `einsum`. Then, I'll explain how to use `einsum` to convert math to code and vice versa. Finally, I will show how to use `einsum` to perform some everyday linear algebraic computations.

## Where?
Generally, speaking whenever we have a number of tensors $A^1, \cdots, A^n$ and we want to obtain a new tensor $B$ whose elements can be specified in the general form of 
$$B\_{i\_1, \cdots, i\_k} = \sum\_{j\_1, \cdots, j\_l} A^1A^2\cdots A^n$$
then we can use `einsum` to code up the operations. This general form, which I like to call the *sum-of-product* form comes up frequently in linear algebraic computations. (Batch) matrix multiplication, matrix-vector product, vector outer-product, computing the trace, etc. are all computations that can be written in this form. In the examples section further down, a number of these applications are discussed.

## How?
The `einsum` function takes two sets of arguments; a specification string and the tensors over which the computations are to be performed. The output of this function is a tensor containing the result. Remember the code for matrix multiplication: `C = np.einsum('ij, jk -> ik', A, B)`. Here, `'ij, jk -> ik'` is the specification string, `A, B` are input tensors and `C` is the output. In the explanations below, I will use this matrix multiplication code.

Let's see how we can interpret the specification string. We can easily convert any `einsum` code into a sum-of-product formula by doing the following:
1. Write down the expression for multiplying the elements of the **input tensors** specified by the indices that are on the **left hand side** of `->` symbol in the specification string. For the matrix multiplication example, this would be $$A\_{ij}B\_{jk}.$$
2. Write down the element of the output tensor specified by the indices on the **right hand side** of the `->` symbol. For the matrix multiplication example, this would be $$C\_{ik}.$$
3. Next we have to identify what I call *slack indices*. These are the indices that are used on the left hand side of the `->` symbol, but not on the right hand side. In other words, these are the indices that were used for the inputs in step 1, but not for the output in step 2. For the matrix multiplication example this would be the `j` index.
4. Compute the sum of the expression in step 1 over all the slack indices to get the element of the output in step 2. For the matrix multiplication example we would write $$\underbrace{C\_{ik}}\_{\text{step 2}} = \sum\_{\underbrace{j}\_{\text{slack index}}} \underbrace{A\_{ij}B\_{jk}}\_{\text{step 1}}.$$
And we are done! Of course, we can convert any sum-of-product expression into an `einops` code following the same logic. See the examples in the next section to get a better grip on these.
There are a couple of things that you should keep in mind:
+ In the specification string, there shouldn't be any index on the right hand side that is not used on the left hand side. So we can't have something like `ij, jk -> il` because `l` is not used on the left hand side. This is logical if you think about it in terms of the corresponding sum-of-product equation.
+ Dimensions of the input that correspond to the same indices in the specification string should be the same. In the matrix multiplication example, the index `j` is used in for both inputs `A` and `B`. Because of this, the corresponding dimensions (first dimension of `A` and second dimension of `B`) must have the same length, which is as if we were saying two matrices $A\_{mn}, B\_{kl}$ can be multiplied together, only if $n=k$. 

## Examples
### Matrix Vector Product
If we have a vector $v = [v\_1, \cdots, v\_n]^T$ and an $m\times n$ matrix $A$, then $u = Av$ is an $m\times 1$ vector whose $i$-th element is the dot product of $v$ with the $i$-th row of $A$. Mathematically, we can represent $u$ in a sum-of-product form:
$$
u\_i = \sum\_j A\_{ij}v\_j.
$$
So the specification string is `'ij, j -> i'`. Notice that here $j$ is a slack index that we sum over. The final code is `u = np.einsum('ij, j -> i', A, v)`.
### Inner Product
The inner product of two $n$ dimensional vectors $u$ and $v$ is a single scalar $p$ which has a sum-of-product form:
$$
p = \sum\_i u\_iv\_i.
$$
This means that the corresponding specification string is `i, i ->`. Notice that there is no index on the right hand side of this string, which means that the output is a scalar and that $i$ is an slack index that we sum over. The final code is `p = np.einsum('i, i -> ', u, v)`.
### Outer Product
If we have an $m$ dimensional vector $u$ and an $n$ dimensional vector $v$, then their outer product $A = u \otimes v$ is a rank-1, $m\times n$ matrix where the $i$-th column is $u$ multiplied by the $i$-th element of $v$. We can represent $A$ in the sum-of-product form:
$$
A\_{ij} = u\_i v\_j.
$$
In the code, we can compute it by `A = np.einsum('i, j -> ij', u, v)`. Notice that here there are no slack indices.
### Row Sum and Column Sum
We can use `einsum` to compute the sum of all elements in each row. For a matrix $A$, the result would be a vector $r$ where 
$$
r\_i = \sum\_j A\_{ij}.
$$
We can turn this into a specification string and write `r = np.einsum('ij -> i', A)`.
Similarly, to compute the sum of all elements in each column we can use `c = np.einsum('ij -> j', A)`.
When we have multi-dimensional tensors and we want to compute their sum over an axis, `einsum` notation could help with the clarity of the code. For instance, if we write
```
Y = np.einsum('BCmn -> BC', X)
```
we can immediately say that `A` has the shape `B x C x m x n` (perhaps a batch of `B` images, each with `C` channels and size `m x n`) and for each channel in each batch, we have computed the sum of all elements to arrive at a tensor that has the shape `B x C`. Contrast this with
```
Y = np.sum(X, axis=(2,3))
```
which does the same job. So `einsum` could help you track the shapes as well!
### Trace
The trace of a matrix $A$ is a scalar $t$ which is the sum of all elements on its main diagonal. In the sum-of-product form, this is represented by
$$
t = \sum\_i A\_{ii}.
$$
This can be coded as `t = np.einsum('ii -> ', A)`. Notice here how the index $i$ is used twice for referencing the same input argument.
### Main Diagonal
Similar to the way we computed the trace, we can extract the main diagonal of a matrix as a vector. In the sum-of-product form, the main diagonal can be seen as
$$
d\_i = A\_{ii}.
$$
We can code this as `d = np.einsum('ii -> i', A)`.
### Transpose
Computing the transpose of a matrix $A$ is also very easy using `einsum`. The sum-of-product notation would simply be $B\_{ji} = A\_{ij}$ and the corresponding code is `B = np.einsum('ij -> ji', A)`.
### Batch Matrix Multiplication
Adding one (or more) batch dimension is very easy using the `einsum` notation. if `A` and `B` are batches of matrices (batch index comes first) that are to be multiplied, then we can write `C = np.einsum('nij, njk -> nik', A, B)`. If we write down the corresponding sum-of-product expression, it becomes evident that the batch index just acts as a counter, not involved in the computations.
$$
C\_{nik} = \sum\_{j} A\_{nij}B\_{njk}
$$
So for the first elements in the batch $n=0$, we would have
$$
C\_{0,ik} = \sum\_{j} A\_{0,ij}B\_{0,jk}.
$$
Which means that the first element in the output batch, `C[0]`, is the matrix product of `A[0]` and `B[0]`.
We can similarly add a batch dimension to any of the other examples. There is also one nice trick when we have more than one batch dimension. We can write  `np.einsum('...ij, ...ji -> ...ik', A, B)` to avoid explicitly writing all batch dimensions that proceed the last two dimensions, over which we want to perform the multiplication.
