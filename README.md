gpuKmeans
=========

kmenas on gpu


Kmeans implemented on CUDA 6.0+, tested under linux (ubuntu 14.04) and windows 7.

Speed is 100x faster than openomp of CPU. Deal with 1 million 128 dimension dataset within 1 minute.

Resutl still needs to check.

There are three main steps in the implementation:

1. calculate belongings.
 
2. use a self-built map (a 1D array of int) on GPU to recod data-center correspondings.

3. update centers.

The step 1 and step 3 is well parallelized, while the calculations in step 2 is sequential.

For example, there are n data vectors, and we want to cluster them into k (k<<n) clusters:

For example n=7, and k=2.

We will do the following:

1. randomly select k data vectors as initial centers

2. use lots of GPUs to find the nearest center from the k centers for each data vector. one example is like: "0111100" (0 means belonging to the first cluster, while 1 means the second)

3. since there is no map on GPU, we will turn "0111100" into three arrays "0561234", "03", and "34". Obviously "34" means the size of the first cluster is 3, and the size of the second cluster is 4. "03" means in "0561234" from index "0" are  the records for the first cluster, while from index "3" are for the second cluster. These efforts will make the parallization for updating centers feasible.

4. update the centers. To update the first center, we see in "0561234", the first index is "0" as from "02", and its size  is "3" as from "34". So the center is updated with the data vectors of "0", "5", and "6".

The procedure will run until convergence (no data vector changes its cluster label) or a max iteration number is reached.




