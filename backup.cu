/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

static const int WORK_SIZE = 256;

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

__host__ __device__ unsigned int bitreverse(unsigned int number) {
	number = ((0xf0f0f0f0 & number) >> 4) | ((0x0f0f0f0f & number) << 4);
	number = ((0xcccccccc & number) >> 2) | ((0x33333333 & number) << 2);
	number = ((0xaaaaaaaa & number) >> 1) | ((0x55555555 & number) << 1);
	return number;
}

/**
 * CUDA kernel function that reverses the order of bits in each element of the array.
 */

__global__ void adda(int count,double* a,double *b,double* c,int st)
{

	int i= st+blockDim.x* blockIdx.x +threadIdx.x;
	if(i<count)
		c[i]=a[i]+b[i];
}

__global__ void mm(int count,double* c,int st)
{

	int i= st+blockDim.x* blockIdx.x +threadIdx.x;
	if(i<count)
		c[i]/=2;
}

void launch(double* a,double* b,double* c,int testsize)
{
	int threadsize=256;
	int blocksize=256;
	for(int i=0;i<testsize;i+=threadsize*blocksize)
		adda<<<blocksize,threadsize>>>(testsize,a,b,c,i);

	for(int i=0;i<testsize;i+=threadsize*blocksize)
		mm<<<blocksize,threadsize>>>(testsize,c,i);

	cudaDeviceSynchronize();
}

int helloworld()
{
	int testsize=10000000;
	double* a;
	double* b;



	cudaMallocManaged(&a,sizeof(double)*testsize);
	cudaMallocManaged(&b,sizeof(double)*testsize);

	for (int i = 0; i < testsize; i++)
	{
		a[i]=i;
		b[i]=testsize-i;
	}
	double* c;


	cudaMallocManaged(&c,sizeof(double)*testsize);
	launch(a,b,c,testsize);

	for (int i = 0; i < testsize; i++)
	{

		if(i%1000000==0)
			printf("%f ",c[i]);

	}

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);

	cudaDeviceReset();
    return 0;
}




__global__ void updatebelonging(int index,double* dataset,int datasize,int dimension,double* centers,int* labels,int kCenter,bool* goodCenterFlag,int* paraClusterCount,double* paraCenters,bool* paraCenterChangeFlag)
{
	int j=index+blockDim.x* blockIdx.x +threadIdx.x;
	int pind=blockDim.x* blockIdx.x +threadIdx.x;
	if(j<datasize)
	{
		int tlabel;
		int firstindex=0;
		while(firstindex<kCenter && (!goodCenterFlag[firstindex]))
			++firstindex;

		tlabel=firstindex;
		double minDis=0.0;

		for(int i=0;i<dimension;++i)
		{
			double tdis=dataset[j*dimension+i]-centers[firstindex*dimension+i];
			minDis+=tdis*tdis;
		}
		for(int k=firstindex+1;k<kCenter;++k)
		{
			if(goodCenterFlag[k])
			{
				double curdis=0.0;
				for(int i=0;i<dimension;++i)
				{
					double tdis=dataset[j*dimension+i]-centers[k*dimension+i];
					curdis+=tdis*tdis;
				}
				if(curdis<minDis)
				{
					minDis=curdis;
					tlabel=k;
				}
			}
		}
		if(tlabel!=labels[j])
			paraCenterChangeFlag[blockDim.x* blockIdx.x +threadIdx.x]=true;

		labels[j]=tlabel;
		++paraClusterCount[pind*kCenter+tlabel];
		for(int k=0;k<dimension;++k)
		{
			paraCenters[pind*(kCenter*dimension)+tlabel*dimension+k]+=dataset[j*dimension+k];
		}
	}

}
__global__ void updateCenter(int index,int dimension,int parallelNumber,double* centers,int kCenter,bool* goodCenterFlag,int* paraClusterCount,double* paraCenters,int* clusterCount)
{
	int j=index+blockDim.x* blockIdx.x +threadIdx.x;
	if(j<kCenter)
	{
		for(int i=0;i<dimension;++i)
			centers[j*dimension+i]=0.0;

		clusterCount[j]=0;

		for(int i=0;i<parallelNumber;++i)
		{
			for(int l=0;l<dimension;++l)
			{
				centers[j*dimension+l]+=paraCenters[i*kCenter*dimension+j*dimension+l];
				paraCenters[i*kCenter*dimension+j*dimension+l]=0.0;
			}
			clusterCount[j]+=paraClusterCount[i*kCenter+j];
			paraClusterCount[i*kCenter+j]=0;
		}
		if(clusterCount[j]==0)
			goodCenterFlag[j]=false;
		else
		{
			for(int i=0;i<dimension;++i)
				centers[j*dimension+i]/=clusterCount[j];
		}
	}

}
void kmeans(double* dataset,int datasize,int dimension,double* centers,int* labels,int kCenter,int maxIterationNumber)
{
	int threadsize=32;
	int blocksize=32;
	vector<int> initialCenterIndex=shuffledOrder(datasize,kCenter);

	for(int i=0;i<kCenter;++i)
		for(int j=0;j<dimension;++j)
			centers[i*dimension+j]=dataset[initialCenterIndex[i]*dimension+j];

	/*
	 vector<bool> goodCenterFlag(kCenter,true);
	vector<vector<int> > paraClusterCount(parallelNumber,vector<int>(kCenter,0));
	vector<int> clusterCount(kCenter,0);

	vector<vector<vector<double> > > paraCenters(parallelNumber,vector<vector<double> >(kCenter,vector<double>(dataset[0].size(),0.0)));

//	vector<bool> centerChangeFlag(dataset.size(),false);
	vector<bool> paraCenterChangeFlag(parallelNumber,false);
	 */

	int parallelNumber=threadsize*blocksize;
	bool* goodCenterFlag;
	int* paraClusterCount;
	int* clusterCount;
	double* paraCenters;
	bool* paraCenterChangeFlag;

	cudaError_t cudaStatus;
	cudaMallocManaged(&goodCenterFlag,sizeof(bool)*kCenter);
	printf("\ns1\n");
	cudaMallocManaged(&paraClusterCount,sizeof(int)*parallelNumber*kCenter);
	printf("s2\n");
	cudaMallocManaged(&clusterCount,sizeof(int)*kCenter);
	printf("s3\n");
	cudaStatus=cudaMallocManaged(&paraCenters,sizeof(double)*dimension*kCenter*parallelNumber);


	printf("s4\n");
	cudaMallocManaged(&paraCenterChangeFlag,sizeof(bool)*parallelNumber);
	printf("s5\n");

	for(int i=0;i<kCenter;++i) goodCenterFlag[i]=true;
	printf("\ns1\n");
	for(int i=0;i<parallelNumber*kCenter;++i) paraClusterCount[i]=0;
	printf("s2\n");
	for(int i=0;i<kCenter;++i) clusterCount[i]=0;
	printf("s3\n");
	for(int i=0;i<dimension*kCenter*parallelNumber;++i)
	{
		paraCenters[i]=0.0;
	}
	printf("s4\n");
	for(int i=0;i<parallelNumber;++i) paraCenterChangeFlag[i]=false;
	printf("s5\n");

	for(int iterN=0;iterN<maxIterationNumber;++iterN)
	{
		for(int i=0;i<datasize;i+=parallelNumber)
		{
			//printf("%d ",i);
			updatebelonging<<<blocksize,threadsize>>>(i,dataset,datasize,dimension,centers,labels,kCenter,goodCenterFlag,paraClusterCount,paraCenters,paraCenterChangeFlag);
		}
		printf("belongings ok\n");

		for(int i=0;i<kCenter;i+=parallelNumber)
		{
			updateCenter<<<blocksize,threadsize>>>(i,dimension,parallelNumber,centers,kCenter,goodCenterFlag,paraClusterCount,paraCenters,clusterCount);
		}
		printf("center ok\n");
		//cudaDeviceSynchronize();
		/*bool noChange=true;

		for(int i=0;i<parallelNumber;i++)
		{
			if(paraCenterChangeFlag[i]==true)
			{
				noChange=false;
				paraCenterChangeFlag[i]=false;
			}
		}
		if(noChange)
			break;*/
		printf("finished iteration NO. %d\n",iterN);

	}

	cudaDeviceSynchronize();
	cudaFree(goodCenterFlag);
	printf("s7\n");
	cudaFree(paraClusterCount);
	printf("s8\n");
	cudaFree(clusterCount);
	printf("s9\n");
	cudaFree(paraCenters);
	printf("s10\n");
	cudaFree(paraCenterChangeFlag);
	printf("s11\n");
}


__global__ void updatebelonging2(int index,double* dataset,int datasize,int dimension,double* centers,int* labels,int kCenter,bool* goodCenterFlag,int* paraClusterCount,bool* paraCenterChangeFlag)
{
	int j=index+blockDim.x* blockIdx.x +threadIdx.x;
	int pind=blockDim.x* blockIdx.x +threadIdx.x;
	if(j<datasize)
	{
		int tlabel;
		int firstindex=0;
		while(firstindex<kCenter && (!goodCenterFlag[firstindex]))
			++firstindex;

		tlabel=firstindex;
		double minDis=0.0;

		for(int i=0;i<dimension;++i)
		{
			double tdis=dataset[j*dimension+i]-centers[firstindex*dimension+i];
			minDis+=tdis*tdis;
		}
		for(int k=firstindex+1;k<kCenter;++k)
		{
			if(goodCenterFlag[k])
			{
				double curdis=0.0;
				for(int i=0;i<dimension;++i)
				{
					double tdis=dataset[j*dimension+i]-centers[k*dimension+i];
					curdis+=tdis*tdis;
				}
				if(curdis<minDis)
				{
					minDis=curdis;
					tlabel=k;
				}
			}
		}
		if(tlabel!=labels[j])
			paraCenterChangeFlag[blockDim.x* blockIdx.x +threadIdx.x]=true;

		labels[j]=tlabel;
		++paraClusterCount[pind*kCenter+tlabel];
		for(int k=0;k<dimension;++k)
		{
//			paraCenters[pind*(kCenter*dimension)+tlabel*dimension+k]+=dataset[j*dimension+k];
		}
	}

}
__global__ void updateCenter2(int index,int dimension,int parallelNumber,double* centers,int kCenter,bool* goodCenterFlag,int* paraClusterCount,int* clusterCount)
{
	int j=index+blockDim.x* blockIdx.x +threadIdx.x;
	if(j<kCenter)
	{
		for(int i=0;i<dimension;++i)
			centers[j*dimension+i]=0.0;

		clusterCount[j]=0;

		for(int i=0;i<parallelNumber;++i)
		{
			for(int l=0;l<dimension;++l)
			{
				//centers[j*dimension+l]+=paraCenters[i*kCenter*dimension+j*dimension+l];
				//paraCenters[i*kCenter*dimension+j*dimension+l]=0.0;
			}
			clusterCount[j]+=paraClusterCount[i*kCenter+j];
			paraClusterCount[i*kCenter+j]=0;
		}
		if(clusterCount[j]==0)
			goodCenterFlag[j]=false;
		else
		{
			for(int i=0;i<dimension;++i)
				centers[j*dimension+i]/=clusterCount[j];
		}
	}

}
void kmeans2(double* dataset,int datasize,int dimension,double* centers,int* labels,int kCenter,int maxIterationNumber)
{
	int threadsize=256;
	int blocksize=256;
	vector<int> initialCenterIndex=shuffledOrder(datasize,kCenter);

	for(int i=0;i<kCenter;++i)
		for(int j=0;j<dimension;++j)
			centers[i*dimension+j]=dataset[initialCenterIndex[i]*dimension+j];

	/*
	 vector<bool> goodCenterFlag(kCenter,true);
	vector<vector<int> > paraClusterCount(parallelNumber,vector<int>(kCenter,0));
	vector<int> clusterCount(kCenter,0);

	vector<vector<vector<double> > > paraCenters(parallelNumber,vector<vector<double> >(kCenter,vector<double>(dataset[0].size(),0.0)));

//	vector<bool> centerChangeFlag(dataset.size(),false);
	vector<bool> paraCenterChangeFlag(parallelNumber,false);
	 */

	int parallelNumber=threadsize*blocksize;
	bool* goodCenterFlag;
	int* paraClusterCount;
	int* clusterCount;
	//double* paraCenters;
	bool* paraCenterChangeFlag;

	cudaError_t cudaStatus;
	cudaMallocManaged(&goodCenterFlag,sizeof(bool)*kCenter);
	printf("\ns1\n");
	cudaMallocManaged(&paraClusterCount,sizeof(int)*parallelNumber*kCenter);
	printf("s2\n");
	cudaMallocManaged(&clusterCount,sizeof(int)*kCenter);
	printf("s3\n");
	//cudaStatus=cudaMallocManaged(&paraCenters,sizeof(double)*dimension*kCenter*parallelNumber);


	printf("s4\n");
	cudaMallocManaged(&paraCenterChangeFlag,sizeof(bool)*parallelNumber);
	printf("s5\n");

	for(int i=0;i<kCenter;++i) goodCenterFlag[i]=true;
	printf("\ns1\n");
	for(int i=0;i<parallelNumber*kCenter;++i) paraClusterCount[i]=0;
	printf("s2\n");
	for(int i=0;i<kCenter;++i) clusterCount[i]=0;
	printf("s3\n");

	printf("s4\n");
	for(int i=0;i<parallelNumber;++i) paraCenterChangeFlag[i]=false;
	printf("s5\n");

	for(int iterN=0;iterN<maxIterationNumber;++iterN)
	{
		for(int i=0;i<datasize;i+=parallelNumber)
		{
			//printf("%d ",i);
			updatebelonging2<<<blocksize,threadsize>>>(i,dataset,datasize,dimension,centers,labels,kCenter,goodCenterFlag,paraClusterCount,paraCenterChangeFlag);
		}
		printf("belongings ok\n");

		for(int i=0;i<kCenter;i+=parallelNumber)
		{
			updateCenter2<<<blocksize,threadsize>>>(i,dimension,parallelNumber,centers,kCenter,goodCenterFlag,paraClusterCount,clusterCount);
		}
		printf("center ok\n");
		//cudaDeviceSynchronize();
		/*bool noChange=true;

		for(int i=0;i<parallelNumber;i++)
		{
			if(paraCenterChangeFlag[i]==true)
			{
				noChange=false;
				paraCenterChangeFlag[i]=false;
			}
		}
		if(noChange)
			break;*/
		printf("finished iteration NO. %d\n",iterN);

	}

	cudaDeviceSynchronize();
	cudaFree(goodCenterFlag);
	printf("s7\n");
	cudaFree(paraClusterCount);
	printf("s8\n");
	cudaFree(clusterCount);
	printf("s9\n");

	printf("s10\n");
	cudaFree(paraCenterChangeFlag);
	printf("s11\n");
}



//this is not going to work, there are not enough of shared memory
__global__ void updatebelonging3(int index,double* dataset,int datasize,int dimension,double* centers,int* labels,int kCenter,bool* goodCenterFlag,double* blockCenters,bool* CenterChangeFlag)
{
	int j=index+blockDim.x* blockIdx.x +threadIdx.x;

	extern __shared__ double localBlockCenters[];
	//int pind=blockDim.x* blockIdx.x +threadIdx.x;

	for (int i = 0; i < dimension*kCenter; i++)
	{
		localBlockCenters[threadIdx.x*(dimension*kCenter)+ i]=0.0;
	}
	int tlabel;

	if(j<datasize)
	{
		tlabel=minIndex(dataset+j*dimension, centers,goodCenterFlag,kCenter,dimension);

		if(tlabel!=labels[j])
			CenterChangeFlag[j]=true;

		labels[j]=tlabel;
		for (int i = 0; i < dimension; i++)
		{
			localBlockCenters[threadIdx.x*(dimension*kCenter)+ tlabel* dimension+i]+=dataset[j*dimension+i];
		}

	}
	__syncthreads();
	for (int size = blockDim.x; size >1 ; size=(size+1)/2)
	{
		int offset=size/2;
		if(threadIdx.x<offset)
		for (int i = 0; i < dimension*kCenter; i++)
		{
			localBlockCenters[threadIdx.x*(dimension*kCenter)+ i]+=localBlockCenters[(size-1-threadIdx.x)*(dimension*kCenter)+ i];
		}
		__syncthreads();
	}

	if(threadIdx.x==0)
	{
		for (int i = 0; i < dimension*kCenter; i++)
		{
			blockCenters[blockIdx.x*dimension*kCenter+i]=localBlockCenters[i];
		}
	}
}



__global__ void updateCenter3(int index,int dimension,int parallelNumber,double* centers,int kCenter,bool* goodCenterFlag,int* blockCenterCount,int* centerCount)
{
	int j=index+blockDim.x* blockIdx.x +threadIdx.x;

	if(j<kCenter)
	{
		for(int i=0;i<dimension;++i)
			centers[j*dimension+i]=0.0;

		centerCount[j]=0;

		for(int i=0;i<parallelNumber;++i)
		{
			for(int l=0;l<dimension;++l)
			{
				//centers[j*dimension+l]+=paraCenters[i*kCenter*dimension+j*dimension+l];
				//paraCenters[i*kCenter*dimension+j*dimension+l]=0.0;
			}
			//clusterCount[j]+=paraClusterCount[i*kCenter+j];
		//	paraClusterCount[i*kCenter+j]=0;
		}

	}

}
