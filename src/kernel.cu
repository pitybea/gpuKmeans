
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <unistd.h>
#include <vector>
#include <iostream>
using namespace std;

#ifndef __CUDACC__  
    #define __CUDACC__
#endif


__device__ int minIndex(double* data,double* centers,bool* centerflags,int kCenter,int dimension)
{
	int result;
	int firstindex=0;
	while(firstindex<kCenter && (! centerflags[ firstindex]))
		++firstindex;

	result=firstindex;
	double mindis=0.0;

	for (int i = 0; i < dimension; i++)
	{
		double tdis=centers[firstindex*dimension+i]-data[i];
		mindis+=tdis*tdis;
	}

	for(int i=firstindex+1;i<kCenter;++i)
	{
		if(centerflags[i])
		{
			double tdis=0.0;
			for (int j = 0; j < dimension; j++)
			{
				double ttdis=centers[i*dimension+j]-data[j];
				tdis+=ttdis*ttdis;
			}
			if(tdis<mindis)
			{
				mindis=tdis;
				result=i;
			}
		}
	}

	return result;
}
vector<int> shuffledOrder(int n,int m)
{
	//assert(n>=m);
	vector<int> result(m);
	vector<int> index(n);
	for(int i=0;i<n;++i)
	{
		index[i]=i;
	}

	for(int i=0;i<m;++i)
	{
		int tem=rand()%(n-i);
		result[i]=index[tem];
		index[tem]=index[n-i-1];

	}
	return result;
}
__global__ void updatebelonging4(int index,double* dataset,int datasize,int dimension,double* centers,int* labels,int kCenter,bool* goodCenterFlag,bool* CenterChangeFlag)
{
	int j=index+blockDim.x* blockIdx.x +threadIdx.x;

	int tlabel;
	if(j<datasize)
	{
		tlabel=minIndex(dataset+j*dimension, centers,goodCenterFlag,kCenter,dimension);
		if(tlabel!=labels[j])
			CenterChangeFlag[j]=true;
		labels[j]=tlabel;

	}
	
	
}

__global__ void updateCorresponds(int* labels,int datasize,int kCenter,int* correspondings,bool* centerChangeFlag,int* centerStartIndex,int* centerCount,int* curCount,bool* goodCenterFlag,bool* nochange)
{
//	cudaMemset(correspondings,0,sizeof(int)*datasize);
//	cudaMemset(centerCount,0,sizeof(int)*kCenter);
	for (int i = 0; i < kCenter; i++)
	{
		centerCount[i]=0;
		curCount[i]=0;
	}
	for (int i = 0; i < datasize; i++)
	{
		++centerCount[labels[i]];

		if(centerChangeFlag[i])
		{
			*nochange=false;
			centerChangeFlag[i]=false;
		}	
	}
	centerStartIndex[0]=0;
	for (int i = 0; i < kCenter; i++)
	{
		if (centerCount[i]==0)
		{
			goodCenterFlag[i]=false;
		}
		if(i>0)
		{
			centerStartIndex[i]=centerStartIndex[i-1]+centerCount[i-1];
		}
	}
	
	//curCount=new int[kCenter];

	for (int i = 0; i < datasize; i++)
	{
		int tlabel=labels[i];
		int ind=centerStartIndex[tlabel] + curCount[tlabel];
		correspondings[ind]=i;
		++curCount[tlabel];
	}

	


}

__global__ void updateCenters4(int ind,double* dataset,int datasize,int dimension,double* centers,int kCenter,int* corresponding,int* centerStartIndex,int* centerCount)
{
	int j=ind+blockDim.x*blockIdx.x+threadIdx.x;
	if(j<kCenter)
	{
		if(centerCount[j]>0)
		{
			for (int i = 0; i < dimension; i++)
			{
				centers[j*dimension+i]=0;
			}
			for (int i = 0; i < centerCount[j]; i++)
			{
				int curinde=corresponding[ centerStartIndex[j]+i];

				for (int k = 0; k < dimension; k++)
				{
					centers[j*dimension+k]+=dataset[curinde*dimension+k]/centerCount[j];
				}
			}

		}
	}
}

void kmeans4(double* dataset,int datasize,int dimension,double* centers,int* labels,int kCenter,int maxIterationNumber,int threadsize,int blocksize=65535)
{
//	int threadsize=256;
//	int blocksize=256;
	vector<int> initialCenterIndex=shuffledOrder(datasize,kCenter);

	for(int i=0;i<kCenter;++i)
		for(int j=0;j<dimension;++j)
			centers[i*dimension+j]=dataset[initialCenterIndex[i]*dimension+j];


	int parallelNumber=threadsize*blocksize;

	bool* goodCenterFlag;
	int* centerCount;
	int* curCount;
	bool* centerChangeFlag;
	int* corresponding;
	int* centerStartIndex;

	bool* noChange;
	
	cudaError_t cudaStatus;
	cudaMallocManaged(&goodCenterFlag,sizeof(bool)*kCenter);

	cudaMallocManaged(&centerCount,sizeof(int)*kCenter);
	cudaMallocManaged(&curCount,sizeof(int)*kCenter);

	cudaMallocManaged(&centerChangeFlag,sizeof(bool)*datasize);

	cudaMallocManaged(&corresponding,sizeof(bool)*datasize);

	cudaMallocManaged(&centerStartIndex,sizeof(int)*kCenter);

	cudaMallocManaged(&noChange,sizeof(bool));
	
	for(int i=0;i<kCenter;++i) goodCenterFlag[i]=true;

	for(int i=0;i<kCenter;++i) centerCount[i]=0;

	for(int i=0;i<datasize;++i) centerChangeFlag[i]=false;


	for(int iterN=0;iterN<maxIterationNumber;++iterN)
	{
		int remain=datasize;
		while(remain>0)
		{
			//printf("%d ",i);
			int tblocksize=blocksize;
			if(blocksize*threadsize>remain)
			{
				tblocksize=remain/threadsize+(remain%threadsize==0?0:1);
			}

			updatebelonging4<<<tblocksize,threadsize>>>(datasize-remain,dataset,datasize,
				dimension,centers,labels,
				kCenter,goodCenterFlag,
				centerChangeFlag);

			remain-=tblocksize*threadsize;
		}
		printf("belongings ok\n");

		updateCorresponds<<<1,1>>>(labels,datasize,kCenter,corresponding,centerChangeFlag,centerStartIndex,centerCount,curCount,goodCenterFlag,noChange);

		remain=kCenter;

		while(remain>0)
		{
			//printf("%d ",i);
			int tblocksize=blocksize;
			if(blocksize*threadsize>remain)
			{
				tblocksize=remain/threadsize+(remain%threadsize==0?0:1);
			}

			updateCenters4<<<tblocksize,threadsize>>>(kCenter-remain,dataset,datasize,dimension,centers,kCenter,corresponding,centerStartIndex,centerCount);
			remain-=tblocksize*threadsize;
		}

		printf("center ok\n");
		
		printf("finished iteration NO. %d\n",iterN);

		bool hnochange;
		cudaMemcpy(&hnochange,noChange,sizeof(bool),cudaMemcpyDeviceToHost);
		if(hnochange)
			break;

	}

	cudaFree(noChange);
	cudaFree(goodCenterFlag);

	cudaFree(corresponding);
	cudaFree(centerStartIndex);

	cudaFree(centerCount);
	cudaFree(curCount);
	cudaFree(centerChangeFlag);


	cudaDeviceSynchronize();
}




int main()
{
	chdir("/home/pitybea/");

	cudaDeviceProp prop;

	cudaGetDeviceProperties(&prop,0);
	cout<<prop.maxThreadsPerBlock<<endl;


	double* dataset;
	FILE* fp=fopen("fea.txt","r");

	int size,dimension;

	fscanf(fp,"%d %d\n",&size,&dimension);

	cudaMallocManaged(&dataset,sizeof(double)*size*dimension);
	printf("%d %d\n",size,dimension);

	for (int i=0;i<size;i++)
	{
		if(i%100==0) printf("%d\t",i);
		for (int j=0;j<dimension;j++)
		{
			fscanf(fp,"%lf ",&dataset[i*dimension+j]);
		}
		fscanf(fp,"\n");
	}

	fclose(fp);

	printf("%lf",dataset[size*dimension-1]);

	int k=size/1000;
	double* centers;
	int* labels;

	cudaMallocManaged(&centers,sizeof(double)*k*dimension);
	cudaMallocManaged(&labels,sizeof(int)*size);

	for(int i=0;i<k*dimension;++i)
		centers[i]=0;
	for(int i=0;i<size;++i)
		labels[i]=0;

	kmeans4(dataset,size,dimension,centers,labels,k,2000,prop.maxThreadsPerBlock);

	cudaFree(labels);
	cudaFree(centers);
	cudaFree(dataset);
	cudaDeviceReset();
	return 0;
}



