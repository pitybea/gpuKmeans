
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
//#include <unistd.h>
#include <vector>
#include <iostream>
using namespace std;

#ifndef __CUDACC__  
    #define __CUDACC__
#endif

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
	//chdir("/home/pitybea/");

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

	kmeans2(dataset,size,dimension,centers,labels,k,2000);

	cudaFree(labels);
	cudaFree(centers);
	cudaFree(dataset);
	cudaDeviceReset();
	return 0;
}



