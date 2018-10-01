#include "nms_cuda.h"

__device__ int get_index(){
  int blockId=blockIdx.y*gridDim.x+blockIdx.x;
  int threadId=blockId*blockDim.x+threadIdx.x;
  return threadId;
}

__device__ int get_block_prefix(){
    int blockId=blockIdx.y*gridDim.x+blockIdx.x;
    int block_prefix=blockId*blockDim.x;
    return block_prefix;
}

__device__ int get_thread(){
  return threadIdx.x;
}

__global__ void nms_cuda_imp(long* bbox, int64_t* bbox_size,
   float* mask,int64_t* mask_size,float thresh){
     int index=get_index();
     int block_prefix=get_block_prefix();
     for(int i=0;i<bbox_size[1];++i){
       __syncthreads();
       if(index>=bbox_size[0]*bbox_size[1]){
         continue;
       }
       if(i>=get_thread()){
         continue;
       }
       if(mask[block_prefix+i]==0){
         continue;
       }
       if(mask[index]==0){
         continue;
       }
       long x11=bbox[4*(block_prefix)+0];
       long y11=bbox[4*(block_prefix)+1];
       long x12=bbox[4*(block_prefix)+2];
       long y12=bbox[4*(block_prefix)+3];
       long x21=bbox[4*(index)+0];
       long y21=bbox[4*(index)+1];
       long x22=bbox[4*(index)+2];
       long y22=bbox[4*(index)+3];
       int areas_u=(x12-x11)*(y12-y11)+(x22-x21)*(y22-y21);
       int max_x1=((x11>=x21)?x11:x21);
       int max_y1=((y11>=y21)?y11:y21);
       int min_x2=((x12>=x22)?x12:x22);
       int min_y2=((y12>=y22)?y12:y22);
       int w=min_x2-max_x1;
       w=(w>=0?w:0);
       int h=min_y2-max_y1;
       h=(h>=0?h:0);
       int areas_n=w*h;
       if(areas_u-areas_n==0){
         continue;
       }
       if(float(areas_n)/float(areas_u-areas_n)>thresh){
         mask[index]=0;
       }
     }

   }

void cuda_cpy(int64_t* from,int64_t** to,int size){
  cudaMalloc((void**)to,size*sizeof(int64_t));
  cudaMemcpy(*to,from,size*sizeof(int64_t),cudaMemcpyHostToDevice);
}

void nms_cuda(long* bbox, int64_t* bbox_size,
   float* mask,int64_t* mask_size,float thresh,cudaStream_t stream){
  int d1=1;
  int d2=1;
  if(bbox_size[0]>512){
    d2=(bbox_size[0]+511)/512;
    d1=512;
  }else{
    d1=bbox_size[0];
    d2=1;
  }
  dim3 batch(d1,d2,1);
  dim3 thread(bbox_size[1],1,1);
  int64_t* bbox_size_cuda;
  int64_t* mask_size_cuda;
  cuda_cpy(bbox_size,&bbox_size_cuda,3);
  cuda_cpy(mask_size,&mask_size_cuda,2);
  nms_cuda_imp<<<batch,thread,0,stream>>>(bbox,bbox_size_cuda,mask,mask_size_cuda,thresh);

  cudaFree(bbox_size_cuda);
  cudaFree(mask_size_cuda);
}
