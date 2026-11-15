"""Single-block deterministic Euclidean FPS, FP64 distances and ID-order ties."""
from experiments import stream_patch_data as d
import os
import torch
CPP='''#include <torch/extension.h>
torch::Tensor fps_cuda(torch::Tensor x, torch::Tensor anchors, int64_t k);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("fps",&fps_cuda);}
'''
CU=r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <ATen/cuda/CUDAContext.h>
__global__ void go(const float* x,const float* a,double* dist,long* out,int n,int na,int k){
 __shared__ double dd[256];__shared__ int ii[256];__shared__ int chosen;
 int tid=threadIdx.x;
 for(int j=tid;j<n;j+=256){double best=DBL_MAX;
  for(int b=0;b<na;b++){double z=0;for(int c=0;c<3;c++){double v=(double)x[j*3+c]-a[b*3+c];z+=v*v;}best=fmin(best,z);}dist[j]=best;}
 __syncthreads();
 for(int step=0;step<k;step++){
  double best=-2;int idx=2147483647;
  for(int j=tid;j<n;j+=256){double v=dist[j];if(v>best||(v==best&&j<idx)){best=v;idx=j;}}
  dd[tid]=best;ii[tid]=idx;__syncthreads();
  for(int gap=128;gap;gap/=2){if(tid<gap){double v=dd[tid+gap];int j=ii[tid+gap];if(v>dd[tid]||(v==dd[tid]&&j<ii[tid])){dd[tid]=v;ii[tid]=j;}}__syncthreads();}
  if(tid==0){chosen=ii[0];out[step]=chosen;}__syncthreads();
  for(int j=tid;j<n;j+=256){if(j==chosen){dist[j]=-1;continue;}if(dist[j]<0)continue;double z=0;for(int c=0;c<3;c++){double v=(double)x[j*3+c]-x[chosen*3+c];z+=v*v;}dist[j]=fmin(dist[j],z);}__syncthreads();
 }
}
torch::Tensor fps_cuda(torch::Tensor x,torch::Tensor a,int64_t k){
 TORCH_CHECK(x.is_cuda()&&a.is_cuda()&&x.scalar_type()==torch::kFloat32&&a.scalar_type()==torch::kFloat32&&x.is_contiguous()&&a.is_contiguous(),"FP32 contiguous CUDA centers");
 TORCH_CHECK(k<=x.size(0),"k<=n");auto out=torch::empty({k},x.options().dtype(torch::kInt64));auto dist=torch::empty({x.size(0)},x.options().dtype(torch::kFloat64));
 go<<<1,256,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),a.data_ptr<float>(),dist.data_ptr<double>(),out.data_ptr<long>(),x.size(0),a.size(0),k);return out;
}
'''
_cached=None
def module():
 global _cached
 if _cached is None:
  os.environ['TORCH_EXTENSIONS_DIR']=str(d.DATA/'extensions');os.environ['MAX_JOBS']='2'
  from torch.utils.cpp_extension import load_inline
  _cached=load_inline('stream_patch_fps1',cpp_sources=CPP,cuda_sources=CU,extra_cuda_cflags=['-O3'],verbose=False)
 return _cached
def fps(x,a,k):return module().fps(x.contiguous(),a.contiguous(),k)
