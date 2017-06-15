#include <cuda.h>
#include "caffe/util/octree.hpp"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe{
namespace octree{
	template <typename Dtype>
	__global__ void pad_forward_kernel(Dtype* Y, const int Hy,
		const Dtype* X, const int Hx, const int* label, const int n)
	{
		CUDA_KERNEL_LOOP(i, n)
		{
			int h = i % Hy;
			int c = i / Hy;

			int idx = label[h];
			Y[i] = idx == -1 ? Dtype(0) : X[c*Hx + idx];
		}
	}

	template <typename Dtype>
	__global__ void pad_backward_kernel(Dtype* X, const int Hx,
		const Dtype* Y, const int Hy, const int* label, const int n)
	{
		CUDA_KERNEL_LOOP(i, n)
		{
			int h = i % Hy;
			int c = i / Hy;

			int idx = label[h];
			if (idx != -1)
			{
				X[c*Hx + idx] = Y[i];
			}
		}
	}

	template<typename Dtype>
	void pad_forward_gpu(Dtype* Y, const int Hy,
		const int Cy, const Dtype* X, const int Hx, const int* label)
	{
		int n = Hy*Cy; // Note: Cx == Cy
		pad_forward_kernel<Dtype> << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(
			Y, Hy, X, Hx, label, n);
		CUDA_POST_KERNEL_CHECK;
	}


	template<typename Dtype>
	void pad_backward_gpu(Dtype* X, const int Hx,
		const int Cx, const Dtype* Y, const int Hy, const int* label)
	{
		int n = Hy*Cx; // Note: Cx == Cy
		pad_backward_kernel<Dtype> << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(
			X, Hx, Y, Hy, label, n);
		CUDA_POST_KERNEL_CHECK;
	}
	
	template <typename Dtype>
	__global__ void octree2col_kernel(Dtype* data_col, const Dtype* data_octree,
		const int height, const int kernel_dim, const int stride, const int* neigh, 
		const int* ni, const int thread_num)
	{
		CUDA_KERNEL_LOOP(i, thread_num)
		{
			int h = i % height;
			int t = i / height;
			int k = t % kernel_dim;
			int c = t / kernel_dim;
			int octree_h = height << 3 * (stride - 1);

			int index = stride == 2 ? (h << 6) + ni[k] :
				(h >> 3 << 6) + ni[(h % 8) * kernel_dim + k];
			int p = neigh[index];
			data_col[i] = p == -1 ? Dtype(0) : data_octree[c*octree_h + p];
		}
	}

	template <typename Dtype>
	__global__ void col2octree_kernel(const Dtype* data_col, Dtype* data_octree,
		const int height, const int kernel_dim, const int stride, const int* neigh, 
		const int* ni, const int thread_num)
	{
		CUDA_KERNEL_LOOP(i, thread_num)
		{
			int h = i % height;
			int t = i / height;
			int k = t % kernel_dim;
			int c = t / kernel_dim;
			int octree_h = height << 3 * (stride - 1);

			int index = stride == 2 ? (h << 6) + ni[k] :
				(h >> 3 << 6) + ni[(h % 8) * kernel_dim + k];
			int p = neigh[index];
			if (p != -1) caffe_gpu_atomic_add(data_col[i], data_octree + c*octree_h + p);
		}
	}


	template <typename Dtype>
	void octree2col_gpu(Dtype* data_col, const Dtype* data_octree,
		const int channel, const int height, const int kernel_size,
		const int stride, const int* neigh, const int* ni)
	{
		const int kernel = kernel_size*kernel_size*kernel_size;
		const int thread_num = channel*kernel*height;
		octree2col_kernel<Dtype> <<<CAFFE_GET_BLOCKS(thread_num), CAFFE_CUDA_NUM_THREADS >>>(
			data_col, data_octree, height, kernel, stride, neigh, ni, thread_num);
		CUDA_POST_KERNEL_CHECK;
	}


	template <typename Dtype>
	void col2octree_gpu(const Dtype* data_col, Dtype* data_octree,
		const int channel, const int height, const int kernel_size,
		const int stride, const int* neigh, const int* ni)
	{
		const int kernel = kernel_size*kernel_size*kernel_size; // kernel size: 3*3*3
		const int thread_num = channel*kernel*height;
		int data_h = height << 3 * (stride - 1);
		caffe_gpu_set(channel*data_h, Dtype(0), data_octree);
		col2octree_kernel<Dtype> <<<CAFFE_GET_BLOCKS(thread_num), CAFFE_CUDA_NUM_THREADS >>>(
			data_col, data_octree, height, kernel, stride, neigh, ni, thread_num);
		CUDA_POST_KERNEL_CHECK;

	}

	__global__ void gen_key_kernel(int* key_split, const int* key,
		const int* children, const int thread_num)
	{
		typedef unsigned char ubyte;

		CUDA_KERNEL_LOOP(id, thread_num)
		{
			int i = id >> 3;
			int j = id % 8;

			int label = children[i];
			if (label != -1)
			{
				const ubyte* k0 = (const ubyte*)(key + i);
				ubyte* k1 = (ubyte*)(key_split + 8 * label + j);
				k1[0] = (k0[0] << 1) | ((j & 4) >> 2);
				k1[1] = (k0[1] << 1) | ((j & 2) >> 1);
				k1[2] = (k0[2] << 1) | (j & 1);
				k1[3] = k0[3];
			}
		}
	}

	__global__ void gen_full_key_kernel(int* key, const int depth,
		const int batch_size, const int thread_num)
	{
		CUDA_KERNEL_LOOP(i, thread_num)
		{
			unsigned node_num = 1 << 3 * depth;
			unsigned k = i % node_num;
			unsigned xyz = 0;
			unsigned char* ptr = (unsigned char*)(&xyz);
			#pragma unroll 3
			for (int d = 0; d < depth; d++)
			{
				ptr[0] |= (k & (1 << 3 * d + 2)) >> (2 * d + 2);
				ptr[1] |= (k & (1 << 3 * d + 1)) >> (2 * d + 1);
				ptr[2] |= (k & (1 << 3 * d + 0)) >> (2 * d + 0);
			}
			ptr[3] = i / node_num;
			key[i] = xyz;
		}
	}

	__global__ void calc_neigh_kernel(int* neigh_split, const int* neigh,
		const int* children, const int* parent, const int* dis, const int thread_num)
	{
		CUDA_KERNEL_LOOP(id, thread_num)
		{
			int i = id >> 6;
			int j = id % 64;

			int l0 = children[i];
			if (l0 != -1)
			{
				const int* ngh0 = neigh + (i >> 3 << 6);
				const int* pi0 = parent + (i % 8) * 64;
				int* ngh1 = neigh_split + (l0 << 6);
				int t = -1;
				int k = ngh0[pi0[j]];
				if (k != -1)
				{
					int l1 = children[k];
					if (l1 != -1)
					{
						t = (l1 << 3) + dis[j];
					}
				}
				ngh1[j] = t;
			}
		}
	}

	__global__ void calc_full_neigh_kernel(int* neigh, const int depth,
		const int batch_size, const int thread_num)
	{
		CUDA_KERNEL_LOOP(id, thread_num)
		{
			const unsigned  bound = 1 << depth;
			unsigned node_num = 1 << 3 * depth;
			unsigned num = node_num >> 3;

			unsigned tm = id;
			unsigned z = tm % 4; tm /= 4;
			unsigned y = tm % 4; tm /= 4;
			unsigned x = tm % 4; tm /= 4;
			unsigned i = (tm % num) * 8;
			unsigned n = tm / num;

			unsigned x0 = 0, y0 = 0, z0 = 0;
			#pragma unroll 4
			for (unsigned d = 0; d < depth; d++)
			{
				x0 |= (i & (1 << 3 * d + 2)) >> (2 * d + 2);
				y0 |= (i & (1 << 3 * d + 1)) >> (2 * d + 1);
				z0 |= (i & (1 << 3 * d + 0)) >> (2 * d + 0);
			}

			unsigned x1 = x0 + x - 1;
			unsigned y1 = y0 + y - 1;
			unsigned z1 = z0 + z - 1;

			int v = -1;
			if ((x1 & bound) == 0 &&
				(y1 & bound) == 0 &&
				(z1 & bound) == 0)
			{
				unsigned key1 = 0;
#pragma unroll 4
				for (int d = 0; d < depth; d++)
				{
					unsigned mask = 1u << d;
					key1 |= ((x1 & mask) << (2 * d + 2)) |
						((y1 & mask) << (2 * d + 1)) |
						((z1 & mask) << (2 * d));
				}
				v = key1 + n*node_num;
			}

			neigh[id] = v;
		}
	}

	void generate_key_gpu(int* key_split, const int* key,
		const int* children, const int node_num)
	{
		// use the information from parent layer to calculate the neigh_split of current layer
		int n = node_num << 3; // node_num: the node number of parent layer
		gen_key_kernel <<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >>> (
			key_split, key, children, n);
	}

	void generate_key_gpu(int* key, const int depth, const int batch_size)
	{
		int thread_num = batch_size * (1 << 3 * depth);
		gen_full_key_kernel <<<CAFFE_GET_BLOCKS(thread_num), CAFFE_CUDA_NUM_THREADS >>> (
			key, depth, batch_size, thread_num);
	}

	void calc_neigh_gpu(int* neigh_split, const int* neigh,
		const int* children, const int node_num)
	{
		// use the information from parent layer to calculate the neigh_split of current layer
		const int* parent = Octree::get_parent_array().gpu_data();
		const int* dis = Octree::get_dis_array().gpu_data();

		int n = node_num << 6; // node_num: the non_empty node number of parent layer
		calc_neigh_kernel <<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >>> (
			neigh_split, neigh, children, parent, dis, n);
	}

	void calc_neigh_gpu(int* neigh, const int depth, const int batch_size)
	{
		int thread_num = batch_size * (1 << 3 * depth + 3);
		calc_full_neigh_kernel <<<CAFFE_GET_BLOCKS(thread_num), CAFFE_CUDA_NUM_THREADS >>> (
			neigh, depth, batch_size, thread_num);
	}

	// Explicit instantiation
	template void pad_forward_gpu<float>(float* Y, const int Hy,
		const int Cy, const float* X, const int Hx, const int* label);
	template void pad_forward_gpu<double>(double* Y, const int Hy,
		const int Cy, const double* X, const int Hx, const int* label);
	template void pad_backward_gpu<float>(float* X, const int Hx, 
		const int Cx, const float* Y, const int Hy, const int* label);
	template void pad_backward_gpu<double>(double* X, const int Hx, 
		const int Cx, const double* Y, const int Hy, const int* label);
	template void octree2col_gpu<float>(float* data_col,
		const float* data_octree, const int channel, const int height,
		const int kernel_size, const int stride, const int* neigh, const int* ni);
	template void octree2col_gpu<double>(double* data_col,
		const double* data_octree, const int channel, const int height,
		const int kernel_size, const int stride, const int* neigh, const int* ni);
	template void col2octree_gpu<float>(const float* data_col,
		float* data_octree, const int channel, const int height,
		const int kernel_size, const int stride, const int* neigh, const int* ni);
	template void col2octree_gpu<double>(const double* data_col,
		double* data_octree, const int channel, const int height,
		const int kernel_size, const int stride, const int* neigh, const int* ni);

} // namespace octree
} // namespace caffe