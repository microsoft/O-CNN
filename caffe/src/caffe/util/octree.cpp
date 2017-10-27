#include <boost/thread.hpp>
#include <unordered_map>
#include "caffe/util/octree.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
	// Make sure each thread can have different values.
	static boost::thread_specific_ptr<Octree> thread_octree_instance_;

	Octree& Octree::Get()
	{
		if (!thread_octree_instance_.get())
		{
			thread_octree_instance_.reset(new Octree());
		}
		return *(thread_octree_instance_.get());
	}

	void Octree::init_neigh_index()
	{
		// ni for kernel_size=3
		vector<int> shape{ 216 };
		ni_[3].reset(new Blob<int>(shape));
		int* ni3 = ni_[3]->mutable_cpu_data();
		int id = 0;
		for (int i = 0; i < 2; ++i)
		{
			for (int j = 0; j < 2; ++j)
			{
				for (int k = 0; k < 2; ++k)
				{
					for (int x = 0; x < 3; ++x)
					{
						for (int y = 0; y < 3; ++y)
						{
							for (int z = 0; z < 3; ++z)
							{
								ni3[id++] = (x + i << 4) | (y + j << 2) | z + k;
							}
						}
					}
				}
			}
		}

		// ni for kernel_size=2
		shape[0] = 64;
		ni_[2].reset(new Blob<int>(shape));
		int* ni2 = ni_[2]->mutable_cpu_data();
		const int arr[] = { 13, 14, 16, 17, 22, 23, 25, 26 };
		for (int i = 0; i < 8; ++i)
		{
			for (int j = 0; j < 8; ++j)
			{
				ni2[i * 8 + j] = ni3[i * 27 + arr[j]];
			}
		}

		// ni for kernel_size=1
		shape[0] = 8;
		ni_[1].reset(new Blob<int>(shape));
		int* ni1 = ni_[1]->mutable_cpu_data();
		for (int i = 0; i < 8; ++i)
		{
			ni1[i] = ni3[i * 27 + 13];
		}


		// init the array parent & displacement
		id = 0;
		int tmp[64];
		shape[0] = 64;
		displacement_.Reshape(shape);
		int* dis_ptr = displacement_.mutable_cpu_data();
		for (int x = 1; x < 5; ++x)
		{
			for (int y = 1; y < 5; ++y)
			{
				for (int z = 1; z < 5; ++z)
				{
					int x1 = x / 2;
					int xb = x % 2;
					int y1 = y / 2;
					int yb = y % 2;
					int z1 = z / 2;
					int zb = z % 2;

					tmp[id] = x1 * 9 + y1 * 3 + z1;
					dis_ptr[id] = (xb << 2) | (yb << 1) | zb;
					id++;
				}
			}
		}

		shape[0] = 512;
		parent_.Reshape(shape);
		int* parent_ptr = parent_.mutable_cpu_data();
		for (int i = 0; i < 8; ++i)
		{
			for (int j = 0; j < 64; ++j)
			{
				parent_ptr[i * 64 + j] = ni3[i * 27 + tmp[j]];
			}
		}
	}

namespace octree{

	template<typename Dtype>
	void pad_forward_cpu(Dtype* Y, const int Hy,
		const int Cy, const Dtype* X, const int Hx, const int* label)
	{
		// Note: Cx == Cy
		for (int c = 0; c < Cy; ++c)
		{
			for (int h = 0; h < Hy; ++h)
			{
				Y[c*Hy + h] = label[h] == -1 ? Dtype(0) : X[c*Hx + label[h]];
			}
		}
	}
	template<typename Dtype>
	void pad_backward_cpu(Dtype* X, const int Hx,
		const int Cx, const Dtype* Y, const int Hy, const int* label)
	{
		// Note: Cx == Cy
		for (int c = 0; c < Cx; ++c)
		{
			for (int h = 0; h < Hy; ++h)
			{
				if (label[h] != -1)
				{
					X[c*Hx + label[h]] = Y[c*Hy + h];
				}
			}
		}
	}

	template <typename Dtype>
	void octree2col_cpu(Dtype* data_col, const Dtype* data_octree,
		const int channel, const int height, const int kernel_size,
		const int stride, const int* neigh, const int* ni)
	{
		// kernel size: 3*3*3
		const int octree_h = height << 3 * (stride - 1);
		const int kernel = kernel_size*kernel_size*kernel_size;
		for (int c = 0; c < channel; ++c)
		{
			for (int k = 0; k < kernel; ++k)
			{
				for (int h = 0; h < height; ++h)
				{
					const int index = stride == 2 ? (h << 6) + ni[k] :
						(h >> 3 << 6) + ni[(h % 8) * kernel + k];
					const int p = neigh[index];
					data_col[(c*kernel + k)*height + h] = p == -1 ?
						Dtype(0) : data_octree[c*octree_h + p];
				}
			}
		}
	}

	template <typename Dtype>
	void col2octree_cpu(const Dtype* data_col, Dtype* data_octree,
		const int channel, const int height, const int kernel_size,
		const int stride, const int* neigh, const int* ni)
	{
		const int octree_h = height << 3 * (stride - 1);
		const int kernel = kernel_size*kernel_size*kernel_size;
		caffe_set(channel*octree_h, Dtype(0), data_octree);
		for (int c = 0; c < channel; ++c)
		{
			for (int k = 0; k < kernel; ++k)
			{
				for (int h = 0; h < height; ++h)
				{
					const int index = stride == 2 ? (h << 6) + ni[k] :
						(h >> 3 << 6) + ni[(h % 8) * kernel + k];
					const int p = neigh[index];
					if (p != -1) data_octree[c*octree_h + p] +=
						data_col[(c*kernel + k)*height + h];
				}
			}
		}
	}

	void generate_key_cpu(int* key_split, const int* key,
		const int* children, const int node_num)
	{
		typedef unsigned char ubyte;
		for (int i = 0; i < node_num; ++i)
		{
			int label = children[i];
			if (label == -1) continue;
			const ubyte* k0 = (const ubyte*)(key + i);
			for (ubyte j = 0; j < 8; ++j)
			{
				ubyte* k1 = (ubyte*)(key_split + 8 * label + j);
				k1[0] = (k0[0] << 1) | ((j & 4) >> 2);
				k1[1] = (k0[1] << 1) | ((j & 2) >> 1);
				k1[2] = (k0[2] << 1) | (j & 1);
				k1[3] = k0[3] + 1;
			}
		}
	}

	void generate_key_cpu(int* key, const int depth, const int batch_size)
	{
		int node_num = 1 << 3 * depth;
		const int mask = depth << 24;
		for (int n = 0; n < batch_size; ++n)
		{
			for (int i = 0; i < node_num; ++i)
			{
				key[i] = i | mask;
			}
		}
	}

	void calc_neigh_cpu(int* neigh_split, const int* neigh,
		const int* children, const int node_num)
	{
		const int* parent = Octree::get_parent_array().cpu_data();
		const int* dis = Octree::get_dis_array().cpu_data();

		for (int i = 0; i < node_num; ++i)
		{
			int l0 = children[i];
			if (l0 == -1) continue;
			const int* ngh0 = neigh + (i >> 3 << 6);
			const int* pi0 = parent + (i % 8) * 64;
			int* ngh1 = neigh_split + (l0 << 6);
			for (int j = 0; j < 64; ++j)
			{
				ngh1[j] = -1;
				int k = ngh0[pi0[j]];
				if (k != -1)
				{
					int l1 = children[k];
					if (l1 != -1)
					{
						ngh1[j] = (l1 << 3) + dis[j];
					}
				}
			}
		}
	}
	
	void calc_neigh_cpu(int* neigh, const int depth, const int batch_size)
	{
		unsigned node_num = 1 << 3 * depth;
		const unsigned  bound = 1 << depth;
		for (unsigned n = 0; n < batch_size; ++n)
		{
			for (unsigned i = 0; i < node_num; i += 8)
			{
				// key to xyz
				unsigned x0 = 0, y0 = 0, z0 = 0;
				for (unsigned d = 0; d < depth; d++)
				{
					x0 |= (i & (1 << 3 * d + 2)) >> (2 * d + 2);
					y0 |= (i & (1 << 3 * d + 1)) >> (2 * d + 1);
					z0 |= (i & (1 << 3 * d + 0)) >> (2 * d + 0);
				}

				for (unsigned x = 0; x < 4; ++x)
				{
					unsigned x1 = x0 + x - 1;
					if (x1 & bound) continue;
					for (unsigned y = 0; y < 4; ++y)
					{
						unsigned y1 = y0 + y - 1;
						if (y1 & bound) continue;
						for (unsigned z = 0; z < 4; ++z)
						{
							int z1 = z0 + z - 1;
							if (z1 & bound) continue;

							// xyz index
							unsigned xyz = (x << 4) | (y << 2) | z;

							// key
							unsigned key1 = 0;
							for (int d = 0; d < depth; d++)
							{
								unsigned mask = 1u << d;
								key1 |= ((x1 & mask) << (2 * d + 2)) |
									((y1 & mask) << (2 * d + 1)) |
									((z1 & mask) << (2 * d));
							}

							// mapping
							neigh[xyz + i * 8 + n*node_num * 8] = key1 + n*node_num;
						}
					}
				}
			}
		}
	}

	void calc_neighbor(int* neigh, const unsigned* key, const int node_num, 
		const int displacement)
	{
		typedef unsigned char ubyte;

		// build hash table
		vector<std::pair<unsigned, int> > entries(node_num);
		for (int id = 0; id < node_num; ++id)
		{	// ignore the root node
			entries[id] = std::make_pair(key[id], id + displacement);
		}
		std::unordered_map<unsigned, int> hash_table(entries.begin(), entries.end());

		// calc neighborhood
		for (int id = 0; id < node_num; id += 8)
		{
			// the neighborhood volume
			int* ngh = neigh + id * 8;
			const ubyte* k0 = (const ubyte*)(key + id);
			// currently the maximize octree depth is 8
			ubyte k1[4] = { 0, 0, 0, k0[3] };
			const ubyte bound = (1 << k0[3]) - 2;
			for (ubyte x = 0; x < 4; ++x)
			{
				k1[0] = k0[0] + x - 1;
				for (ubyte y = 0; y < 4; ++y)
				{
					k1[1] = k0[1] + y - 1;
					for (ubyte z = 0; z < 4; ++z)
					{
						k1[2] = k0[2] + z - 1;

						// find							
						unsigned* k2 = reinterpret_cast<unsigned*>(k1);
						auto rst = hash_table.find(*k2);
						ubyte i = (x << 4) | (y << 2) | z;
						if (rst != hash_table.end())
						{
							ngh[i] = rst->second;
						}
						else {
							ngh[i] = -1;
						}
					}
				}
			}
		}
	}

	inline void compute_key(int& key, const int* pt, const int depth)
	{
		key = 0;
		for (int i = 0; i < depth; i++)
		{
			int mask = 1u << i;
			for (int j = 0; j < 3; j++)
			{
				key |= (pt[j] & mask) << (2 * i + 2 - j);
			}
		}
	}

	inline void compute_pt(int* pt, const int& key, const int depth)
	{
		// init
		for (int i = 0; i < 3; pt[i++] = 0u);

		// convert
		for (int i = 0; i < depth; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				// bit mask
				int mask = 1u << (3 * i + 2 - j);
				// put the bit to position i
				pt[j] |= (key & mask) >> (2 * i + 2 - j);
			}
		}
	}

} // namespace octree

	OctreeParser::OctreeParser(const void* data)
	{
		metadata_ = data;

		total_node_num_ = (const int*)(metadata_);
		final_node_num_ = total_node_num_ + 1;
		depth_ = final_node_num_ + 1;
		full_layer_ = depth_ + 1;
		node_num_ = full_layer_ + 1;
		node_num_accu_ = node_num_ + (*depth_) + 1;

		key_ = node_num_accu_ + (*depth_) + 2;
		children_ = key_ + (*total_node_num_);

		signal_ = children_ + (*total_node_num_);
		seg_label_ = signal_ + 3 * (*final_node_num_);
	}

	int OctreeParser::node_number_nempty(int depth)
	{
		// point to the last element of the children array
		const int* ptr = children_ + node_num_accu_[depth + 1] - 1;
		// find the last element which is not equal to -1
		while (-1 == *ptr) --ptr;
		return (*ptr) + 1;
	}
	
	void OctreeBatchParser::set_cpu(void* data, int depth,
		int total_nnum, int batch_size, int content_flags)
	{
		metadata_ = (int*)data;

		batch_size_ = metadata_;
		if (0 == batch_size ) batch_size = *batch_size_;
		CHECK(batch_size > 0) << "Invalid octree depth";
		depth_ = batch_size_ + 1;
		full_layer_ = depth_ + 1;
		node_num_ = full_layer_ + 1;
		if (0 == depth ) depth = *depth_;
		CHECK(depth < 10 && depth>0) << "Invalid octree depth";
		node_num_cum_ = node_num_ + depth + 1;
		node_num_nempty_ = node_num_cum_ + depth + 2;
		node_num_oct_ = node_num_nempty_ + depth + 1;
		node_num_nempty_oct_ = node_num_oct_ + (depth + 1)*batch_size;
		if (0 == total_nnum) total_nnum = node_num_cum_[depth + 1];
		CHECK(total_nnum > 0) << "Invalid node number";

		content_flags_ = node_num_nempty_oct_ + (depth + 1)*batch_size;
		if (0 == content_flags ) content_flags = *content_flags_;
		CHECK(content_flags != 0) << "Invalid flags";
		int* ptr = content_flags_ + 1;
		key_ = children_ = neigh_ = nullptr;
		if (0 != (content_flags & 1))
		{
			key_ = ptr;
			ptr += total_nnum;
		}
		if (0 != (content_flags & 2))
		{
			children_ = ptr;
			ptr += total_nnum;
		}
		if (0 != (content_flags & 4))
		{
			neigh_ = ptr;
		}
	}

	int OctreeBatchParser::header_sizeofint(const int depth, const int batch_size)
	{
		return 3 + (depth + 1) + (depth + 2) + (depth + 1)
			+ 2 * (depth + 1)*batch_size + 1;
	}

	int OctreeBatchParser::octree_batch_sizeofint(const int depth, const int total_nnum,
		const int batch_size, int content_flags)
	{
		int sz = header_sizeofint(depth, batch_size);	// header size
		if (0 != (content_flags & 1)) sz += total_nnum; // key array size
		if (0 != (content_flags & 2)) sz += total_nnum; // children array size		
		//CHECK_EQ(total_nnum % 8, 1);		  // NOTE: only for 3*3*3 neighborhood
		if (0 != (content_flags & 4)) sz += total_nnum * AVG_NGH_NUM;
		return sz;
	}

	template<typename Dtype>
	void OctreeBatch<Dtype>::set_octreebatch(const vector<vector<char> >& octree_buffer,
		const vector<int>& label_buffer, const bool segmentation /* = false */)
	{
		/// octree parser
		int content_flags = 7;	// key & children & neighborhood
		int batch_size = octree_buffer.size();
		vector<OctreeParser> octree_parsers;
		for (int i = 0; i < batch_size; ++i)
		{
			octree_parsers.push_back(OctreeParser(octree_buffer[i].data()));
		}

		/// get the node number information
		// get depth and full_layer information
		int depth = *octree_parsers[0].depth_;
		int full_layer = *octree_parsers[0].full_layer_;
		for (int i = 1; i < batch_size; ++i)
		{
			CHECK_EQ(depth, *octree_parsers[i].depth_);
			CHECK_EQ(full_layer, *octree_parsers[i].full_layer_);
		}

		// node and non-empty node number in each octree
		int sz = (depth + 1)*batch_size;
		vector<int> nnum(sz), nnum_nempty(sz);
		for (int i = 0; i < batch_size; ++i)
		{
			for (int d = 0; d < depth + 1; ++d)
			{
				int p = i*(depth + 1) + d;
				nnum[p] = octree_parsers[i].node_num_[d];
				nnum_nempty[p] = octree_parsers[i].node_number_nempty(d);
			}
		}

		// cumulative node and non-empty node number in each layers
		sz = (depth + 1)*(batch_size + 1);
		vector<int> nnum_cum_layer(sz), nnum_cum_nempty_layer(sz);
		for (int d = 0; d < depth + 1; ++d)
		{
			nnum_cum_layer[d] = 0;
			nnum_cum_nempty_layer[d] = 0;
			for (int i = 0; i < batch_size; ++i)
			{
				int p = i*(depth + 1) + d;
				int q = p + depth + 1;
				nnum_cum_layer[q] = nnum[p] + nnum_cum_layer[p];
				nnum_cum_nempty_layer[q] = nnum_nempty[p] + nnum_cum_nempty_layer[p];
			}
		}

		// cumulative node number for each octree
		sz = (depth + 1)*batch_size;
		vector<int> nnum_cum_octree(sz);
		for (int i = 0; i < batch_size; ++i)
		{
			nnum_cum_octree[i*(depth + 1)] = 0;
			for (int d = 0; d < depth; ++d)
			{
				int p = i*(depth + 1) + d;
				nnum_cum_octree[p + 1] = nnum_cum_octree[p] + nnum[p];
			}
		}

		// node and non-empty node number of the batch
		vector<int> nnum_batch(depth + 1), nnum_batch_nempty(depth + 1);
		for (int d = 0; d < depth + 1; ++d)
		{
			int p = batch_size*(depth + 1) + d;
			nnum_batch[d] = nnum_cum_layer[p];
			nnum_batch_nempty[d] = nnum_cum_nempty_layer[p];
		}

		// cumulative node number of the batch
		vector<int> nnum_batch_cum(depth + 2);
		nnum_batch_cum[0] = 0;
		for (int d = 0; d < depth + 1; ++d)
		{
			nnum_batch_cum[d + 1] = nnum_batch_cum[d] + nnum_batch[d];
		}

		/// init space
		// octree_
		int total_nnum = nnum_batch_cum[depth + 1];
		vector<int> octree_shape{ OctreeBatchParser::octree_batch_sizeofint(depth,
			total_nnum, batch_size, content_flags) };
		octree_.Reshape(octree_shape);
		int* octree_ptr = octree_.mutable_cpu_data();
		OctreeBatchParser octbatch_parser;
		octbatch_parser.set_cpu(octree_ptr, depth, total_nnum, batch_size, content_flags);

		// data_
		int deepest_nnum = nnum_batch[depth]; // node number in the deepest layer
		vector<int> data_shape(4, 1);
		data_shape[1] = 3;
		data_shape[2] = deepest_nnum;
		data_.Reshape(data_shape);
		Dtype* data_ptr = data_.mutable_cpu_data();

		// label_
		vector<int> label_shape;
		if (!segmentation)	
		{
			label_shape = { batch_size };
		}
		else 
		{
			// dense label, i.e. there is label for each node in the deepest layer
			label_shape = { 1, data_shape[2] };
		}
		label_.Reshape(label_shape);
		Dtype* label_ptr = label_.mutable_cpu_data();

		/// set data
		// set octree header info
		*octbatch_parser.batch_size_ = batch_size;
		*octbatch_parser.depth_ = depth;
		*octbatch_parser.full_layer_ = full_layer;
		memcpy(octbatch_parser.node_num_, nnum_batch.data(),
			sizeof(int)*nnum_batch.size());
		memcpy(octbatch_parser.node_num_cum_, nnum_batch_cum.data(),
			sizeof(int)*nnum_batch_cum.size());
		memcpy(octbatch_parser.node_num_nempty_, nnum_batch_nempty.data(),
			sizeof(int)*nnum_batch_nempty.size());
		memcpy(octbatch_parser.node_num_oct_, nnum.data(),
			sizeof(int)*nnum.size());
		memcpy(octbatch_parser.node_num_nempty_oct_, nnum_nempty.data(),
			sizeof(int)*nnum_nempty.size());
		*octbatch_parser.content_flags_ = content_flags;

		// Here the lambda and boost::thread are used.
		// If the OpenMP is available, the code can be more concise.
		//omp_set_num_threads(4);
		//#pragma omp parallel for
		//for (int i = 0; i < batch_size; ++i){...}
		auto worker = [&octree_parsers, &octbatch_parser, &data_ptr, &label_ptr,
			&nnum, &nnum_cum_layer, &nnum_batch_cum, &nnum_cum_nempty_layer,
			&nnum_batch, &nnum_cum_octree, &batch_size, &depth, &segmentation, 
			&label_buffer] (int thread_id, int thread_num)
		{
			for (int i = thread_id; i < batch_size; i += thread_num)
			{
				// copy key - TODO: optimize!
				for (int d = 0; d < depth + 1; ++d)
				{
					int p = i*(depth + 1) + d;
					int* des = octbatch_parser.key_ + nnum_cum_layer[p] + nnum_batch_cum[d];
					const int* src = octree_parsers[i].key_ + nnum_cum_octree[p];
					for (int j = 0; j < nnum[p]; ++j)
					{
						des[j] = src[j];
						unsigned char* ptr = (unsigned char*)(des + j);
						ptr[3] = i;
					}
				}

				// copy children
				for (int d = 0; d < depth + 1; ++d)
				{
					int p = i*(depth + 1) + d;
					int* des = octbatch_parser.children_ + nnum_cum_layer[p] + nnum_batch_cum[d];
					const int* src = octree_parsers[i].children_ + nnum_cum_octree[p];
					for (int j = 0; j < nnum[p]; ++j)
					{
						des[j] = -1 == src[j] ? src[j] : src[j] + nnum_cum_nempty_layer[p];
					}
				}

				// copy data
				int p = i*(depth + 1) + depth;
				for (int c = 0; c < 3; c++)
				{
					Dtype* des = data_ptr + c*nnum_batch[depth] + nnum_cum_layer[p];
					const Dtype* src = (const Dtype*)octree_parsers[i].signal_ + c* nnum[p];
					memcpy(des, src, nnum[p] * sizeof(Dtype));
				}


				// copy label
				if (segmentation)
				{
					for (int j = 0; j < nnum[p]; ++j)
					{
						label_ptr[nnum_cum_layer[p] + j] = (Dtype)octree_parsers[i].seg_label_[j];
					}
				}
				else
				{
					label_ptr[i] = label_buffer[i];
				}

				// calc and set neighbor info
				for (int d = 1; d < depth + 1; ++d)
				{
					int p = i*(depth + 1) + d;
					const unsigned* key = (const unsigned*)octree_parsers[i].key_ + nnum_cum_octree[p];
					int* neigh = octbatch_parser.neigh_ + OctreeBatchParser::AVG_NGH_NUM *
						(nnum_cum_layer[p] + nnum_batch_cum[d]);
					octree::calc_neighbor(neigh, key, nnum[p], nnum_cum_layer[p]);
				}
			}
		};

		const int thread_num = 8;
		vector<shared_ptr<boost::thread> > workers(thread_num);
		for (int id = 1; id < thread_num; ++id)
		{
			workers[id].reset(new boost::thread(worker, id, thread_num));
		}
		// for the master thread
		worker(0, thread_num); 

		for (int id = 1; id < thread_num; ++id)
		{
			workers[id]->join();
		}
	}

	// Explicit instantiation
	template void octree::pad_forward_cpu<float>(float* Y, const int Hy,
		const int Cy, const float* X, const int Hx, const int* label);
	template void octree::pad_forward_cpu<double>(double* Y, const int Hy,
		const int Cy, const double* X, const int Hx, const int* label);
	template void octree::pad_backward_cpu<float>(float* X, const int Hx,
		const int Cx, const float* Y, const int Hy, const int* label);
	template void octree::pad_backward_cpu<double>(double* X, const int Hx,
		const int Cx, const double* Y, const int Hy, const int* label);
	template void octree::octree2col_cpu<float>(float* data_col, 
		const float* data_octree, const int channel, const int height,
		const int kernel_size, const int stride, const int* neigh, const int* ni);
	template void octree::octree2col_cpu<double>(double* data_col,
		const double* data_octree, const int channel, const int height,
		const int kernel_size, const int stride, const int* neigh, const int* ni);
	template void octree::col2octree_cpu<float>(const float* data_col,
		float* data_octree, const int channel, const int height,
		const int kernel_size, const int stride, const int* neigh, const int* ni);
	template void octree::col2octree_cpu<double>(const double* data_col,
		double* data_octree, const int channel, const int height,
		const int kernel_size, const int stride, const int* neigh, const int* ni);
	INSTANTIATE_CLASS(OctreeBatch);
}