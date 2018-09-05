#include "Octree.h"
#include <algorithm>
#include <fstream>
#include <iostream>


void Octree::build(const int depth, const int full_layer,
	const int npt, const float* bbmin, const float* bbmax,
	const float* pts, const float* normals, const int* labels)
{
	// init
	full_layer_ = full_layer < 1 ? 1 : full_layer;
	depth_ = full_layer < depth ? depth : full_layer;
	set_bbox(bbmin, bbmax);
	children_.clear();
	children_.resize(depth + 1);
	keys_.clear();
	keys_.resize(depth + 1);

	// preprocess, get key and sort
	vector<uint32> node_keys, sorted_idx;
	vector<float> weights;
	processing_points(node_keys, sorted_idx, weights, pts, npt);
 
	// get unique key
	vector<uint32> unique_idx;
	unique_key(node_keys, unique_idx);	

	// build tree
	// build layer 0 to full_layer_depth_
	for (int curr_depth = 0; curr_depth <= full_layer_; curr_depth++)
	{
		vector<int>& children = children_[curr_depth];
		vector<uint32>& keys = keys_[curr_depth];

		int n = 1 << 3 * curr_depth;
		keys.resize(n, -1); children.resize(n, -1);
		for (int i = 0; i < n; i++)
		{
			keys[i] = i;

			if (curr_depth != full_layer_)
			{
				children[i] = i;
			}
		}

	}
	// build layers
	for (int curr_depth = depth_; curr_depth > full_layer_; --curr_depth)
	{
		// compute parent key, i.e. keys of layer (curr_depth -1)
		int n = node_keys.size();
		vector<uint32> parent_keys(n);
		#pragma omp parallel for
		for (int i = 0; i < n; i++)
		{
			parent_keys[i] = node_keys[i] >> 3;
		}

		// compute unique parent key
		vector<uint32> parent_pidx;
		unique_key(parent_keys, parent_pidx);

		// augment children keys and create nodes
		int np = parent_keys.size();
		int nch = np << 3;
		vector<int>& children = children_[curr_depth];
		vector<uint32>& keys = keys_[curr_depth];

		children.resize(nch, -1);
		keys.resize(nch, 0);

		for (int i = 0; i < nch; i++)
		{
			int j = i >> 3;
			keys[i] = (parent_keys[j] << 3) | (i % 8);
		}


		// compute base address for each node
		vector<uint32> addr(nch);
		for (int i = 0; i < np; i++)
		{
			for (uint32 j = parent_pidx[i]; j < parent_pidx[i + 1]; j++)
			{
				addr[j] = i << 3;
			}
		}

		// set children pointer and parent pointer
		#pragma omp parallel for
		for (int i = 0; i < n; i++)
		{
			// address
			uint32 k = (node_keys[i] & 7u) | addr[i];

			// set children pointer for layer curr_depth
			children[k] = i;
		}

		// save data and prepare for the following iteration
		node_keys.swap(parent_keys);
	}
	// set the children for the layer full_layer_depth_
	// Now the node_keys are the key for full_layer
	if (depth_ > full_layer_)
	{
		for (int i = 0; i < node_keys.size(); i++)
		{
			uint32 j = node_keys[i];
			children_[full_layer_][j] = i;
		}
	}

	// splat normal (according to Ps, Ns, pnum, pidx)
 	splat_normal(normals, weights, sorted_idx, unique_idx, children_[depth_]);
	if(labels != nullptr)
		splat_label(labels, weights, sorted_idx, unique_idx, children_[depth_]);
 }
 
bool Octree::save(std::string& filename)
{
	std::ofstream outfile(filename, std::ios::binary);
	if (!outfile) return false;

	vector<int> node_num;
	for (auto& keys : keys_)
	{
		node_num.push_back(keys.size());
	}

	vector<int> node_num_accu(depth_ + 2, 0);
	for (int i = 1; i < depth_ + 2; ++i)
	{
		node_num_accu[i] = node_num_accu[i - 1] + node_num[i - 1];
	}
	int total_node_num = node_num_accu[depth_ + 1];
	int final_node_num = node_num[depth_];

	// calc key
	std::vector<int> key(total_node_num), children(total_node_num);
	int idx = 0;
	for (int d = 0; d <= depth_; ++d)
	{
		vector<uint32>& keys = keys_[d];
		for (int i =0; i < keys.size(); ++i)
		{
			// calc point
			uint32 k = keys[i], pt[3];
			compute_pt(pt, k, d);

			// compress
			unsigned char* ptr = reinterpret_cast<unsigned char*>(&key[idx]);
			ptr[0] = static_cast<unsigned char>(pt[0]);
			ptr[1] = static_cast<unsigned char>(pt[1]);
			ptr[2] = static_cast<unsigned char>(pt[2]);
			ptr[3] = static_cast<unsigned char>(d);

			// children
			children[idx] = children_[d][i];

			// update index
			idx++;
		}
	}

	// write
	outfile.write((char*)&total_node_num, sizeof(int));
	outfile.write((char*)&final_node_num, sizeof(int));
	outfile.write((char*)&depth_, sizeof(int));
	outfile.write((char*)&full_layer_, sizeof(int));
	outfile.write((char*)node_num.data(), sizeof(int)*(depth_ + 1));
	outfile.write((char*)node_num_accu.data(), sizeof(int)*(depth_ + 2));
	outfile.write((char*)key.data(), sizeof(int)*total_node_num);
	outfile.write((char*)children.data(), sizeof(int)*total_node_num);
	outfile.write((char*)data_.data(), sizeof(float) * 3 * final_node_num);
	if (!label_.empty())
	{
		outfile.write((char*)label_.data(), label_.size()*sizeof(int));
	}
	outfile.close();

	return true;
}

void Octree::set_bbox(const float* bbmin, const float* bbmax)
{
	float center[3];
	scale_ = -1.0e20;
	for (int i = 0; i < 3; ++i)
	{
		float dis = bbmax[i] - bbmin[i];
		if (dis > scale_) scale_ = dis;
		center[i] = (bbmin[i] + bbmax[i])*0.5;
	}

	// set the bounding box and place the object in the center
	float radius = scale_ * 0.5;
	for (int i = 0; i < 3; ++i)
	{
		bbmax_[i] = center[i] + radius;
		bbmin_[i] = center[i] - radius;
	}
}

void Octree::processing_points(vector<uint32>& sorted_keys, 
	vector<uint32>& sorted_idx,	vector<float>& weights, 
	const float* pts, const int npt)
{
	sorted_keys.resize(npt);
	sorted_idx.resize(npt);
	weights.resize(npt);
	vector<uint64> code(npt);
	const float mul = float(1 << depth_) / scale_;
	#pragma omp parallel for
	for (int i = 0; i < npt; i++)
	{
		// normalize coordinate
		uint32 pt[3];
		weights[i] = 1.0f;
		for (int j = 0; j < 3; j++)
		{
			float tmp = (pts[i * 3 + j] - bbmin_[j]) * mul;
			pt[j] = tmp;
			weights[i] *= bspline(float(pt[j]) + 0.5f - tmp);
		}

		// calc key
		uint32 key;
		compute_key(key, pt, depth_);

		// generate code
		uint32* ptr = (uint32*)(&code[i]);
		ptr[0] = i;
		ptr[1] = key;
	}

	// sort all sample points
	std::sort(code.begin(), code.end());

	// unpack the code
	#pragma omp parallel for
	for (int i = 0; i < npt; i++)
	{
		uint32* ptr = (uint32*)(&code[i]);
		sorted_idx[i] = ptr[0];
		sorted_keys[i] = ptr[1];
	}
}

void Octree::unique_key(vector<uint32>& keys, vector<uint32>& idx)
{
	// init
	idx.clear();
	idx.push_back(0);

	// unique
	int n = keys.size(), j = 1;
	for (int i = 1; i < n; i++)
	{
		if (keys[i] != keys[i - 1])
		{
			idx.push_back(i);
			keys[j++] = keys[i];
		}
	}

	keys.resize(j);
	idx.push_back(n);
}

inline void Octree::compute_key(uint32& key, const uint32* pt, const int depth)
{
	key = 0;
	for (int i = 0; i < depth; i++)
	{
		uint32 mask = 1u << i;
		for (int j = 0; j < 3; j++)
		{
			key |= (pt[j] & mask) << (2 * i + 2 - j);
		}
	}
}

inline void Octree::compute_pt(uint32* pt, const uint32& key, const int depth)
{
	// init
	for (int i = 0; i < 3; pt[i++] = 0u);

	// convert
	for (int i = 0; i < depth; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			// bit mask
			uint32 mask = 1u << (3 * i + 2 - j); 
			// put the bit to position i
			pt[j] |= (key & mask) >> (2 * i + 2 - j); 
		}
	}
}

inline float Octree::bspline(float x)
{
	if (x < -1.5f)
	{
		return 0;
	}
	else if (x < -0.5f)
	{
		return 0.5f * (x + 1.5f) * (x + 1.5f);
	}
	else if ( x < 0.5f)
	{
		return 0.75f - x * x;
	}
	else if (x <= 1.5f)
	{
		return 0.5f * (x - 1.5f) * (x - 1.5f);
	}
	else
	{
		return 0;
	}
}

void Octree::splat_normal(const float* normals, const vector<float>& weights, 
	const vector<uint32>& sorted_idx, const vector<uint32>& unique_idx,
	const vector<int>& children)
{
	int n = children.size();
	data_.resize(3 * n);
	#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		int t = children[i];
		vector<float> Nw(3, 0.0);
		if (t != -1)
		{
			for (int j = unique_idx[t]; j < unique_idx[t + 1]; j++)
			{
				int h = sorted_idx[j];
				Nw[0] += weights[h] * normals[3 * h];
				Nw[1] += weights[h] * normals[3 * h + 1];
				Nw[2] += weights[h] * normals[3 * h + 2];
			}

			float len = sqrt(Nw[0] * Nw[0] + Nw[1] * Nw[1] + Nw[2] * Nw[2]);
			if (len < ESP) len = ESP;

			Nw[0] /= len;
			Nw[1] /= len;
			Nw[2] /= len;
		}
		
		data_[i] = Nw[0];
		data_[n + i] = Nw[1];
		data_[2 * n + i] = Nw[2];
	}
}

void Octree::splat_label(const int* labels, const vector<float>& weights,
	const vector<uint32>& sorted_idx, const vector<uint32>& unique_idx,
	const vector<int>& children)
{
	int n = children.size();
	int nl = *std::max_element(labels, labels + weights.size()) + 1;

	label_.clear(); 
	label_.resize(n, -1);
	#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		int t = children[i];
		if (t == -1) continue;
		vector<float> Lw(nl, 0);

		for (int j = unique_idx[t]; j < unique_idx[t + 1]; j++)
		{
			int h = sorted_idx[j];
			Lw[labels[h]] += weights[h];
		}

		label_[i] = 0;
		float v = Lw[0];
		for (int j = 1; j < nl; ++j)
		{
			if (Lw[j] > v)
			{
				v = Lw[j];
				label_[i] = j;
			}
		}
	}
}
