#pragma once
#include <vector>
#include <string>
using std::vector;
using std::string;

class Octree
{
public:
	typedef unsigned long long uint64;
	typedef unsigned int uint32;

public:
	void build(const int depth, const int depth_full, const int npt,
		const float* bbmin, const float* bbmax, const float* pts,
		const float* normals, const int* labels = nullptr);

	bool save(string& filename);

protected:
	void set_bbox(const float* bbmin, const float* bbmax);

	void processing_points(vector<uint32>& keys, vector<uint32>& idx,
		vector<float>& weights, const float* pts, const int npt);
	
	// compute the key for the sepcified point
	inline void compute_key(uint32& key, const uint32* pt, const int depth);
	
	// compute the point coordinate given the key
	inline void compute_pt(uint32* pt, const uint32& key, const int depth);

	inline float bspline(float x);

	void unique_key(vector<uint32>& node_key, vector<uint32>& pidx);

	void splat_normal(const float* normals, const vector<float>& weights,
		const vector<uint32>& sorted_idx, const vector<uint32>& unique_idx, 
		const vector<int>& children);

	void splat_label(const int* labels, const vector<float>& weights,
		const vector<uint32>& sorted_idx, const vector<uint32>& unique_idx, 
		const vector<int>& children);

protected:
	// node array
	vector<vector<uint32>> keys_;
	vector<vector<int>> children_;
	// data array, a nx3 matrix
	vector<float> data_;
	// label array
	vector<int> label_;

	// tree depth
	int depth_;
	int full_layer_;

	// bounding box
	float bbmax_[3];
	float bbmin_[3];
	float scale_;

	// const
	const float C_ = 1.05f;
	const float ESP = 1.0e-20f;
};

