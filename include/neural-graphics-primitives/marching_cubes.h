/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   marching_cubes.h
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#pragma once

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/common_host.h>

#include <tiny-cuda-nn/common.h>

#include <queue>

namespace ngp {

ivec3 get_marching_cubes_res(uint32_t res_1d, const BoundingBox& render_aabb);

void marching_cubes_gpu(cudaStream_t stream, BoundingBox render_aabb, mat3 render_aabb_to_local, ivec3 res_3d, float thresh, const GPUMemory<float>& density, GPUMemory<vec3>& vert_out, GPUMemory<uint32_t>& indices_out);

// computes the average of the 1ring of all verts, as homogenous coordinates
void compute_mesh_1ring(const GPUMemory<vec3>& verts, const GPUMemory<uint32_t>& indices, GPUMemory<vec4>& output_pos, GPUMemory<vec3>& output_normals);

void compute_mesh_opt_gradients(
	float thresh,
	const GPUMemory<vec3>& verts,
	const GPUMemory<vec3>& vert_normals,
	const GPUMemory<vec4>& verts_smoothed,
	const network_precision_t* densities,
	uint32_t input_gradient_width,
	const float* input_gradients,
	GPUMemory<vec3>& verts_gradient_out,
	float k_smooth_amount,
	float k_density_amount,
	float k_inflate_amount
);

void save_mesh(
	GPUMemory<vec3>& verts,
	GPUMemory<vec3>& normals,
	GPUMemory<vec3>& colors,
	GPUMemory<uint32_t>& indices,
	const fs::path& path,
	bool unwrap_it,
	float nerf_scale,
	vec3 nerf_offset
);

#ifdef NGP_GUI
void draw_mesh_gl(
	const GPUMemory<vec3>& verts,
	const GPUMemory<vec3>& normals,
	const GPUMemory<vec3>& cols,
	const GPUMemory<uint32_t>& indices,
	const ivec2& resolution,
	const vec2& focal_length,
	const mat4x3& camera_matrix,
	const vec2& screen_center,
	int mesh_render_mode
);

void glCheckError(const char* file, unsigned int line);
uint32_t compile_shader(bool pixel, const char* code);
bool check_shader(uint32_t handle, const char* desc, bool program);
#endif

void save_density_grid_to_png(const GPUMemory<float>& density, const fs::path& path, ivec3 res3d, float thresh, bool swap_y_z = true, float density_range = 4.f);
void save_rgba_grid_to_png_sequence(const GPUMemory<vec4>& rgba, const fs::path& path, ivec3 res3d, bool swap_y_z = true);
void save_rgba_grid_to_raw_file(const GPUMemory<vec4>& rgba, const fs::path& path, ivec3 res3d, bool swap_y_z, int cascade);

vec3 vertex_interp(float iso, vec3 p1, vec3 p2, float d1, float d2);

struct MCMesh {
	std::vector<vec3> vertices;
	std::vector<unsigned int> indices;
};

// Extend with DFS:
// - sample leaf subspace
// - construct subtree bottom-up
// - depending on condition and depth: prune, extend or set min-density and exit
class DensityOctree {
public:
	static std::array<vec3, 8> s_corner_offsets;
	static std::array<std::array<int, 2>, 12> s_edge_connections;
	static float s_min_density;

	struct Node;
	using ExtendibleQueue = std::queue<std::pair<std::reference_wrapper<DensityOctree::Node>, int>>;

  	struct Node {
		vec3 origin;
		float size;

		float density;
		std::unique_ptr<std::array<Node, 8>> children;

		Node();
		Node(vec3 origin, float size);

		bool is_leaf() const;
		bool is_empty() const;

		void extend(std::vector<float>& vertices, int res, int depth, int max_tree_height, float min_density, ExtendibleQueue& extendibles);
		void traverse(const std::function<void(const Node&)>& fun) const;
		float sample_density(vec3 pos) const;
		void polygonize(MCMesh& mesh, float iso) const;

	private:
		void extend(std::vector<float>& vertices, int res, int curr_res, ivec3 pos, int depth, int max_tree_height, float min_density, ExtendibleQueue& extendibles);
  	};

	static DensityOctree init(vec3 origin, float size);

	DensityOctree::Node& get_root();
	void traverse(const std::function<void(const Node&)>& fun) const;
	float sample_density(vec3 pos) const;

	MCMesh polygonize(float iso) const;

private:
  	Node m_root;

	DensityOctree(Node&& root);
};

}
