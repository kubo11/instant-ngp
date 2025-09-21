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
#include <neural-graphics-primitives/vertex_cache.h>

#include <tiny-cuda-nn/common.h>

#include <queue>
#include <optional>
#include <vector>
#include <functional>

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
	CPUHash64 mcEdgeCache;
	CPUHash64 faceEdgeCache;

	vec3   base_origin = {0,0,0};
    float  base_h      = 1.0f; 

    uint32_t add_vertex(const vec3& p) {
        uint32_t id = (uint32_t)vertices.size();
        vertices.push_back(p);
        return id;
    }
};

enum class Face {PX, NX, PY, NY, PZ, NZ};

// Extend with DFS:
// - sample leaf subspace
// - construct subtree bottom-up
// - depending on condition and depth: prune, extend or set min-density and exit
class DensityOctree {
public:
	static std::array<vec3, 8> s_corner_offsets;
	static std::array<std::array<int, 2>, 12> s_edge_connections;
	static std::array<int, 8> s_corner_mapping;
	static float s_min_density;

  // Node types: corner, leaf, intermediate
  struct Node {
		vec3 origin;
		float size;

		float density;
		std::unique_ptr<std::array<Node, 8>> children;

		Node* parent;
		int level;
		int child_idx;

		Node();
		Node(vec3 origin, float size);
		Node(vec3 origin, float size, float density, int level, Node* parent, int child_idx);

		bool is_corner() const;
		bool is_leaf() const;
		bool is_intermediate() const;

		void extend(int res, int depth, int max_tree_height, float min_density, const std::function<std::vector<float>(const vec3& origin, float size)>& get_density_on_grid);
		void traverse(const std::function<void(Node&)>& fun);
		float sample_density(vec3 pos);
		void polygonize(MCMesh& mesh, float iso, const std::function<float(vec3 pos)>& sample_density);
		void polygonize_marching_cubes(MCMesh& mesh, float iso);
		void polygonize_transvoxel(MCMesh& mesh, float iso, const std::function<float(vec3 pos)>& sample_density);

		void emit_transition(MCMesh& mesh, float iso, const std::function<float(vec3 pos)>& sd, Node& nb, Face f);
		std::vector<std::pair<Node&, Face>> get_neighors_faces();
		Node* get_face_neighbor(Face f);
		bool intersects_band(float band, float iso);
  };

	static DensityOctree init(vec3 origin, float size);
	void build(int res, float min_density, int max_tree_height, const std::function<std::vector<float>(const vec3& origin, float size)>& get_density_on_grid);

	DensityOctree::Node& get_root();
	void traverse(const std::function<void(Node&)>& fun);
	float sample_density(vec3 pos);
	float sample_nearest(int x, int y, int z);
	float sample_trilinear(float nx, float ny, float nz);

	MCMesh polygonize(float iso, int max_tree_height);
	void fill_transvoxel_data();

private:
  	Node m_root;

	DensityOctree(Node&& root);
};

}
