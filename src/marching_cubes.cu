/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   marchingcubes.cu
 *  @author Alex Evans, NVIDIA
 */

#include <neural-graphics-primitives/marching_cubes.h>

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/common_host.h>
#include <neural-graphics-primitives/random_val.cuh> // helpers to generate random values, directions
#include <neural-graphics-primitives/thread_pool.h>
#include <neural-graphics-primitives/transvoxel.h>

#include <tiny-cuda-nn/gpu_memory.h>
#include <filesystem/path.h>

#include <stdarg.h>

#include <limits>

#ifdef NGP_GUI
#  ifdef _WIN32
#    include <GL/gl3w.h>
#  else
#    include <GL/glew.h>
#  endif
#  include <GLFW/glfw3.h>
#  include <cuda_gl_interop.h>
#endif

#include <vector>

namespace ngp {

ivec3 get_marching_cubes_res(uint32_t res_1d, const BoundingBox &aabb) {
	float scale = res_1d / max(aabb.max - aabb.min);
	ivec3 res3d = (aabb.max - aabb.min) * scale + 0.5f;
	res3d.x = next_multiple((unsigned int)res3d.x, 16u);
	res3d.y = next_multiple((unsigned int)res3d.y, 16u);
	res3d.z = next_multiple((unsigned int)res3d.z, 16u);
	return res3d;
}

#ifdef NGP_GUI
void glCheckError(const char* file, unsigned int line) {
	GLenum errorCode = glGetError();
	while (errorCode != GL_NO_ERROR) {
		std::string fileString(file);
		std::string error = "unknown error";
		// clang-format off
		switch (errorCode) {
			case GL_INVALID_ENUM:      error = "GL_INVALID_ENUM"; break;
			case GL_INVALID_VALUE:     error = "GL_INVALID_VALUE"; break;
			case GL_INVALID_OPERATION: error = "GL_INVALID_OPERATION"; break;
			case GL_STACK_OVERFLOW:    error = "GL_STACK_OVERFLOW"; break;
			case GL_STACK_UNDERFLOW:   error = "GL_STACK_UNDERFLOW"; break;
			case GL_OUT_OF_MEMORY:     error = "GL_OUT_OF_MEMORY"; break;
		}
		// clang-format on

		tlog::error() << "OpenglError : file=" << file << " line=" << line << " error:" << error;
		errorCode = glGetError();
	}
}

bool check_shader(uint32_t handle, const char* desc, bool program) {
	GLint status = 0, log_length = 0;
	if (program) {
		glGetProgramiv(handle, GL_LINK_STATUS, &status);
		glGetProgramiv(handle, GL_INFO_LOG_LENGTH, &log_length);
	} else {
		glGetShaderiv(handle, GL_COMPILE_STATUS, &status);
		glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &log_length);
	}
	if ((GLboolean)status == GL_FALSE) {
		tlog::error() << "Failed to compile shader: " << desc;
	}
	if (log_length > 1) {
		std::vector<char> log; log.resize(log_length+1);
		if (program) {
			glGetProgramInfoLog(handle, log_length, NULL, (GLchar*)log.data());
		} else {
			glGetShaderInfoLog(handle, log_length, NULL, (GLchar*)log.data());
		}
		log.back() = 0;
		tlog::error() << log.data();
	}
	return (GLboolean)status == GL_TRUE;
}

uint32_t compile_shader(bool pixel, const char* code) {
	GLuint g_VertHandle = glCreateShader(pixel ? GL_FRAGMENT_SHADER : GL_VERTEX_SHADER );
	const char* glsl_version = "#version 140\n";
	const GLchar* strings[2] = { glsl_version, code};
	glShaderSource(g_VertHandle, 2, strings, NULL);
	glCompileShader(g_VertHandle);
	if (!check_shader(g_VertHandle, pixel? "pixel" : "vertex", false)) {
		glDeleteShader(g_VertHandle);
		return 0;
	}
	return g_VertHandle;
}

void draw_mesh_gl(
	const GPUMemory<vec3>& verts,
	const GPUMemory<vec3>& normals,
	const GPUMemory<vec3>& colors,
	const GPUMemory<uint32_t>& indices,
	const ivec2& resolution,
	const vec2& focal_length,
	const mat4x3& camera_matrix,
	const vec2& screen_center,
	int mesh_render_mode
) {
	if (verts.size() == 0 || indices.size() == 0 || mesh_render_mode == 0) {
		return;
	}

	static GLuint vs = 0, ps = 0, program = 0, VAO = 0, VBO[3] = {}, els = 0, vbosize = 0, elssize = 0;
	if (!VAO) {
		glGenVertexArrays(1, &VAO);
		glBindVertexArray(VAO);
	}
	if (vbosize != verts.size()) {
		for (int i= 0; i < 3; ++i) {
			if (VBO[i]) {
				cudaGLUnregisterBufferObject(VBO[i]);
				glDeleteBuffers(1, &VBO[i]);
			}
			glGenBuffers(1, &VBO[i]);
			vbosize = verts.size();
			glBindBuffer(GL_ARRAY_BUFFER, VBO[i]);
			glBufferData(GL_ARRAY_BUFFER, vbosize * sizeof(vec3), NULL, GL_DYNAMIC_COPY);
			cudaGLRegisterBufferObject(VBO[i]);
		}
	}
	if (elssize != indices.size()) {
		if (els) {
			cudaGLUnregisterBufferObject(els);
			glDeleteBuffers(1, &els);
		}
		glGenBuffers(1, &els);
		elssize = indices.size();
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, els);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, elssize * sizeof(int), NULL, GL_STREAM_DRAW);
		cudaGLRegisterBufferObject(els);
	}
	void *ptr=nullptr;
	cudaGLMapBufferObject(&ptr, VBO[0]);
	if (ptr) cudaMemcpy(ptr, verts.data(), vbosize * sizeof(vec3), cudaMemcpyDeviceToDevice);
	cudaGLMapBufferObject(&ptr, VBO[1]);
	if (ptr) cudaMemcpy(ptr, normals.data(), vbosize * sizeof(vec3), cudaMemcpyDeviceToDevice);
	cudaGLMapBufferObject(&ptr, VBO[2]);
	if (ptr) cudaMemcpy(ptr, colors.data(), vbosize * sizeof(vec3), cudaMemcpyDeviceToDevice);

	//std::vector<vec3> cpucols; cpucols.resize(verts.size());
	//colors.copy_to_host(cpucols);

	cudaGLUnmapBufferObject(VBO[2]);
	cudaGLUnmapBufferObject(VBO[1]);
	cudaGLUnmapBufferObject(VBO[0]);
	cudaGLMapBufferObject(&ptr, els);
	if (ptr) cudaMemcpy(ptr, indices.data(), indices.get_bytes(), cudaMemcpyDeviceToDevice);
	cudaGLUnmapBufferObject(els);

	if (!program) {
		vs = compile_shader(false, R"foo(
in vec3 pos;
in vec3 nor;
in vec3 col;
out vec3 vtxcol;
uniform mat4 camera;
uniform vec2 f;
uniform ivec2 res;
uniform vec2 cen;
uniform int mode;
void main()
{
	vec4 p = camera * vec4(pos, 1.0);
	p.xy *= vec2(2.0, -2.0) * f.xy / vec2(res.xy);
	p.w = p.z;
	p.z = p.z - 0.1;
	p.xy += cen * p.w;
	if (mode == 2) {
		vtxcol = normalize(nor) * 0.5 + vec3(0.5); // visualize vertex normals
	} else {
		vtxcol = col;
	}
	gl_Position = p;
}
)foo");
		ps = compile_shader(true, R"foo(
out vec4 o;
in vec3 vtxcol;
uniform int mode;
void main() {
	o = vec4(vtxcol, 1.0);
}
)foo");
		program = glCreateProgram();
		glAttachShader(program, vs);
		glAttachShader(program, ps);
		glLinkProgram(program);
		if (!check_shader(program, "shader program", true)) {
			glDeleteProgram(program);
			program = 0;
		}
	}
	mat4 view2world = camera_matrix;
	mat4 world2view = inverse(view2world);
	glBindVertexArray(VAO);
	glUseProgram(program);
	glUniformMatrix4fv(glGetUniformLocation(program, "camera"), 1, GL_FALSE, (GLfloat*)&world2view);
	glUniform2f(glGetUniformLocation(program, "f"), focal_length.x, focal_length.y);
	glUniform2f(glGetUniformLocation(program, "cen"), screen_center.x*2.f-1.f, screen_center.y*-2.f+1.f);
	glUniform2i(glGetUniformLocation(program, "res"), resolution.x, resolution.y);
	glUniform1i(glGetUniformLocation(program, "mode"), mesh_render_mode);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, els);
	GLuint posat = (GLuint)glGetAttribLocation(program, "pos");
	GLuint norat = (GLuint)glGetAttribLocation(program, "nor");
	GLuint colat = (GLuint)glGetAttribLocation(program, "col");
	glEnableVertexAttribArray(posat);
	glEnableVertexAttribArray(norat);
	glEnableVertexAttribArray(colat);
	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glVertexAttribPointer(posat, 3, GL_FLOAT, GL_FALSE, 3*4, 0);
	glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
	glVertexAttribPointer(norat, 3, GL_FLOAT, GL_FALSE, 3*4, 0);
	glBindBuffer(GL_ARRAY_BUFFER, VBO[2]);
	glVertexAttribPointer(colat, 3, GL_FLOAT, GL_FALSE, 3*4, 0);
	glCullFace(GL_BACK);
	glDisable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glDrawElements(GL_TRIANGLES, (GLsizei)indices.size(), GL_UNSIGNED_INT , (GLvoid*)0);
	glDisable(GL_CULL_FACE);

	glUseProgram(0);
}
#endif //NGP_GUI

/*
vertex indices with z=0
0 -> 1   <-- first edge is 0->1, then 1->2 etc
^    |
|	 v
3 <- 2

with z=1
4 -> 5   <-- fourth edge is 4->5 etc
^    |
|	 v
7 <- 6

edges 8-11 go in +z direction from vertex 0-3
*/
__global__ void gen_vertices(BoundingBox render_aabb, mat3 render_aabb_to_local, ivec3 res_3d, const float* __restrict__ density, int*__restrict__ vertidx_grid, vec3* verts_out, float thresh, uint32_t* __restrict__ counters) {
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
	if (x>=res_3d.x || y>=res_3d.y || z>=res_3d.z) return;
	vec3 scale = (render_aabb.max - render_aabb.min) / vec3(res_3d - 1);
	vec3 offset=render_aabb.min;
	uint32_t res2=res_3d.x*res_3d.y;
	uint32_t res3=res_3d.x*res_3d.y*res_3d.z;
	uint32_t idx=x+y*res_3d.x+z*res2;
	float f0 = density[idx];
	bool inside=(f0>thresh);
	if (x<res_3d.x-1) {
		float f1 = density[idx+1];
		if (inside != (f1>thresh)) {
			uint32_t vidx = atomicAdd(counters,1);
			if (verts_out) {
				vertidx_grid[idx]=vidx+1;
				float prevf=f0,nextf=f1;
				float dt=((thresh-prevf)/(nextf-prevf));
				verts_out[vidx] = transpose(render_aabb_to_local) * (vec3{float(x)+dt, float(y), float(z)} * scale + offset);
			}
		}
	}
	if (y<res_3d.y-1) {
		float f1 = density[idx+res_3d.x];
		if (inside != (f1>thresh)) {
			uint32_t vidx = atomicAdd(counters,1);
			if (verts_out) {
				vertidx_grid[idx+res3]=vidx+1;
				float prevf=f0,nextf=f1;
				float dt=((thresh-prevf)/(nextf-prevf));
				verts_out[vidx] = transpose(render_aabb_to_local) * (vec3{float(x), float(y)+dt, float(z)} * scale + offset);
			}
		}
	}
	if (z<res_3d.z-1) {
		float f1 = density[idx+res2];
		if (inside != (f1>thresh)) {
			uint32_t vidx = atomicAdd(counters,1);
			if (verts_out) {
				vertidx_grid[idx+res3*2]=vidx+1;
				float prevf=f0,nextf=f1;
				float dt=((thresh-prevf)/(nextf-prevf));
				verts_out[vidx] = transpose(render_aabb_to_local) * (vec3{float(x), float(y), float(z)+dt} * scale + offset);
			}
		}
	}
}

__global__ void accumulate_1ring(uint32_t num_tris, const uint32_t* indices, const vec3* verts_in, vec4* verts_out, vec3 *normals_out) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i>=num_tris) return;
	uint32_t ia=indices[i*3+0];
	uint32_t ib=indices[i*3+1];
	uint32_t ic=indices[i*3+2];
	vec3 pa=verts_in[ia];
	vec3 pb=verts_in[ib];
	vec3 pc=verts_in[ic];

	atomicAdd(&verts_out[ia][0], pb.x+pc.x);
	atomicAdd(&verts_out[ia][1], pb.y+pc.y);
	atomicAdd(&verts_out[ia][2], pb.z+pc.z);
	atomicAdd(&verts_out[ia][3], 2.f);
	atomicAdd(&verts_out[ib][0], pa.x+pc.x);
	atomicAdd(&verts_out[ib][1], pa.y+pc.y);
	atomicAdd(&verts_out[ib][2], pa.z+pc.z);
	atomicAdd(&verts_out[ib][3], 2.f);
	atomicAdd(&verts_out[ic][0], pb.x+pa.x);
	atomicAdd(&verts_out[ic][1], pb.y+pa.y);
	atomicAdd(&verts_out[ic][2], pb.z+pa.z);
	atomicAdd(&verts_out[ic][3], 2.f);

	if (normals_out) {
		vec3 n= cross(pb - pa, pa - pc); // don't normalise so it's weighted by area
		atomicAdd(&normals_out[ia][0], n.x);
		atomicAdd(&normals_out[ia][1], n.y);
		atomicAdd(&normals_out[ia][2], n.z);
		atomicAdd(&normals_out[ib][0], n.x);
		atomicAdd(&normals_out[ib][1], n.y);
		atomicAdd(&normals_out[ib][2], n.z);
		atomicAdd(&normals_out[ic][0], n.x);
		atomicAdd(&normals_out[ic][1], n.y);
		atomicAdd(&normals_out[ic][2], n.z);
	}
}

__global__ void compute_centroids(uint32_t num_verts, vec3* centroids_out, const vec4* verts_in) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i>=num_verts) return;
	vec4 p = verts_in[i];
	if (p.w<=0.f) return;
	vec3 c = verts_in[i].xyz() * (1.f / p.w);
	centroids_out[i]=c;
}

__global__ void gen_faces(ivec3 res_3d, const float* __restrict__ density, const int*__restrict__ vertidx_grid, uint32_t* indices_out, float thresh, uint32_t *__restrict__ counters) {
	// marching cubes tables from https://github.com/pmneila/PyMCubes/blob/master/mcubes/src/marchingcubes.cpp which in turn seems to be from https://web.archive.org/web/20181127124338/http://paulbourke.net/geometry/polygonise/
	// License is BSD 3-clause, which can be found here: https://github.com/pmneila/PyMCubes/blob/master/LICENSE
	/*
	static constexpr uint16_t edge_table[256] =
	{
		0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
		0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
		0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
		0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
		0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
		0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
		0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
		0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
		0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
		0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
		0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
		0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
		0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
		0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
		0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
		0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
	};
	*/
	static constexpr int8_t triangle_table[256][16] =
	{
		{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
		{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
		{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
		{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
		{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
		{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
		{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
		{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
		{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
		{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
		{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
		{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
		{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
		{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
		{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
		{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
		{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
		{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
		{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
		{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
		{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
		{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
		{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
		{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
		{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
		{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
		{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
		{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
		{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
		{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
		{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
		{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
		{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
		{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
		{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
		{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
		{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
		{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
		{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
		{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
		{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
		{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
		{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
		{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
		{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
		{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
		{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
		{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
		{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
		{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
		{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
		{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
		{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
		{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
		{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
		{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
		{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
		{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
		{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
		{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
		{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
		{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
		{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
		{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
		{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
		{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
		{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
		{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
		{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
		{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
		{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
		{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
		{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
		{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
		{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
		{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
		{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
		{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
		{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
		{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
		{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
		{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
		{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
		{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
		{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
		{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
		{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
		{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
		{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
		{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
		{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
		{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
		{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
		{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
		{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
		{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
		{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
		{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
		{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
		{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
		{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
		{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
		{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
		{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
		{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
		{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
		{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
		{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
		{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
		{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
		{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
		{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
		{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
		{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
		{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
		{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
		{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
		{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
		{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
		{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
		{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
		{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
		{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
		{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
		{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
		{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
		{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
		{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
		{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
		{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
		{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
		{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
		{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
		{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
		{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
		{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
		{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
		{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
		{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
		{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
		{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
		{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
		{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
		{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
		{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
		{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
		{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
		{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
		{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
		{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
		{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
		{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
		{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
		{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
		{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
		{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
		{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
		{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
		{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
		{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
		{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
		{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
		{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
		{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
		{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
		{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
		{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
		{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
		{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
	};

	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
	if (x>=res_3d.x-1 || y>=res_3d.y-1 || z>=res_3d.z-1) return;
	uint32_t res1=res_3d.x;
	uint32_t res2=res_3d.x*res_3d.y;
	uint32_t res3=res_3d.x*res_3d.y*res_3d.z;
	uint32_t idx=x+y*res_3d.x+z*res2;

	uint32_t idx_x=idx;
	uint32_t idx_y=idx+res3;
	uint32_t idx_z=idx+res3*2;

	int mask=0;
	if (density[idx]>thresh) mask|=1;
	if (density[idx+1]>thresh) mask|=2;
	if (density[idx+1+res1]>thresh) mask|=4;
	if (density[idx+res1]>thresh) mask|=8;
	idx+=res2;
	if (density[idx]>thresh) mask|=16;
	if (density[idx+1]>thresh) mask|=32;
	if (density[idx+1+res1]>thresh) mask|=64;
	if (density[idx+res1]>thresh) mask|=128;
	idx-=res2;

	if (!mask || mask==255) return;
	int local_edges[12];
	if (vertidx_grid) {
		local_edges[0]=vertidx_grid[idx_x];
		local_edges[1]=vertidx_grid[idx_y+1];
		local_edges[2]=vertidx_grid[idx_x+res1];
		local_edges[3]=vertidx_grid[idx_y];

		local_edges[4]=vertidx_grid[idx_x+res2];
		local_edges[5]=vertidx_grid[idx_y+1+res2];
		local_edges[6]=vertidx_grid[idx_x+res1+res2];
		local_edges[7]=vertidx_grid[idx_y+res2];

		local_edges[8]=vertidx_grid[idx_z];
		local_edges[9]=vertidx_grid[idx_z+1];
		local_edges[10]=vertidx_grid[idx_z+1+res1];
		local_edges[11]=vertidx_grid[idx_z+res1];
	}
	uint32_t tricount=0;
	const int8_t *triangles=triangle_table[mask];
	for (;tricount<15;tricount+=3) if (triangles[tricount]<0) break;
	uint32_t tidx = atomicAdd(counters+1,tricount);
	if (indices_out) {
		for (int i=0;i<15;++i) {
			int j = triangles[i];
			if (j<0) break;
			if (!local_edges[j]) {
				printf("at %d %d %d, mask is %d, j is %d, local_edges is 0\n", x,y,z,mask,j);
			}
			indices_out[tidx+i]=local_edges[j]-1;
		}
	}
}

void compute_mesh_1ring(const GPUMemory<vec3> &verts, const GPUMemory<uint32_t> &indices, GPUMemory<vec4> &output_pos, GPUMemory<vec3> &output_normals) { // computes the average of the 1ring of all verts, as homogenous coordinates
	output_pos.resize(verts.size());
	output_pos.memset(0);
	output_normals.resize(verts.size());
	output_normals.memset(0);
	linear_kernel(accumulate_1ring, 0, nullptr, indices.size()/3, indices.data(), verts.data(), output_pos.data(), output_normals.data());
}

__global__ void compute_mesh_opt_gradients_kernel(
	uint32_t n_verts,
	float thresh,
	const vec3* verts,
	const vec3* normals,
	const vec4* verts_smoothed,
	const network_precision_t* densities,
	uint32_t input_gradient_width,
	const float* input_gradients,
	vec3* verts_gradient_out,
	float k_smooth_amount,
	float k_density_amount,
	float k_inflate_amount
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_verts) return;

	vec3 src = verts[i];
	vec4 p = verts_smoothed[i];

	if (p.w <= 0.f) {
		p.w = 1.f;
	}

	vec3 target = p.xyz() * (1.0f / p.w);
	vec3 smoothing_grad = src - target; // negative...

	vec3 input_gradient = *(const vec3 *)(input_gradients + i * input_gradient_width);

	vec3 n = normalize(input_gradient);
	float density = densities[i];
	verts_gradient_out[i] = n * sign(density - thresh) * k_density_amount + smoothing_grad * k_smooth_amount - normalize(normals[i]) * k_inflate_amount;
}

void compute_mesh_opt_gradients(
	float thresh,
	const GPUMemory<vec3>& verts,
	const GPUMemory<vec3>& normals,
	const GPUMemory<vec4>& verts_smoothed,
	const network_precision_t* densities,
	uint32_t input_gradients_width,
	const float* input_gradients,
	GPUMemory<vec3>& verts_gradient_out,
	float k_smooth_amount,
	float k_density_amount,
	float k_inflate_amount
) {
	linear_kernel(
		compute_mesh_opt_gradients_kernel,
		0,
		nullptr,
		verts.size(),
		thresh,
		verts.data(),
		normals.data(),
		verts_smoothed.data(),
		densities,
		input_gradients_width,
		input_gradients,
		verts_gradient_out.data(),
		k_smooth_amount,
		k_density_amount,
		k_inflate_amount
	);
}

void marching_cubes_gpu(cudaStream_t stream, BoundingBox render_aabb, mat3 render_aabb_to_local, ivec3 res_3d, float thresh, const GPUMemory<float>& density, GPUMemory<vec3>& verts_out, GPUMemory<uint32_t>& indices_out) {
	GPUMemory<uint32_t> counters;

	counters.enlarge(4);
	counters.memset(0);

	size_t n_bytes = res_3d.x * (size_t)res_3d.y * res_3d.z * 3 * sizeof(int);
	auto workspace = allocate_workspace(stream, n_bytes);
	CUDA_CHECK_THROW(cudaMemsetAsync(workspace.data(), -1, n_bytes, stream));

	int* vertex_grid = (int*)workspace.data();

	const dim3 threads = { 4, 4, 4 };
	const dim3 blocks = { div_round_up((uint32_t)res_3d.x, threads.x), div_round_up((uint32_t)res_3d.y, threads.y), div_round_up((uint32_t)res_3d.z, threads.z) };
	// count only
	gen_vertices<<<blocks, threads, 0>>>(render_aabb, render_aabb_to_local, res_3d, density.data(), nullptr, nullptr, thresh, counters.data());
	gen_faces<<<blocks, threads, 0>>>(res_3d, density.data(), nullptr, nullptr, thresh, counters.data());
	std::vector<uint32_t> cpucounters; cpucounters.resize(4);
	counters.copy_to_host(cpucounters);
	tlog::info() << "#vertices=" << cpucounters[0] << " #triangles=" << (cpucounters[1]/3);

	uint32_t n_verts = next_multiple(cpucounters[0], BATCH_SIZE_GRANULARITY); // round for later nn stuff
	verts_out.resize(n_verts);
	verts_out.memset(0);
	indices_out.resize(cpucounters[1]);

	// actually generate verts
	gen_vertices<<<blocks, threads, 0>>>(render_aabb, render_aabb_to_local, res_3d, density.data(), vertex_grid, verts_out.data(), thresh, counters.data()+2);
	gen_faces<<<blocks, threads, 0>>>(res_3d, density.data(), vertex_grid, indices_out.data(), thresh, counters.data()+2);
}

void save_mesh(
	GPUMemory<vec3>& verts,
	GPUMemory<vec3>& normals,
	GPUMemory<vec3>& colors,
	GPUMemory<uint32_t>& indices,
	const fs::path& path,
	bool unwrap_it,
	float nerf_scale,
	vec3 nerf_offset
) {
	std::vector<vec3> cpuverts; cpuverts.resize(verts.size());
	std::vector<vec3> cpunormals; cpunormals.resize(normals.size());
	std::vector<vec3> cpucolors; cpucolors.resize(colors.size());
	std::vector<uint32_t> cpuindices; cpuindices.resize(indices.size());
	verts.copy_to_host(cpuverts);
	normals.copy_to_host(cpunormals);
	colors.copy_to_host(cpucolors);
	indices.copy_to_host(cpuindices);

	// Replace invalid values with reasonable defaults
	for (size_t i = 0; i < cpuverts.size(); ++i) {
		if (!all(isfinite(cpuverts[i]))) cpuverts[i] = vec3(0.0f);
		if (!all(isfinite(cpunormals[i]))) cpunormals[i] = vec3{0.0f, 1.0f, 0.0f};
		if (!all(isfinite(cpucolors[i]))) cpucolors[i] = vec3(0.0f);
	}

	uint32_t numquads = ((cpuindices.size()/3)+1)/2;
	uint32_t numquadsx = uint32_t(sqrtf(numquads)+4) & (~3);
	uint32_t numquadsy = (numquads+numquadsx-1)/numquadsx;
	uint32_t quadresy = 8;
	uint32_t quadresx = quadresy+3;
	uint32_t texw = quadresx*numquadsx;
	uint32_t texh = quadresy*numquadsy;

	if (unwrap_it) {
		uint8_t* tex = (uint8_t*)malloc(texw*texh*3);
		for (uint32_t y = 0; y < texh; ++y) {
			for (uint32_t x = 0; x < texw; ++x) {
				uint32_t q = (x/quadresx)+(y/quadresy)*numquadsx;
				// 0 x x 3 - - 4
				// | .\x x\. . |
				// | . .\x x\. |
				// 2 - - 1 x x 5
				uint32_t xi = x % quadresx, yi = y % quadresy;
				uint32_t t = q*2 + (xi>yi+1);
				int r = (t*923)&255;
				int g = (t*3572)&255;
				int b = (t*5423)&255;
				//if (xi==yi+1 || xi==yi+2)
				//	r=g=b=0;
				tex[x*3+y*3*texw+0]=r;
				tex[x*3+y*3*texw+1]=g;
				tex[x*3+y*3*texw+2]=b;
			}
		}

		write_stbi(path.with_extension(".tga"), texw, texh, 3, tex);
		free(tex);
	}

	FILE* f = native_fopen(path, "wb");
	if (!f) {
		throw std::runtime_error{fmt::format("Failed to open '{}' for writing", path.str())};
	}

	if (equals_case_insensitive(path.extension(), "ply")) {
		// ply file
		fprintf(f,
			"ply\n"
			"format ascii 1.0\n"
			"comment output from https://github.com/NVlabs/instant-ngp\n"
			"element vertex %u\n"
			"property float x\n"
			"property float y\n"
			"property float z\n"
			"property float nx\n"
			"property float ny\n"
			"property float nz\n"
			"property uchar red\n"
			"property uchar green\n"
			"property uchar blue\n"
			"element face %u\n"
			"property list uchar int vertex_index\n"
			"end_header\n"
			, (unsigned int)cpuverts.size()
			, (unsigned int)cpuindices.size()/3
		);

		for (size_t i=0;i<cpuverts.size();++i) {
			vec3 p = (cpuverts[i]-nerf_offset)/nerf_scale;
			vec3 c = cpucolors[i];
			vec3 n = normalize(cpunormals[i]);
			unsigned char c8[3] = {(unsigned char)clamp(c.x*255.f,0.f,255.f),(unsigned char)clamp(c.y*255.f,0.f,255.f),(unsigned char)clamp(c.z*255.f,0.f,255.f)};
			fprintf(f, "%0.5f %0.5f %0.5f %0.3f %0.3f %0.3f %d %d %d\n", p.x, p.y, p.z, n.x, n.y, n.z, c8[0], c8[1], c8[2]);
		}

		for (size_t i = 0; i < cpuindices.size(); i += 3) {
			fprintf(f, "3 %d %d %d\n", cpuindices[i+2], cpuindices[i+1], cpuindices[i+0]);
		}
	} else {
		// obj file
		if (unwrap_it) {
			fprintf(f, "mtllib nerf.mtl\n");
		}

		for (size_t i = 0; i < cpuverts.size(); ++i) {
			vec3 p = (cpuverts[i]-nerf_offset)/nerf_scale;
			vec3 c = cpucolors[i];
			fprintf(f, "v %0.5f %0.5f %0.5f %0.3f %0.3f %0.3f\n", p.x, p.y, p.z, clamp(c.x, 0.f, 1.f), clamp(c.y, 0.f, 1.f), clamp(c.z, 0.f, 1.f));
		}

		for (auto &v: cpunormals) {
			auto n = normalize(v);
			fprintf(f, "vn %0.5f %0.5f %0.5f\n", n.x, n.y, n.z);
		}

		if (unwrap_it) {
			for (size_t i = 0; i < cpuindices.size(); i++) {
				uint32_t q = (uint32_t)(i/6);
				uint32_t x = (q%numquadsx)*quadresx;
				uint32_t y = (q/numquadsx)*quadresy;
				uint32_t d = quadresy-1;
				switch (i % 6) {
					case 0: break;
					case 1: x += d; y += d; break;
					case 2: y += d; break;
					case 3: x += 3; break;
					case 4: x += 3+d; break;
					case 5: x += 3+d; y += d; break;
				}
				fprintf(f, "vt %0.5f %0.5f\n", ((float)x+0.5f)/float(texw), 1.f-((float)y+0.5f)/float(texh));
			}

			fprintf(f, "g default\nusemtl nerf\ns 1\n");
			for (size_t i = 0; i < cpuindices.size(); i += 3) {
				fprintf(f,"f %u/%u/%u %u/%u/%u %u/%u/%u\n",
					cpuindices[i+2]+1,(uint32_t)i+3,  cpuindices[i+2]+1,
					cpuindices[i+1]+1,(uint32_t)i+2,cpuindices[i+1]+1,
					cpuindices[i+0]+1,(uint32_t)i+1,cpuindices[i+0]+1
				);
			}
		} else {
			for (size_t i = 0; i < cpuindices.size(); i += 3) {
				fprintf(f, "f %u//%u %u//%u %u//%u\n",
					cpuindices[i+2]+1, cpuindices[i+2]+1, cpuindices[i+1]+1, cpuindices[i+1]+1, cpuindices[i+0]+1, cpuindices[i+0]+1
				);
			}
		}
	}
	fclose(f);
}

void save_density_grid_to_png(const GPUMemory<float>& density, const fs::path& path, ivec3 res3d, float thresh, bool swap_y_z, float density_range) {
	float density_scale = 128.f / density_range; // map from -density_range to density_range into 0-255
	std::vector<float> density_cpu;
	density_cpu.resize(density.size());
	density.copy_to_host(density_cpu);
	uint32_t num_voxels = 0;
	uint32_t num_lattice_points_near_zero_crossing = 0;
	uint32_t N = res3d.x*res3d.y*res3d.z;
	for (int z = 1; z < res3d.z - 1; ++z) {
		for (int y = 1; y < res3d.y - 1; ++y) {
			for (int x = 1; x < res3d.x - 1; ++x) {
				int count = 0;
				count += density_cpu[(x+0)+(y+0)*res3d.x+(z+0)*res3d.x*res3d.y] < thresh;
				count += density_cpu[(x+1)+(y+0)*res3d.x+(z+0)*res3d.x*res3d.y] < thresh;
				count += density_cpu[(x+0)+(y+1)*res3d.x+(z+0)*res3d.x*res3d.y] < thresh;
				count += density_cpu[(x+1)+(y+1)*res3d.x+(z+0)*res3d.x*res3d.y] < thresh;
				count += density_cpu[(x+0)+(y+0)*res3d.x+(z+1)*res3d.x*res3d.y] < thresh;
				count += density_cpu[(x+1)+(y+0)*res3d.x+(z+1)*res3d.x*res3d.y] < thresh;
				count += density_cpu[(x+0)+(y+1)*res3d.x+(z+1)*res3d.x*res3d.y] < thresh;
				count += density_cpu[(x+1)+(y+1)*res3d.x+(z+1)*res3d.x*res3d.y] < thresh;
				if (count>0 && count<8) {
					num_voxels++;
				}

				bool mysign = density_cpu[(x+0)+(y+0)*res3d.x+(z+0)*res3d.x*res3d.y] < thresh;
				bool near_zero_crossing=false;
				near_zero_crossing |= (density_cpu[(x+1)+(y+0)*res3d.x+(z+0)*res3d.x*res3d.y] < thresh) != mysign;
				near_zero_crossing |= (density_cpu[(x-1)+(y+0)*res3d.x+(z+0)*res3d.x*res3d.y] < thresh) != mysign;
				near_zero_crossing |= (density_cpu[(x+0)+(y+1)*res3d.x+(z+0)*res3d.x*res3d.y] < thresh) != mysign;
				near_zero_crossing |= (density_cpu[(x+0)+(y-1)*res3d.x+(z+0)*res3d.x*res3d.y] < thresh) != mysign;
				near_zero_crossing |= (density_cpu[(x+0)+(y+0)*res3d.x+(z+1)*res3d.x*res3d.y] < thresh) != mysign;
				near_zero_crossing |= (density_cpu[(x+0)+(y+0)*res3d.x+(z-1)*res3d.x*res3d.y] < thresh) != mysign;
				if (near_zero_crossing) {
					num_lattice_points_near_zero_crossing++;
				}
			}
		}
	}


	if (swap_y_z) {
		res3d = {res3d.x, res3d.z, res3d.y};
	}

	uint32_t ndown = uint32_t(sqrtf(res3d.z));
	uint32_t nacross = (res3d.z+ndown-1)/ndown;
	uint32_t w = res3d.x * nacross;
	uint32_t h = res3d.y * ndown;
	uint8_t* pngpixels = (uint8_t *)malloc(size_t(w)*size_t(h));
	uint8_t* dst = pngpixels;
	for (int v = 0; v < h; ++v) {
		for (int u = 0; u < w; ++u) {
			int x = u % res3d.x;
			int y = v % res3d.y;
			int z = (u / res3d.x) + (v / res3d.y) * nacross;
			if (z < res3d.z) {
				if (swap_y_z) {
					*dst++ = (uint8_t)clamp((density_cpu[x + z*res3d.x + y*res3d.x*res3d.z]-thresh)*density_scale + 128.5f, 0.f, 255.f);
				} else {
					*dst++ = (uint8_t)clamp((density_cpu[x + (res3d.y-1-y)*res3d.x + z*res3d.x*res3d.y]-thresh)*density_scale + 128.5f, 0.f, 255.f);
				}
			} else {
				*dst++ = 0;
			}
		}
	}

	write_stbi(path, w, h, 1, pngpixels);

	tlog::success() << "Wrote density PNG to " << path.str();
	tlog::info()
		<< "  #lattice points=" << N
		<< " #zero-x voxels=" << num_voxels << " (" << ((num_voxels*100.0)/N) << "%%)"
		<< " #lattice near zero-x=" << num_lattice_points_near_zero_crossing << " (" << ((num_lattice_points_near_zero_crossing*100.0)/N) << "%%)";

	free(pngpixels);
}

// Distinct from `save_density_grid_to_png` not just in that is writes RGBA, but also
// in that it writes a sequence of PNGs rather than a single large PNG.
// TODO: make both methods configurable to do either single PNG or PNG sequence.
void save_rgba_grid_to_png_sequence(const GPUMemory<vec4>& rgba, const fs::path& path, ivec3 res3d, bool swap_y_z) {
	std::vector<vec4> rgba_cpu;
	rgba_cpu.resize(rgba.size());
	rgba.copy_to_host(rgba_cpu);

	if (swap_y_z) {
		res3d = {res3d.x, res3d.z, res3d.y};
	}

	uint32_t w = res3d.x;
	uint32_t h = res3d.y;

	auto progress = tlog::progress(res3d.z);

	std::atomic<int> n_saved{0};
	ThreadPool{}.parallel_for<int>(0, res3d.z, [&](int z) {
		uint8_t* pngpixels = (uint8_t*)malloc(size_t(w) * size_t(h) * 4);
		uint8_t* dst = pngpixels;
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				size_t i = swap_y_z ? (x + z*res3d.x + y*res3d.x*res3d.z) : (x + (res3d.y-1-y)*res3d.x + z*res3d.x*res3d.y);
				*dst++ = (uint8_t)clamp(rgba_cpu[i].x * 255.f, 0.f, 255.f);
				*dst++ = (uint8_t)clamp(rgba_cpu[i].y * 255.f, 0.f, 255.f);
				*dst++ = (uint8_t)clamp(rgba_cpu[i].z * 255.f, 0.f, 255.f);
				*dst++ = (uint8_t)clamp(rgba_cpu[i].w * 255.f, 0.f, 255.f);
			}
		}

		write_stbi(path / fmt::format("{:04d}_{}x{}.png", z, w, h), w, h, 4, pngpixels);
		free(pngpixels);

		progress.update(++n_saved);
	});
	tlog::success() << "Wrote RGBA PNG sequence to " << path;
}

void save_rgba_grid_to_raw_file(const GPUMemory<vec4>& rgba, const fs::path& path, ivec3 res3d, bool swap_y_z, int cascade) {
	std::vector<vec4> rgba_cpu;
	rgba_cpu.resize(rgba.size());
	rgba.copy_to_host(rgba_cpu);

	if (swap_y_z) {
		res3d = {res3d.x, res3d.z, res3d.y};
	}

	uint32_t w = res3d.x;
	uint32_t h = res3d.y;
	uint32_t d = res3d.z;

	auto actual_path = path / fmt::format("{}x{}x{}_{}.bin", w, h, d, cascade);
	FILE* f = native_fopen(actual_path, "wb");
	if (!f)
		return ;
	const static float zero[4]={0.f,0.f,0.f,0.f};
	int border = 1; // extra ring of voxels to make the donut hole smaller
	for (int z = 0; z < d; ++z) {
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				size_t i = swap_y_z ? (x + z*res3d.x + y*res3d.x*res3d.z) : (x + (res3d.y-1-y)*res3d.x + z*res3d.x*res3d.y);
				float* rgba = (float*)&rgba_cpu[i];
				// the intention is that if cascade > 0, we have set up the render aabb such that we are outputing exactly one cascade of the nerf
				// we then punch a transparent hole in the middle of each cascade, so that they fit together when placed on top of each other.
				if (cascade && x>=res3d.x/4+border && y>=res3d.y/4+border && z>=res3d.z/4+border && x<res3d.x-res3d.x/4-border && y<res3d.y-res3d.y/4-border && z<res3d.z-res3d.z/4-border)
					fwrite(zero, sizeof(float), 4, f);
				else
					fwrite(rgba, sizeof(float), 4, f);
			}
		}
	}
	fclose(f);
	tlog::success() << "Wrote RGBA raw file to " << actual_path.str();
}

inline vec3 vertex_interp(float iso, vec3 p1, vec3 p2, float d1, float d2) {
    float denom = (d2 - d1);
    float t = (fabsf(denom) > 1e-8f) ? (iso - d1) / denom : 0.5f;
    t = std::min(std::max(t, 0.0f), 1.0f);
    return p1 + t * (p2 - p1);
}

std::array<vec3, 8> DensityOctree::s_corner_offsets = {{
	{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
	{0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 1.0f}
}};
std::array<int, 8> DensityOctree::s_corner_mapping = {
	0, 1, 3, 2, 4, 5, 7, 6
};
std::array<std::array<int, 2>, 12> DensityOctree::s_edge_connections = {{
	{0,1}, {1,2}, {2,3}, {3,0}, {4,5}, {5,6},
	{6,7}, {7,4}, {0,4}, {1,5}, {2,6}, {3,7}
}};
float DensityOctree::s_min_density = -1e9f;

DensityOctree::DensityOctree(Node&& root) : m_root(std::move(root)) {}

DensityOctree DensityOctree::init(vec3 origin, float size) {
	return DensityOctree(std::move(Node(origin, size)));
}

void DensityOctree::build(int res, float min_density, int max_tree_height, const std::function<std::vector<float>(const vec3& origin, float size)>& get_density_on_grid) {
	m_root.extend(res, 0, max_tree_height, min_density, get_density_on_grid);
	fill_transvoxel_data();
}

DensityOctree::Node::Node() : origin{0.0f, 0.0f, 0.0f}, size{0.0f}, density{0.0f}, children{nullptr}, parent{nullptr}, level{0}, child_idx{-1} {}

DensityOctree::Node::Node(vec3 origin, float size) : origin{origin}, size{size}, density{0.0f}, children{nullptr}, parent{nullptr}, level{0}, child_idx{-1} {}

DensityOctree::Node::Node(vec3 origin, float size, float density, int level, Node* parent, int child_idx) : origin{origin}, size{size}, density{density}, children{nullptr}, parent{parent}, level{level}, child_idx{child_idx} {}

bool DensityOctree::Node::is_corner() const {
	return children == nullptr;
}

bool DensityOctree::Node::is_leaf() const {
	if (is_corner()) return false;

	for (auto& child : *children) {
		if (!child.is_corner()) return false;
	}

	return true;
}

bool DensityOctree::Node::is_intermediate() const {
	return !is_corner() && !is_leaf();
}

void DensityOctree::Node::extend(int res, int depth, int max_tree_height, float min_density, const std::function<std::vector<float>(const vec3& origin, float size)>& get_density_on_grid) {
	float band = 0.01f;
	auto grid_res = res + 1;
	auto density = get_density_on_grid(origin, size);
	auto leaf_nodes = std::vector<DensityOctree::Node>();
	leaf_nodes.reserve(res*res*res);
	auto leaf_size = size / static_cast<float>(res);
	auto corner_size = leaf_size / 2.0f;
	auto extension_height = static_cast<int>(std::log2(res));
	auto leaf_depth = depth + extension_height;
	for(int z = 0; z < res; ++z) {
		for (int y = 0; y < res; ++y) {
			for (int x = 0; x < res; ++x) {
				leaf_nodes.push_back(DensityOctree::Node(origin+leaf_size*vec3(x, y, z), leaf_size));
				leaf_nodes.back().children = std::make_unique<std::array<DensityOctree::Node, 8>>();
				for (int i = 0; i < s_corner_offsets.size(); ++i) {
					(*leaf_nodes.back().children)[i] = DensityOctree::Node(leaf_nodes.back().origin+corner_size*s_corner_offsets[i], corner_size);
					auto corner_pos = ivec3(x, y, z) + ivec3(s_corner_offsets[i]);
					(*leaf_nodes.back().children)[i].density = density[corner_pos.x + grid_res * corner_pos.y + grid_res * grid_res * corner_pos.z];	
					leaf_nodes.back().density += (*leaf_nodes.back().children)[i].density;
				}
				leaf_nodes.back().density /= 8.0f;
				if (!leaf_nodes.back().intersects_band(band, min_density)) {
					if (leaf_nodes.back().density < min_density) leaf_nodes.back().children = nullptr;
				}
				else {
					if (leaf_depth < max_tree_height) leaf_nodes.back().extend(res, leaf_depth, max_tree_height, min_density, get_density_on_grid);
				}
			}
		}
	}
	auto current_res = res;
	while(current_res > 2) {
		auto half_res = current_res >> 1;
		auto intermediate_nodes = std::vector<DensityOctree::Node>();
		intermediate_nodes.reserve(half_res*half_res*half_res);
		for (int z = 0; z < current_res; z+=2) {
			for (int y = 0; y < current_res; y+=2) {
				for (int x = 0; x < current_res; x+=2) {
					auto& base_corner_node = leaf_nodes[x + current_res * y + current_res * current_res * z];
					intermediate_nodes.push_back(DensityOctree::Node(base_corner_node.origin, base_corner_node.size * 2.0f));
					intermediate_nodes.back().children = std::make_unique<std::array<DensityOctree::Node, 8>>();
					for (int i = 0; i < s_corner_offsets.size(); ++i) {
						auto pos = ivec3(x, y, z) + ivec3(s_corner_offsets[i]);
						(*intermediate_nodes.back().children)[i] = std::move(leaf_nodes[pos.x + current_res * pos.y + current_res * current_res * pos.z]);
						intermediate_nodes.back().density += (*intermediate_nodes.back().children)[i].density;
					}
					intermediate_nodes.back().density /= 8.0f;
					if (!intermediate_nodes.back().intersects_band(band, min_density)) {
						if (intermediate_nodes.back().density < min_density) intermediate_nodes.back().children = nullptr;
					}
				}
			}
		}
		current_res = half_res;
		leaf_nodes = std::move(intermediate_nodes);
	}
	this->children = std::make_unique<std::array<DensityOctree::Node, 8>>();
	for (int i = 0; i < leaf_nodes.size(); ++i) {
		(*children)[i] = std::move(leaf_nodes[i]);
		this->density += (*children)[i].density;
	}
	this->density /= 8.0f;
	if (!this->intersects_band(band, min_density)) {
		if (this->density < min_density) this->children = nullptr;
	}
}

DensityOctree::Node& DensityOctree::get_root() {
	return m_root;
}

void DensityOctree::traverse(const std::function<void(Node&)>& fun) {
	m_root.traverse(fun);
}

void DensityOctree::Node::traverse(const std::function<void(Node&)>& fun) {
	fun(*this);
	if (children == nullptr) return;
	for (auto& node : *children) {
		node.traverse(fun);
	}
}

float DensityOctree::sample_density(vec3 pos) {
	return m_root.sample_density(pos);
}

float DensityOctree::Node::sample_density(vec3 pos) {
	if (pos.x < origin.x || pos.x > origin.x + size ||
		pos.y < origin.y || pos.y > origin.y + size ||
		pos.z < origin.z || pos.z > origin.z + size) {
		return s_min_density;
	}

	if (children == nullptr) return density;

    int child_idx = 0;
    if (pos.x >= origin.x + size / 2.0f) child_idx |= 1;
    if (pos.y >= origin.y + size / 2.0f) child_idx |= 2;
    if (pos.z >= origin.z + size / 2.0f) child_idx |= 4;
	child_idx = s_corner_mapping[child_idx];

	if ((*children)[child_idx].is_intermediate()) return (*children)[child_idx].sample_density(pos);

	vec3 local = (pos - origin) / size;

	float d00 = (*children)[0].density * (1.0f - local.x) + (*children)[1].density * local.x;
	float d01 = (*children)[3].density * (1.0f - local.x) + (*children)[2].density * local.x;
	float d10 = (*children)[4].density * (1.0f - local.x) + (*children)[5].density * local.x;
	float d11 = (*children)[7].density * (1.0f - local.x) + (*children)[6].density * local.x;

	float d0 = d00 * (1.0f - local.y) + d01 * local.y;
	float d1 = d10 * (1.0f - local.y) + d11 * local.y;

	return d0 * (1.0 - local.z) + d1 * local.z;
}

inline uint64_t mc_edge_key_from_tables(const ivec3& cellMinBase, int cellSizeBase, int edge) {
    const int ca = DensityOctree::s_edge_connections[edge][0];
    const int cb = DensityOctree::s_edge_connections[edge][1];

    auto to_int_off = [&](int corner)->ivec3 {
        const vec3& o = DensityOctree::s_corner_offsets[corner];
        return ivec3{ int(o.x + 0.5f), int(o.y + 0.5f), int(o.z + 0.5f) };
    };

    const ivec3 oa = to_int_off(ca);
    const ivec3 ob = to_int_off(cb);

    const ivec3 pa{ cellMinBase.x + oa.x*cellSizeBase,
                 cellMinBase.y + oa.y*cellSizeBase,
                 cellMinBase.z + oa.z*cellSizeBase };
    const ivec3 pb{ cellMinBase.x + ob.x*cellSizeBase,
                 cellMinBase.y + ob.y*cellSizeBase,
                 cellMinBase.z + ob.z*cellSizeBase };

    int axis = 0;
    if (pa.x != pb.x) axis = 0;
    else if (pa.y != pb.y) axis = 1;
    else                   axis = 2;

    ivec3 lower = pa;
    if ((axis == 0 && pb.x < pa.x) ||
        (axis == 1 && pb.y < pa.y) ||
        (axis == 2 && pb.z < pa.z)) {
        lower = pb;
    }

    return (morton3D_21(uint32_t(lower.x), uint32_t(lower.y), uint32_t(lower.z)) << 2)
         | uint64_t(axis & 3);
}

MCMesh DensityOctree::polygonize(float iso, int max_tree_height) {
	MCMesh mesh{};
	mesh.base_h = 1.0f / (1 << max_tree_height);

	traverse([&mesh, &iso, this](Node& node) {
		if (node.is_leaf()) node.polygonize(mesh, iso, std::bind(&DensityOctree::sample_density, this, std::placeholders::_1));
	});

	mesh.mcEdgeCache.clear();
	mesh.faceEdgeCache.clear();

	return mesh;
}

static constexpr uint16_t edge_table[256] =
{
	0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
	0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
	0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
	0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
	0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
	0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
	0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
	0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
	0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
	0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
	0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
	0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
	0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
	0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
	0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
	0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
};

static constexpr int8_t triangle_table[256][16] =
{
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
	{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
	{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
	{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
	{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
	{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
	{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
	{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
	{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
	{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
	{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
	{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
	{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
	{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
	{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
	{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
	{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
	{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
	{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
	{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
	{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
	{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
	{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
	{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
	{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
	{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
	{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
	{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
	{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
	{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
	{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
	{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
	{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
	{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
	{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
	{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
	{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
	{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
	{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
	{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
	{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
	{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
	{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
	{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
	{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
	{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
	{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
	{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
	{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
	{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
	{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
	{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
	{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
	{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
	{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
	{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
	{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
	{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
	{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
	{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
	{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
	{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
	{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
	{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
	{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
	{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
	{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
	{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
	{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
	{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
	{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
	{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
	{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
	{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
	{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
	{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
	{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
	{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
	{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
	{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
	{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
	{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
	{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
	{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
	{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
	{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
	{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
	{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
	{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
	{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
	{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
	{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
	{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
	{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
	{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
	{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
	{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
	{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
	{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
	{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
	{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
	{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
	{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
	{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
	{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
	{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
	{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
	{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
	{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
	{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
	{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
	{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
	{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
	{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
	{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
	{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
	{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
	{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
	{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
	{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
	{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
	{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
	{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
	{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
	{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
	{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
	{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
	{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
	{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
	{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
	{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
	{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
	{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
	{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
	{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
	{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
	{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
	{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
	{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};

void DensityOctree::Node::polygonize(MCMesh& mesh, float iso, const std::function<float(vec3 pos)>& sample_density) {
	polygonize_marching_cubes(mesh, iso);
	polygonize_transvoxel(mesh, iso, sample_density);
}

void DensityOctree::Node::polygonize_marching_cubes(MCMesh& mesh, float iso) {
    std::array<vec3, 8>  corner_pos;
    std::array<float, 8> corner_densities;

    for (int i = 0; i < 8; ++i) {
        corner_pos[i]      = origin + size * s_corner_offsets[i];
        corner_densities[i]= (*children)[i].density;  // your stored density per corner
    }

    int cubeIndex = 0;
    for (int i = 0; i < 8; ++i) {
        if (corner_densities[i] < iso) cubeIndex |= (1 << i);
    }
    if (edge_table[cubeIndex] == 0) return;

    const ivec3 cellMinBase = to_base(origin, mesh.base_origin, mesh.base_h);
    const int cellSizeBase = (int)lroundf(size / mesh.base_h);

    int vertIdx[12];
    for (int e = 0; e < 12; ++e) {
        if (edge_table[cubeIndex] & (1 << e)) {

            auto creator = [&]() -> int {
                const int a = s_edge_connections[e][0];
                const int b = s_edge_connections[e][1];
                const vec3 pA = corner_pos[a];
                const vec3 pB = corner_pos[b];
                const float dA = corner_densities[a];
                const float dB = corner_densities[b];
                const vec3 P = vertex_interp(iso, pA, pB, dA, dB);
                return (int)mesh.add_vertex(P);
            };

            vertIdx[e] = mesh.mcEdgeCache.get_or_create(
				mc_edge_key_from_tables(cellMinBase, cellSizeBase, e),
				creator
			);
        } else {
            vertIdx[e] = -1;
        }
    }

    for (int i = 0; triangle_table[cubeIndex][i] != -1; i += 3) {
        const int e0 = triangle_table[cubeIndex][i+0];
        const int e1 = triangle_table[cubeIndex][i+1];
        const int e2 = triangle_table[cubeIndex][i+2];

        if (vertIdx[e0] < 0 || vertIdx[e1] < 0 || vertIdx[e2] < 0) continue;

        mesh.indices.push_back((uint32_t)vertIdx[e0]);
        mesh.indices.push_back((uint32_t)vertIdx[e1]);
        mesh.indices.push_back((uint32_t)vertIdx[e2]);
    }
}


std::vector<std::pair<DensityOctree::Node&, Face>> DensityOctree::Node::get_neighors_faces() {
	std::vector<std::pair<DensityOctree::Node&, Face>> result;

	for (auto& face : {Face::NX, Face::PX, Face::NY, Face::PY, Face::NZ, Face::PZ}) {
		auto res = get_face_neighbor(face);
		if (res) result.push_back({*res, face});
	}

	return result;
}

struct Dir { int axis; int dir; };
Dir faceDir(Face f){
    switch (f){
        case Face::PX: return {0,+1}; case Face::NX: return {0,-1};
        case Face::PY: return {1,+1}; case Face::NY: return {1,-1};
        case Face::PZ: return {2,+1}; default:       return {2,-1};
    }
}

DensityOctree::Node* descendTowardInterface(DensityOctree::Node* nb, const std::vector<uint8_t>& pathBits, const Dir& d) {
    for (int i = (int)pathBits.size() - 1; nb && nb->children != nullptr && i >= 0; --i) {
        uint8_t b = pathBits[i];
        int cx = (d.axis==0) ? ((d.dir==+1)?0:1) : (b & 1);
        int cy = (d.axis==1) ? ((d.dir==+1)?0:1) : ((b >> 1) & 1);
        int cz = (d.axis==2) ? ((d.dir==+1)?0:1) : ((b >> 2) & 1);
        int child = (cx) | (cy<<1) | (cz<<2);
        nb = &(*nb->children)[child];
    }
    return nb;
}

int bitX(uint8_t childIndex) { return (childIndex >> 0) & 1; }
int bitY(uint8_t childIndex) { return (childIndex >> 1) & 1; }
int bitZ(uint8_t childIndex) { return (childIndex >> 2) & 1; }

DensityOctree::Node* DensityOctree::Node::get_face_neighbor(Face f) {
    Dir d = faceDir(f);

    if (this->parent) {
        int b = (d.axis==0) ? bitX(this->child_idx) : (d.axis==1) ? bitY(this->child_idx) : bitZ(this->child_idx);
        bool canStepInsideParent = (d.dir==+1 ? (b==0) : (b==1));
        if (canStepInsideParent) {
            int acrossChild = this->child_idx ^ (1 << d.axis);
            Node* nb = this->parent->children != nullptr ? &(*(this->parent->children))[acrossChild] : nullptr;
            if (!nb) return nullptr;

            std::vector<uint8_t> oneStep { (uint8_t)this->child_idx };
            nb = descendTowardInterface(nb, oneStep, d);

            if (!nb) return nullptr;
            return nb;
        }
    }

    std::vector<uint8_t> pathBits;
    Node* a = this;
    Node* p = a->parent;

    while (p) {
        int b = (d.axis==0) ? bitX(a->child_idx) : (d.axis==1) ? bitY(a->child_idx) : bitZ(a->child_idx);
        bool onOuterSide = (d.dir==+1 ? (b==1) : (b==0));
        if (!onOuterSide) break;
        pathBits.push_back(a->child_idx);
        a = p;
        p = a->parent;
    }

    if (!p) {

        return nullptr;
    }

    int acrossChild = a->child_idx ^ (1 << d.axis);
    Node* nb = (p->children != nullptr) ? &(*p->children)[acrossChild] : nullptr;
    if (!nb) return nullptr;

    nb = descendTowardInterface(nb, pathBits, d);

    if (!nb) return nullptr;
    return nb;
}

static vec3 ex() { return {1,0,0}; }
static vec3 ey() { return {0,1,0}; }
static vec3 ez() { return {0,0,1}; }

void setRow(mat4& M, int r, const vec3& d, const vec3& p0) {
    M.m[r][0] = d.x;  M.m[r][1] = d.y;  M.m[r][2] = d.z;  M.m[r][3] = -dot(d, p0);
}

mat4 faceBasis(Face f, const vec3& origin, float h, uint8_t childIndex) {
    vec3 u, v, w;
    vec3 p0;

    switch (f) {
        case Face::PX: {
            w = ex(); u = ey(); v = ez();
            const float y0 = origin.y - bitY(childIndex) * h;
            const float z0 = origin.z - bitZ(childIndex) * h;
            p0 = { origin.x + h, y0, z0 };
            break;
        }
        case Face::NX: {
            w = { -1, 0, 0 }; u = ez(); v = ey();
            const float y0 = origin.y - bitY(childIndex) * h;
            const float z0 = origin.z - bitZ(childIndex) * h;
            p0 = { origin.x, y0, z0 };
            break;
        }
        case Face::PY: {
            w = ey(); u = ez(); v = ex();
            const float x0 = origin.x - bitX(childIndex) * h;
            const float z0 = origin.z - bitZ(childIndex) * h;
            p0 = { x0, origin.y + h, z0 };
            break;
        }
        case Face::NY: {
            w = { 0, -1, 0 }; u = ex(); v = ez();
            const float x0 = origin.x - bitX(childIndex) * h;
            const float z0 = origin.z - bitZ(childIndex) * h;
            p0 = { x0, origin.y, z0 };
            break;
        }
        case Face::PZ: {
            w = ez(); u = ex(); v = ey();
            const float x0 = origin.x - bitX(childIndex) * h;
            const float y0 = origin.y - bitY(childIndex) * h;
            p0 = { x0, y0, origin.z + h };
            break;
        }
        case Face::NZ: {
            w = { 0, 0, -1 }; u = ey(); v = ex();
            const float x0 = origin.x - bitX(childIndex) * h;
            const float y0 = origin.y - bitY(childIndex) * h;
            p0 = { x0, y0, origin.z };
            break;
        }
    }

    mat4 M = mat4::identity();
    setRow(M, 0, u, p0);
    setRow(M, 1, v, p0);
    setRow(M, 2, w, p0);
    return M;
}

struct SampleSet {
    std::array<vec3, 13> pos;
    std::array<float, 13> val;
};

vec3 transformPoint(const mat4& M, const vec3& p) {
    vec4 h = M * vec4(p, 1.0f);
    return vec3(h.x, h.y, h.z) / h.w;
}

void gatherNearFaceSamples9(SampleSet& S,
                            int qx, int qy, float h,
                            const mat4& toWorld,
                            const std::function<float(vec3)>& sd,
                            int baseIndex = 0)
{
    const float x0 = qx * h;
    const float y0 = qy * h;
    const float x1 = x0 + h;
    const float y1 = y0 + h;
    const float dx = h * 0.5f;
    const float dy = h * 0.5f;

    {
        vec3 p0(x0,      y0,      0.0f);
        vec3 p1(x0+dx,   y0,      0.0f);
        vec3 p2(x1,      y0,      0.0f);
        S.pos[baseIndex + 0] = p0; S.val[baseIndex + 0] = sd(transformPoint(toWorld, p0));
        S.pos[baseIndex + 1] = p1; S.val[baseIndex + 1] = sd(transformPoint(toWorld, p1));
        S.pos[baseIndex + 2] = p2; S.val[baseIndex + 2] = sd(transformPoint(toWorld, p2));
    }

    {
        vec3 p3(x0,      y0+dy,   0.0f);
        vec3 p4(x0+dx,   y0+dy,   0.0f);
        vec3 p5(x1,      y0+dy,   0.0f);
        S.pos[baseIndex + 3] = p3; S.val[baseIndex + 3] = sd(transformPoint(toWorld, p3));
        S.pos[baseIndex + 4] = p4; S.val[baseIndex + 4] = sd(transformPoint(toWorld, p4));
        S.pos[baseIndex + 5] = p5; S.val[baseIndex + 5] = sd(transformPoint(toWorld, p5));
    }

    {
        vec3 p6(x0,      y1,      0.0f);
        vec3 p7(x0+dx,   y1,      0.0f);
        vec3 p8(x1,      y1,      0.0f);
        S.pos[baseIndex + 6] = p6; S.val[baseIndex + 6] = sd(transformPoint(toWorld, p6));
        S.pos[baseIndex + 7] = p7; S.val[baseIndex + 7] = sd(transformPoint(toWorld, p7));
        S.pos[baseIndex + 8] = p8; S.val[baseIndex + 8] = sd(transformPoint(toWorld, p8));
    }
}

void addBackSamples4(SampleSet& S,
                     int qx, int qy, float h,
                     const mat4& toWorld,
                     const std::function<float(vec3)>& sd,
                     float zBack = -1.0f,
                     int baseIndex = 9) {
    if (zBack < 0.0f) zBack = h;

    const float x0 = qx * h;
    const float y0 = qy * h;
    const float x1 = x0 + h;
    const float y1 = y0 + h;

    const vec3 pBL(x0, y0, zBack);
    const vec3 pBR(x1, y0, zBack);
    const vec3 pTL(x0, y1, zBack);
    const vec3 pTR(x1, y1, zBack);

    auto W = [&](const vec3& p)->vec3{ return transformPoint(toWorld, p); };

    S.pos[baseIndex + 0] = pBL; S.val[baseIndex + 0] = sd(W(pBL));
    S.pos[baseIndex + 1] = pBR; S.val[baseIndex + 1] = sd(W(pBR));
    S.pos[baseIndex + 2] = pTL; S.val[baseIndex + 2] = sd(W(pTL));
    S.pos[baseIndex + 3] = pTR; S.val[baseIndex + 3] = sd(W(pTR));
}

inline uint8_t sign_bit(float d, float iso) {
    constexpr float EPS = 1e-6f;
    float q = d - iso;
    if (q <= -EPS) return 1u;
    if (q >=  EPS) return 0u;
    return 0u;
}

uint16_t buildTransitionCaseIndex(const SampleSet& S, float iso) {
    uint32_t nearMask = 0u;
    for (int i = 0; i < 9; ++i) {
        nearMask |= (uint32_t(sign_bit(S.val[i], iso)) << i);
    }
    return (uint16_t)nearMask;
}

TransitionTables T = buildTransitionTables();

void quadrantForFace(Face f, uint8_t childIndex, int& qx, int& qy) {
    switch (f) {
        case Face::PX: case Face::NX: qx = bitY(childIndex); qy = bitZ(childIndex); break;
        case Face::PY: case Face::NY: qx = bitZ(childIndex); qy = bitX(childIndex); break;
        case Face::PZ: case Face::NZ: qx = bitX(childIndex); qy = bitY(childIndex); break;
    }
}

static inline void world_base_to_face_uv(const ivec3& coarseC, int coarseS, int face,
                                         const ivec3& w, uint32_t& u, uint32_t& v) {
    switch(face){
        case 0: u = (uint32_t)(w.y - coarseC.y); v = (uint32_t)(w.z - coarseC.z); break;
        case 1: u = (uint32_t)(w.y - coarseC.y); v = (uint32_t)(w.z - coarseC.z); break;
        case 2: u = (uint32_t)(w.x - coarseC.x); v = (uint32_t)(w.z - coarseC.z); break;
        case 3: u = (uint32_t)(w.x - coarseC.x); v = (uint32_t)(w.z - coarseC.z); break;
        case 4: u = (uint32_t)(w.x - coarseC.x); v = (uint32_t)(w.y - coarseC.y); break;
        default:u = (uint32_t)(w.x - coarseC.x); v = (uint32_t)(w.y - coarseC.y); break;
    }
}

static inline int dir_from_two_points_on_face(int face, const ivec3& w0, const ivec3& w1) {
    switch(face){
        case 0: case 1: return (w0.y != w1.y) ? 0 : 1;
        case 2: case 3: return (w0.x != w1.x) ? 0 : 1;
        default:        return (w0.x != w1.x) ? 0 : 1;
    }
}


void DensityOctree::Node::emit_transition(
    MCMesh& mesh, float iso, const std::function<float(vec3)>& sd,
    DensityOctree::Node& nb, Face f)
{
    const float  fine_h_world   = this->size;
    const float  coarse_h_world = nb.size;
    const int    coarseSBase    = int(std::lround(coarse_h_world / mesh.base_h));
    const ivec3     coarseCBase    = to_base(nb.origin, mesh.base_origin, mesh.base_h);

    mat4 toFace  = faceBasis(f, this->origin, fine_h_world, this->child_idx);
    mat4 toWorld = inverse(toFace);

    SampleSet S;
    int qx=0, qy=0; quadrantForFace(f, this->child_idx, qx, qy);
    gatherNearFaceSamples9(S, qx, qy, fine_h_world, toWorld, sd);
    addBackSamples4(S, qx, qy, fine_h_world, toWorld, sd, fine_h_world);

    const uint16_t tCase = buildTransitionCaseIndex(S, iso);
    static TransitionTables T = buildTransitionTables();

    const uint32_t mask = T.edgeTable[tCase];
    if (mask == 0) return;

    std::array<int, TransitionTables::MAX_T_EDGES> vIdx{};
    for (int e = 0; e < (int)T.edgeCount[tCase]; ++e) {
        if (!(mask & (1u << e))) { vIdx[e] = -1; continue; }

        auto [fp0, fp1, d0, d1] = T.edgeEndpoints(tCase, e, S);
        vec3 w0 = transformPoint(toWorld, fp0);
        vec3 w1 = transformPoint(toWorld, fp1);

        const ivec3 wb0 = to_base(w0, mesh.base_origin, mesh.base_h);
        const ivec3 wb1 = to_base(w1, mesh.base_origin, mesh.base_h);

        const int dirUV = dir_from_two_points_on_face((int)f, wb0, wb1);

        ivec3 lowerW = ivec3{ std::min(wb0.x, wb1.x), std::min(wb0.y, wb1.y), std::min(wb0.z, wb1.z) };

        uint32_t u0=0, v0=0;
        world_base_to_face_uv(coarseCBase, coarseSBase, (int)f, lowerW, u0, v0);

        const bool onBorder = is_face_edge_on_border(u0, v0, dirUV, coarseSBase);

        auto face_creator = [&]() -> int {
            vec3 P = vertex_interp(iso, w0, w1, d0, d1);
            int id = (int)mesh.vertices.size();
            mesh.vertices.push_back(P);
            return id;
        };

        auto coarse_creator = [&]() -> int {
            ivec3 mcLower; int mcAxis;
            face_edge_to_coarse_mc_edge(coarseCBase, coarseSBase, (int)f, u0, v0, dirUV, mcLower, mcAxis);

            vec3 pA = mesh.base_origin + vec3{ mcLower.x * mesh.base_h, mcLower.y * mesh.base_h, mcLower.z * mesh.base_h };
            vec3 pB = pA;
            if (mcAxis == 0) pB.x += coarseSBase * mesh.base_h;
            else if (mcAxis == 1) pB.y += coarseSBase * mesh.base_h;
            else                  pB.z += coarseSBase * mesh.base_h;

            float dA = sd(pA);
            float dB = sd(pB);
            vec3  P  = vertex_interp(iso, pA, pB, dA, dB);

            int id = (int)mesh.vertices.size();
            mesh.vertices.push_back(P);
            return id;
        };

        if (onBorder) {
            ivec3 mcLower; int mcAxis;
            face_edge_to_coarse_mc_edge(coarseCBase, coarseSBase, (int)f, u0, v0, dirUV, mcLower, mcAxis);
            const uint64_t mcKey = mc_edge_key_from_lower_and_axis(mcLower, mcAxis);
            vIdx[e] = mesh.mcEdgeCache.get_or_create(mcKey, coarse_creator);
        } else {
            const uint64_t feKey = face_edge_key(coarseCBase, coarseSBase, (int)f, u0, v0, dirUV);
            vIdx[e] = mesh.faceEdgeCache.get_or_create(feKey, face_creator);
        }
    }

    for (int i = 0; T.triTable[tCase][i] != -1; i += 3) {
        const int e0 = T.triTable[tCase][i+0];
        const int e1 = T.triTable[tCase][i+1];
        const int e2 = T.triTable[tCase][i+2];
        if (vIdx[e0] < 0 || vIdx[e1] < 0 || vIdx[e2] < 0) continue;
        mesh.indices.push_back((uint32_t)vIdx[e0]);
        mesh.indices.push_back((uint32_t)vIdx[e1]);
        mesh.indices.push_back((uint32_t)vIdx[e2]);
    }
}

void DensityOctree::Node::polygonize_transvoxel(MCMesh& mesh, float iso, const std::function<float(vec3 pos)>& sample_density) {
    for (auto& [nb, f] : get_neighors_faces()) {
        if (nb.size == size) continue;
        if (nb.size == size*2) {
            emit_transition(mesh, iso, sample_density, nb, f);
        }
    }
}

void DensityOctree::fill_transvoxel_data() {
	traverse([](DensityOctree::Node& node){
		if (node.children == nullptr) return;
		for (int i = 0; i < 8; ++i) {
			(*node.children)[i].parent = &node;
			(*node.children)[i].child_idx = i;
			(*node.children)[i].level = node.level + 1;
		}
	});
}

bool DensityOctree::Node::intersects_band(float band, float iso) {
	if (is_corner()) return 0;

    float dmin = std::numeric_limits<float>::max();
	float dmax = std::numeric_limits<float>::min();
    for (int i=0;i<8;++i) {
        dmin = std::min(dmin, (*children)[i].density);
        dmax = std::max(dmax, (*children)[i].density);
    }

	return (dmin <= iso + band) && (dmax >= iso - band);;
}

}
