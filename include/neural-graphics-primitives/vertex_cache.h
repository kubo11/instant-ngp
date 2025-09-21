#pragma once
#include <cstdint>
#include <unordered_map>
#include <cmath>
#include <algorithm>

#include <tiny-cuda-nn/vec.h>
using namespace tcnn;

inline uint64_t part1by2_64(uint32_t x){
    uint64_t v = x & 0x1fffff;
    v = (v | (v << 32)) & 0x1f00000000ffffULL;
    v = (v | (v << 16)) & 0x1f0000ff0000ffULL;
    v = (v | (v << 8 )) & 0x100f00f00f00f00fULL;
    v = (v | (v << 4 )) & 0x10c30c30c30c30c3ULL;
    v = (v | (v << 2 )) & 0x1249249249249249ULL;
    return v;
}
inline uint64_t morton3D_21(uint32_t x, uint32_t y, uint32_t z){
    return (part1by2_64(z) << 2) | (part1by2_64(y) << 1) | part1by2_64(x);
}
inline uint64_t part1by1_64(uint32_t x){
    uint64_t v = x & 0x000fffff;
    v = (v | (v << 16)) & 0x0000ffff0000ffffULL;
    v = (v | (v << 8 )) & 0x00ff00ff00ff00ffULL;
    v = (v | (v << 4 )) & 0x0f0f0f0f0f0f0f0fULL;
    v = (v | (v << 2 )) & 0x3333333333333333ULL;
    v = (v | (v << 1 )) & 0x5555555555555555ULL;
    return v;
}
inline uint64_t morton2D_20(uint32_t x, uint32_t y){
    return (part1by1_64(y) << 1) | part1by1_64(x);
}

inline ivec3 to_base(const vec3& p, const vec3& origin, float h) {
    return { (int)floorf((p.x - origin.x)/h + 1e-6f),
             (int)floorf((p.y - origin.y)/h + 1e-6f),
             (int)floorf((p.z - origin.z)/h + 1e-6f) };
}

inline int mc_edge_axis(int e){
    static const int A[12]={0,1,2, 0,1,2, 0,1,2, 0,1,2};
    return A[e];
}
inline void mc_edge_lower_offset(int e, int& ox,int& oy,int& oz){
    static const int C[12][3]={
        {0,0,0},{1,0,0},{0,0,0},{0,1,0},
        {0,0,1},{1,0,1},{0,0,1},{0,1,1},
        {0,0,0},{1,0,0},{1,1,0},{0,1,0}
    };
    ox=C[e][0]; oy=C[e][1]; oz=C[e][2];
}

inline uint64_t mc_edge_key(const ivec3& C, int s, int edge){
    int ox,oy,oz; mc_edge_lower_offset(edge,ox,oy,oz);
    uint32_t x = uint32_t(C.x + ox*s);
    uint32_t y = uint32_t(C.y + oy*s);
    uint32_t z = uint32_t(C.z + oz*s);
    return (morton3D_21(x,y,z) << 2) | uint64_t(mc_edge_axis(edge) & 3);
}

inline uint64_t mc_edge_key_from_lower_and_axis(const ivec3& lower, int axis){
    return (morton3D_21(uint32_t(lower.x),uint32_t(lower.y),uint32_t(lower.z)) << 2)
           | uint64_t(axis & 3);
}

inline uint64_t face_uid(const ivec3& coarseC, int coarseS, int face){
    uint64_t mort = morton3D_21(uint32_t(coarseC.x),uint32_t(coarseC.y),uint32_t(coarseC.z));
    return ((mort << 8) | uint64_t(coarseS & 0xff)) << 3 | uint64_t(face & 7);
}
inline uint64_t face_edge_key(const ivec3& coarseC, int coarseS, int face,
                              uint32_t u0, uint32_t v0, int dirUV){
    return (face_uid(coarseC,coarseS,face) << 45) | uint64_t((dirUV&1) << 44)
           | morton2D_20(u0, v0);
}
inline bool is_face_edge_on_border(uint32_t u0,uint32_t v0,int dir,int coarseS){
    const uint32_t S = uint32_t(coarseS);
    return (dir==0) ? (v0==0u || v0==S) : (u0==0u || u0==S);
}

inline void face_edge_to_coarse_mc_edge(const ivec3& coarseC, int coarseS, int face,
                                        uint32_t u0, uint32_t v0, int dir,
                                        ivec3& lower_world, int& axis_out){
    switch(face){
        case 0: lower_world = ivec3(coarseC.x,             coarseC.y + int(u0), coarseC.z + int(v0)); axis_out=(dir==0)?1:2; break;
        case 1: lower_world = ivec3(coarseC.x + coarseS,   coarseC.y + int(u0), coarseC.z + int(v0)); axis_out=(dir==0)?1:2; break;
        case 2: lower_world = ivec3(coarseC.x + int(u0),   coarseC.y,           coarseC.z + int(v0)); axis_out=(dir==0)?0:2; break;
        case 3: lower_world = ivec3(coarseC.x + int(u0),   coarseC.y + coarseS, coarseC.z + int(v0)); axis_out=(dir==0)?0:2; break;
        case 4: lower_world = ivec3(coarseC.x + int(u0),   coarseC.y + int(v0), coarseC.z);           axis_out=(dir==0)?0:1; break;
        default:lower_world = ivec3(coarseC.x + int(u0),   coarseC.y + int(v0), coarseC.z + coarseS); axis_out=(dir==0)?0:1; break;
    }
}

struct CPUHash64 {
    std::unordered_map<uint64_t,int> map_;
    template <class Creator>
    int get_or_create(uint64_t key, Creator create){
        auto it = map_.find(key);
        if (it != map_.end()) return it->second;
        int v = create();
        map_[key] = v;
        return v;
    }
    void clear() {
      map_.clear();
    }
};

template <class Creator>
inline int mc_vertex_index(CPUHash64& cache,
                           const ivec3& C, int s, int edge,
                           Creator create){
    return cache.get_or_create(mc_edge_key(C,s,edge), create);
}

template <class CreatorFace, class CreatorMC>
inline int transvoxel_vertex_index(CPUHash64& faceEdgeCache,
                                   CPUHash64& mcEdgeCache,
                                   const ivec3& coarseC, int coarseS, int face,
                                   uint32_t u0, uint32_t v0, int dirUV,
                                   CreatorFace faceCreator,
                                   CreatorMC   mcCreator){
    if (is_face_edge_on_border(u0,v0,dirUV,coarseS)){
        ivec3 lower; int axis;
        face_edge_to_coarse_mc_edge(coarseC,coarseS,face,u0,v0,dirUV,lower,axis);
        uint64_t k = mc_edge_key_from_lower_and_axis(lower, axis);
        return mcEdgeCache.get_or_create(k, mcCreator);   // must use COARSE endpoints
    } else {
        uint64_t k = face_edge_key(coarseC,coarseS,face,u0,v0,dirUV);
        return faceEdgeCache.get_or_create(k, faceCreator);
    }
}
