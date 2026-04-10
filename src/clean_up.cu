#include "cumesh.h"
#include "dtypes.cuh"
#include "shared.h"
#include <cub/cub.cuh>

#define cudaMalloc torch_cudaMalloc
#define cudaFree torch_cudaFree

namespace cumesh {


template<typename T, typename U>
static __global__ void copy_T_to_T3_kernel(
    const T* input,
    const size_t N,
    U* output
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    output[tid] = { input[3 * tid], input[3 * tid + 1], input[3 * tid + 2] };
}


void CuMesh::remove_faces(torch::Tensor& face_mask) {
    size_t F = this->faces.size;

    size_t temp_storage_bytes = 0;
    int *cu_new_num_faces;
    int3 *cu_new_faces;
    CUDA_CHECK(cudaMalloc(&cu_new_num_faces, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_new_faces, F * sizeof(int3)));
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, temp_storage_bytes,
        this->faces.ptr, face_mask.data_ptr<bool>(), cu_new_faces, cu_new_num_faces,
        F
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        this->faces.ptr, face_mask.data_ptr<bool>(), cu_new_faces, cu_new_num_faces,
        F
    ));
    int new_num_faces;
    CUDA_CHECK(cudaMemcpy(&new_num_faces, cu_new_num_faces, sizeof(int), cudaMemcpyDeviceToHost));
    this->faces.resize(new_num_faces);
    CUDA_CHECK(cudaMemcpy(this->faces.ptr, cu_new_faces, new_num_faces * sizeof(int3), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(cu_new_num_faces));
    CUDA_CHECK(cudaFree(cu_new_faces));

    this->remove_unreferenced_vertices();
}


void CuMesh::_remove_faces(uint8_t* face_mask) {
    size_t F = this->faces.size;

    size_t temp_storage_bytes = 0;
    int *cu_new_num_faces;
    int3 *cu_new_faces;
    CUDA_CHECK(cudaMalloc(&cu_new_num_faces, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_new_faces, F * sizeof(int3)));
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, temp_storage_bytes,
        this->faces.ptr, face_mask, cu_new_faces, cu_new_num_faces,
        F
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        this->faces.ptr, face_mask, cu_new_faces, cu_new_num_faces,
        F
    ));
    int new_num_faces;
    CUDA_CHECK(cudaMemcpy(&new_num_faces, cu_new_num_faces, sizeof(int), cudaMemcpyDeviceToHost));
    this->faces.resize(new_num_faces);
    CUDA_CHECK(cudaMemcpy(this->faces.ptr, cu_new_faces, new_num_faces * sizeof(int3), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(cu_new_num_faces));
    CUDA_CHECK(cudaFree(cu_new_faces));

    this->remove_unreferenced_vertices();
}


static __global__ void set_vertex_is_referenced(
    const int3* faces,
    const size_t F,
    int* vertex_is_referenced
) {
    const int fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= F) return;
    int3 face = faces[fid];
    vertex_is_referenced[face.x] = 1;
    vertex_is_referenced[face.y] = 1;
    vertex_is_referenced[face.z] = 1;
}


static __global__ void compress_vertices_kernel(
    const int* vertices_map,
    const float3* old_vertices,
    const int V,
    float3* new_vertices
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= V) return;
    int new_id = vertices_map[tid];
    int is_kept = vertices_map[tid + 1] == new_id + 1;
    if (is_kept) {
        new_vertices[new_id] = old_vertices[tid];
    }
}


static __global__ void remap_faces_kernel(
    const int* vertices_map,
    const int F,
    int3* faces
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;
    faces[tid].x = vertices_map[faces[tid].x];
    faces[tid].y = vertices_map[faces[tid].y];
    faces[tid].z = vertices_map[faces[tid].z];
}


void CuMesh::remove_unreferenced_vertices() {
    size_t V = this->vertices.size;
    size_t F = this->faces.size;

    // Mark referenced vertices
    int* cu_vertex_is_referenced;
    CUDA_CHECK(cudaMalloc(&cu_vertex_is_referenced, (V+1) * sizeof(int)));
    CUDA_CHECK(cudaMemset(cu_vertex_is_referenced, 0, (V+1) * sizeof(int)));
    set_vertex_is_referenced<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->faces.ptr,
        F,
        cu_vertex_is_referenced
    );
    CUDA_CHECK(cudaGetLastError());

    // Get vertices map
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes,
        cu_vertex_is_referenced, V+1
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_vertex_is_referenced, V+1
    ));
    int new_num_vertices;
    CUDA_CHECK(cudaMemcpy(&new_num_vertices, cu_vertex_is_referenced + V, sizeof(int), cudaMemcpyDeviceToHost));

    // Compress vertices
    this->temp_storage.resize(new_num_vertices * sizeof(float3));
    compress_vertices_kernel<<<(V+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_vertex_is_referenced,
        this->vertices.ptr,
        V,
        reinterpret_cast<float3*>(this->temp_storage.ptr)
    );
    CUDA_CHECK(cudaGetLastError());
    swap_buffers(this->temp_storage, this->vertices);

    // Update faces
    remap_faces_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_vertex_is_referenced,
        F,
        this->faces.ptr
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(cu_vertex_is_referenced));

    // Delete all cached info since mesh has changed
    this->clear_cache();
}


static __global__ void sort_faces_kernel(
    int3* faces,
    const size_t F
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;

    int3 face = faces[tid];
    int tmp;

    // bubble sort 3 elements (x, y, z)
    if (face.x > face.y) { tmp = face.x; face.x = face.y; face.y = tmp; }
    if (face.y > face.z) { tmp = face.y; face.y = face.z; face.z = tmp; }
    if (face.x > face.y) { tmp = face.x; face.x = face.y; face.y = tmp; }

    faces[tid] = face;
}


static __global__ void select_first_in_each_group_kernel(
    const int3* faces,
    const size_t F,
    uint8_t* face_mask
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;
    if (tid == 0) {
        face_mask[tid] = 1;
    } else {
        int3 face = faces[tid];
        int3 prev_face = faces[tid-1];
        if (face.x == prev_face.x && face.y == prev_face.y && face.z == prev_face.z) {
            face_mask[tid] = 0;
        } else {
            face_mask[tid] = 1;
        }
    }
}


struct int3_decomposer
{
    __host__ __device__ ::cuda::std::tuple<int&, int&, int&> operator()(int3& key) const
    {
        return {key.x, key.y, key.z};
    }
};


void CuMesh::remove_duplicate_faces() {
    size_t F = this->faces.size;

    // Create a temporary sorted copy of faces for duplicate detection
    // Do NOT modify the original faces to preserve vertex order and normals
    int3 *cu_sorted_faces;
    CUDA_CHECK(cudaMalloc(&cu_sorted_faces, F * sizeof(int3)));
    CUDA_CHECK(cudaMemcpy(cu_sorted_faces, this->faces.ptr, F * sizeof(int3), cudaMemcpyDeviceToDevice));

    // Sort vertices within each face (in the temporary copy)
    sort_faces_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_sorted_faces,
        F
    );
    CUDA_CHECK(cudaGetLastError());

    // Sort all faces globally by their sorted vertex indices
    size_t temp_storage_bytes = 0;
    int *cu_sorted_face_indices;
    CUDA_CHECK(cudaMalloc(&cu_sorted_face_indices, F * sizeof(int)));
    arange_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(cu_sorted_face_indices, F);
    CUDA_CHECK(cudaGetLastError());

    int *cu_sorted_indices_output;
    int3 *cu_sorted_faces_output;
    CUDA_CHECK(cudaMalloc(&cu_sorted_indices_output, F * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_sorted_faces_output, F * sizeof(int3)));

    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes,
        cu_sorted_faces, cu_sorted_faces_output,
        cu_sorted_face_indices, cu_sorted_indices_output,
        F,
        int3_decomposer{}
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_sorted_faces, cu_sorted_faces_output,
        cu_sorted_face_indices, cu_sorted_indices_output,
        F,
        int3_decomposer{}
    ));
    CUDA_CHECK(cudaFree(cu_sorted_faces));
    CUDA_CHECK(cudaFree(cu_sorted_face_indices));

    // Select first in each group of duplicate faces (based on sorted faces)
    uint8_t* cu_face_mask_sorted;
    CUDA_CHECK(cudaMalloc(&cu_face_mask_sorted, F * sizeof(uint8_t)));
    select_first_in_each_group_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_sorted_faces_output,
        F,
        cu_face_mask_sorted
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(cu_sorted_faces_output));

    // Map the mask back to original face order using scatter
    // scatter: output[indices[i]] = values[i]
    // This maps: cu_face_mask_original[original_idx] = cu_face_mask_sorted[sorted_position]
    uint8_t* cu_face_mask_original;
    CUDA_CHECK(cudaMalloc(&cu_face_mask_original, F * sizeof(uint8_t)));
    scatter_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_sorted_indices_output,  // indices: sorted_position -> original_idx
        cu_face_mask_sorted,       // values: mask at sorted_position
        F,
        cu_face_mask_original      // output: mask at original position
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(cu_face_mask_sorted));
    CUDA_CHECK(cudaFree(cu_sorted_indices_output));

    // Select faces to keep (preserving original vertex order)
    this->_remove_faces(cu_face_mask_original);
    CUDA_CHECK(cudaFree(cu_face_mask_original));
}


static __global__ void mark_degenerate_faces_kernel(
    const float3* vertices,
    const int3* faces,
    const float abs_thresh,
    const float rel_thresh,
    const size_t F,
    uint8_t* face_mask
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;
    int3 face = faces[tid];

    // 1. Check if any vertex is duplicated
    if (face.x == face.y || face.y == face.z || face.z == face.x) {
        face_mask[tid] = 0;
        return;
    }

    // 2. Check if slim or zero area
    Vec3f v0 = Vec3f(vertices[face.x]);
    Vec3f v1 = Vec3f(vertices[face.y]);
    Vec3f v2 = Vec3f(vertices[face.z]);
    Vec3f e0 = v1 - v0;
    Vec3f e1 = v2 - v1;
    Vec3f e2 = v0 - v2;
    float max_edge_len = fmaxf(fmaxf(e0.norm(), e1.norm()), e2.norm());
    float area = e0.cross(e1).norm() / 2.0f;
    float thresh = fminf(rel_thresh * max_edge_len * max_edge_len, abs_thresh);
    if (area < thresh) {
        face_mask[tid] = 0;
        return;
    }

    face_mask[tid] = 1;
}


void CuMesh::remove_degenerate_faces(float abs_thresh, float rel_thresh) {
    size_t F = this->faces.size;

    uint8_t* cu_face_mask;
    CUDA_CHECK(cudaMalloc(&cu_face_mask, F * sizeof(uint8_t)));
    mark_degenerate_faces_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->vertices.ptr,
        this->faces.ptr,
        abs_thresh, rel_thresh,
        F,
        cu_face_mask
    );
    CUDA_CHECK(cudaGetLastError());

    this->_remove_faces(cu_face_mask);
    CUDA_CHECK(cudaFree(cu_face_mask));
}


static __global__ void compute_loop_boundary_lengths(
    const float3* vertices,
    const uint64_t* edges,
    const int* loop_boundaries,
    const size_t E,
    float* loop_boundary_lengths
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;
    uint64_t edge = edges[loop_boundaries[tid]];
    int e0 = int(edge & 0xFFFFFFFF);
    int e1 = int(edge >> 32);
    Vec3f v0 = Vec3f(vertices[e0]);
    Vec3f v1 = Vec3f(vertices[e1]);
    loop_boundary_lengths[tid] = (v1 - v0).norm();
}


static __device__ __forceinline__ int packed_edge_v0(uint64_t edge) {
    return int(edge >> 32);
}


static __device__ __forceinline__ int packed_edge_v1(uint64_t edge) {
    return int(edge & 0xFFFFFFFF);
}


static __device__ __forceinline__ bool face_has_directed_edge(
    const int3& face,
    const int u,
    const int v
) {
    return
        (face.x == u && face.y == v) ||
        (face.y == u && face.z == v) ||
        (face.z == u && face.x == v);
}


static __global__ void order_selected_boundary_loops_kernel(
    const int3* faces,
    const uint64_t* edges,
    const int* edge2face,
    const int* edge2face_offset,
    const int* loop_boundaries,
    const int* loop_boundaries_offset,
    const int num_loops,
    int* ordered_loop_vertices,
    uint8_t* loop_is_valid
) {
    if (blockIdx.x >= num_loops || threadIdx.x != 0) return;

    const int loop_id = blockIdx.x;
    const int start = loop_boundaries_offset[loop_id];
    const int end = loop_boundaries_offset[loop_id + 1];
    const int count = end - start;
    if (count < 3) {
        loop_is_valid[loop_id] = 0;
        return;
    }

    const int first_edge_id = loop_boundaries[start];
    const int3 first_face = faces[edge2face[edge2face_offset[first_edge_id]]];
    const uint64_t first_edge = edges[first_edge_id];
    int first_u = packed_edge_v0(first_edge);
    int first_v = packed_edge_v1(first_edge);
    if (!face_has_directed_edge(first_face, first_u, first_v)) {
        const int tmp = first_u;
        first_u = first_v;
        first_v = tmp;
    }

    ordered_loop_vertices[start + 0] = first_u;
    int prev_edge_id = first_edge_id;
    int current_vertex = first_v;
    bool valid = true;

    for (int i = 1; i < count; ++i) {
        ordered_loop_vertices[start + i] = current_vertex;

        int next_edge_id = -1;
        int next_vertex = -1;
        for (int j = start; j < end; ++j) {
            const int candidate_edge_id = loop_boundaries[j];
            if (candidate_edge_id == prev_edge_id) continue;

            const int3 candidate_face = faces[edge2face[edge2face_offset[candidate_edge_id]]];
            const uint64_t candidate_edge = edges[candidate_edge_id];
            int u = packed_edge_v0(candidate_edge);
            int v = packed_edge_v1(candidate_edge);
            if (!face_has_directed_edge(candidate_face, u, v)) {
                const int tmp = u;
                u = v;
                v = tmp;
            }

            if (u == current_vertex) {
                next_edge_id = candidate_edge_id;
                next_vertex = v;
                break;
            }
        }

        if (next_edge_id < 0) {
            valid = false;
            break;
        }

        prev_edge_id = next_edge_id;
        current_vertex = next_vertex;
    }

    if (valid && current_vertex != first_u) {
        valid = false;
    }
    if (!valid) {
        loop_is_valid[loop_id] = 0;
        return;
    }

    // Reverse the oriented boundary loop so the new patch uses each shared edge in the opposite direction.
    for (int i = 0; i < count / 2; ++i) {
        const int j = count - 1 - i;
        const int tmp = ordered_loop_vertices[start + i];
        ordered_loop_vertices[start + i] = ordered_loop_vertices[start + j];
        ordered_loop_vertices[start + j] = tmp;
    }

    loop_is_valid[loop_id] = 1;
}


static __device__ __forceinline__ int find_edge_index(
    const uint64_t* edges,
    const int num_edges,
    const uint64_t edge_key
) {
    int left = 0;
    int right = num_edges - 1;
    while (left <= right) {
        const int mid = left + ((right - left) >> 1);
        const uint64_t value = edges[mid];
        if (value == edge_key) return mid;
        if (value < edge_key) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1;
}


static __global__ void compute_ear_clip_triangle_counts_kernel(
    const uint64_t* edges,
    const int num_edges,
    const int* loop_boundaries_offset,
    const int* ordered_loop_vertices,
    int* loop_work_indices,
    const uint8_t* loop_is_valid,
    const int num_loops,
    int* loop_triangle_counts
) {
    if (blockIdx.x >= num_loops || threadIdx.x != 0) return;

    const int loop_id = blockIdx.x;
    loop_triangle_counts[loop_id] = 0;
    if (!loop_is_valid[loop_id]) return;

    const int loop_start = loop_boundaries_offset[loop_id];
    const int loop_end = loop_boundaries_offset[loop_id + 1];
    const int loop_size = loop_end - loop_start;
    if (loop_size < 3) return;

    for (int i = 0; i < loop_size; ++i) {
        loop_work_indices[loop_start + i] = i;
    }

    int work_size = loop_size;
    int tri_count = 0;
    while (work_size > 3) {
        int ear_i = -1;
        for (int i = 0; i < work_size; ++i) {
            const int prev_local = loop_work_indices[loop_start + (i - 1 + work_size) % work_size];
            const int curr_local = loop_work_indices[loop_start + i];
            const int next_local = loop_work_indices[loop_start + (i + 1) % work_size];
            const int a = ordered_loop_vertices[loop_start + prev_local];
            const int b = ordered_loop_vertices[loop_start + curr_local];
            const int c = ordered_loop_vertices[loop_start + next_local];
            if (a == b || b == c || c == a) continue;

            const uint64_t diagonal = (uint64_t(min(a, c)) << 32) | uint64_t(max(a, c));
            if (find_edge_index(edges, num_edges, diagonal) >= 0) continue;

            ear_i = i;
            break;
        }

        if (ear_i < 0) {
            loop_triangle_counts[loop_id] = 0;
            return;
        }

        for (int i = ear_i; i < work_size - 1; ++i) {
            loop_work_indices[loop_start + i] = loop_work_indices[loop_start + i + 1];
        }
        work_size--;
        tri_count++;
    }

    if (work_size == 3) {
        tri_count++;
    }
    loop_triangle_counts[loop_id] = tri_count;
}


static __global__ void triangulate_selected_boundary_loops_kernel(
    const uint64_t* edges,
    const int num_edges,
    const int* loop_boundaries_offset,
    const int* ordered_loop_vertices,
    int* loop_work_indices,
    const int* loop_triangle_offsets,
    const int* loop_triangle_counts,
    const uint8_t* loop_is_valid,
    const int num_loops,
    int3* output_faces
) {
    if (blockIdx.x >= num_loops || threadIdx.x != 0) return;

    const int loop_id = blockIdx.x;
    if (!loop_is_valid[loop_id]) return;

    const int loop_start = loop_boundaries_offset[loop_id];
    const int loop_end = loop_boundaries_offset[loop_id + 1];
    const int loop_size = loop_end - loop_start;
    const int tri_start = loop_triangle_offsets[loop_id];
    const int tri_count_target = loop_triangle_counts[loop_id];
    if (loop_size < 3 || tri_count_target == 0) return;

    for (int i = 0; i < loop_size; ++i) {
        loop_work_indices[loop_start + i] = i;
    }
    int work_size = loop_size;
    int tri_count = 0;
    while (work_size > 3) {
        int ear_i = -1;
        for (int i = 0; i < work_size; ++i) {
            const int prev_local = loop_work_indices[loop_start + (i - 1 + work_size) % work_size];
            const int curr_local = loop_work_indices[loop_start + i];
            const int next_local = loop_work_indices[loop_start + (i + 1) % work_size];
            const int a = ordered_loop_vertices[loop_start + prev_local];
            const int b = ordered_loop_vertices[loop_start + curr_local];
            const int c = ordered_loop_vertices[loop_start + next_local];
            if (a == b || b == c || c == a) continue;

            const uint64_t diagonal = (uint64_t(min(a, c)) << 32) | uint64_t(max(a, c));
            if (find_edge_index(edges, num_edges, diagonal) >= 0) continue;
            ear_i = i;
            break;
        }

        if (ear_i < 0) {
            return;
        }

        const int prev_local = loop_work_indices[loop_start + (ear_i - 1 + work_size) % work_size];
        const int curr_local = loop_work_indices[loop_start + ear_i];
        const int next_local = loop_work_indices[loop_start + (ear_i + 1) % work_size];
        int3 tri = {
            ordered_loop_vertices[loop_start + prev_local],
            ordered_loop_vertices[loop_start + curr_local],
            ordered_loop_vertices[loop_start + next_local]
        };
        output_faces[tri_start + tri_count] = tri;
        tri_count++;

        for (int i = ear_i; i < work_size - 1; ++i) {
            loop_work_indices[loop_start + i] = loop_work_indices[loop_start + i + 1];
        }
        work_size--;
    }

    if (work_size == 3) {
        int3 tri = {
            ordered_loop_vertices[loop_start + loop_work_indices[loop_start + 0]],
            ordered_loop_vertices[loop_start + loop_work_indices[loop_start + 1]],
            ordered_loop_vertices[loop_start + loop_work_indices[loop_start + 2]]
        };
        output_faces[tri_start + tri_count] = tri;
    }
}


struct LessThanOp {
    __device__ bool operator()(float a, float b) const {
        return a < b;
    }
};


void CuMesh::fill_holes(float max_hole_perimeter) {
    if (this->loop_boundaries.is_empty() || this->loop_boundaries_offset.is_empty()) {
        this->get_boundary_loops();
    }
    if (this->edge2face.is_empty() || this->edge2face_offset.is_empty()) {
        this->get_edge_face_adjacency();
    }

    size_t F = this->faces.size;
    size_t L = this->num_bound_loops;
    size_t E = this->loop_boundaries.size;

    // Early return if no boundary loops
    if (L == 0 || E == 0) {
        return;
    }

    // Compute loop boundary lengths
    float* cu_loop_boundary_lengths;
    CUDA_CHECK(cudaMalloc(&cu_loop_boundary_lengths, E * sizeof(float)));
    compute_loop_boundary_lengths<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->vertices.ptr,
        this->edges.ptr,
        this->loop_boundaries.ptr,
        E,
        cu_loop_boundary_lengths
    );
    CUDA_CHECK(cudaGetLastError());

    // Segment sum
    size_t temp_storage_bytes = 0;
    float *cu_bound_loop_perimeters;
    CUDA_CHECK(cudaMalloc(&cu_bound_loop_perimeters, L * sizeof(float)));
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(
        nullptr, temp_storage_bytes,
        cu_loop_boundary_lengths, cu_bound_loop_perimeters,
        L,
        this->loop_boundaries_offset.ptr,
        this->loop_boundaries_offset.ptr + 1
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_loop_boundary_lengths, cu_bound_loop_perimeters,
        L,
        this->loop_boundaries_offset.ptr,
        this->loop_boundaries_offset.ptr + 1
    ));
    CUDA_CHECK(cudaFree(cu_loop_boundary_lengths));

    // Mask small loops
    uint8_t* cu_bound_loop_mask;
    CUDA_CHECK(cudaMalloc(&cu_bound_loop_mask, L * sizeof(uint8_t)));
    compare_kernel<<<(L+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_bound_loop_perimeters,
        max_hole_perimeter,
        L,
        LessThanOp(),
        cu_bound_loop_mask
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(cu_bound_loop_perimeters));

    // Compress bound loops size
    int* cu_bound_loops_cnt;
    CUDA_CHECK(cudaMalloc(&cu_bound_loops_cnt, L * sizeof(int)));
    diff_kernel<<<(L+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->loop_boundaries_offset.ptr,
        L,
        cu_bound_loops_cnt
    );
    CUDA_CHECK(cudaGetLastError());
    int *cu_new_loop_boundaries_cnt, *cu_new_num_bound_loops;
    CUDA_CHECK(cudaMalloc(&cu_new_loop_boundaries_cnt, (L+1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_new_num_bound_loops, sizeof(int)));
    CUDA_CHECK(cudaMemset(cu_new_loop_boundaries_cnt, 0, (L + 1) * sizeof(int)));
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, temp_storage_bytes,
        cu_bound_loops_cnt, cu_bound_loop_mask, cu_new_loop_boundaries_cnt, cu_new_num_bound_loops,
        L
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_bound_loops_cnt, cu_bound_loop_mask, cu_new_loop_boundaries_cnt, cu_new_num_bound_loops,
        L
    ));
    int new_num_bound_loops;
    CUDA_CHECK(cudaMemcpy(&new_num_bound_loops, cu_new_num_bound_loops, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(cu_bound_loops_cnt));
    CUDA_CHECK(cudaFree(cu_new_num_bound_loops));
    if (new_num_bound_loops == 0) {
        CUDA_CHECK(cudaFree(cu_new_loop_boundaries_cnt));
        CUDA_CHECK(cudaFree(cu_bound_loop_mask));
        return;
    }

    // Get loop ids of loop boundaries
    int* cu_loop_bound_loop_ids;
    CUDA_CHECK(cudaMalloc(&cu_loop_bound_loop_ids, E * sizeof(int)));
    CUDA_CHECK(cudaMemset(cu_loop_bound_loop_ids, 0, E * sizeof(int)));
    if (L > 1) {
        set_flag_kernel<<<(L-1+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            this->loop_boundaries_offset.ptr + 1, L - 1,
            cu_loop_bound_loop_ids
        );
        CUDA_CHECK(cudaGetLastError());
    }
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, temp_storage_bytes,
        cu_loop_bound_loop_ids,
        E
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_loop_bound_loop_ids,
        E
    ));

    // Mask loop boundaries
    uint8_t* cu_loop_boundary_mask;
    CUDA_CHECK(cudaMalloc(&cu_loop_boundary_mask, E * sizeof(uint8_t)));
    index_kernel<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_bound_loop_mask,
        cu_loop_bound_loop_ids,
        E,
        cu_loop_boundary_mask
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(cu_bound_loop_mask));
    CUDA_CHECK(cudaFree(cu_loop_bound_loop_ids));

    // Compress loop boundaries
    int *cu_new_loop_boundaries, *cu_new_num_loop_boundaries;
    CUDA_CHECK(cudaMalloc(&cu_new_loop_boundaries, E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_new_num_loop_boundaries, sizeof(int)));
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, temp_storage_bytes,
        this->loop_boundaries.ptr, cu_loop_boundary_mask, cu_new_loop_boundaries, cu_new_num_loop_boundaries,
        E
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        this->loop_boundaries.ptr, cu_loop_boundary_mask, cu_new_loop_boundaries, cu_new_num_loop_boundaries,
        E
    ));
    int new_num_loop_boundaries;
    CUDA_CHECK(cudaMemcpy(&new_num_loop_boundaries, cu_new_num_loop_boundaries, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(cu_new_num_loop_boundaries));
    CUDA_CHECK(cudaFree(cu_loop_boundary_mask));

    // Reconstruct the selected loop offsets.
    int* cu_new_loop_boundaries_offset;
    CUDA_CHECK(cudaMalloc(&cu_new_loop_boundaries_offset, (new_num_bound_loops + 1) * sizeof(int)));
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes,
        cu_new_loop_boundaries_cnt, cu_new_loop_boundaries_offset,
        new_num_bound_loops + 1
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_new_loop_boundaries_cnt, cu_new_loop_boundaries_offset,
        new_num_bound_loops + 1
    ));
    int* cu_ordered_loop_vertices;
    uint8_t* cu_loop_is_valid;
    CUDA_CHECK(cudaMalloc(&cu_ordered_loop_vertices, new_num_loop_boundaries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_loop_is_valid, new_num_bound_loops * sizeof(uint8_t)));
    order_selected_boundary_loops_kernel<<<new_num_bound_loops, 1>>>(
        this->faces.ptr,
        this->edges.ptr,
        this->edge2face.ptr,
        this->edge2face_offset.ptr,
        cu_new_loop_boundaries,
        cu_new_loop_boundaries_offset,
        new_num_bound_loops,
        cu_ordered_loop_vertices,
        cu_loop_is_valid
    );
    CUDA_CHECK(cudaGetLastError());

    int* cu_loop_triangle_counts;
    int* cu_loop_triangle_offsets;
    int* cu_loop_work_indices;
    CUDA_CHECK(cudaMalloc(&cu_loop_triangle_counts, (new_num_bound_loops + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_loop_triangle_offsets, (new_num_bound_loops + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_loop_work_indices, new_num_loop_boundaries * sizeof(int)));
    CUDA_CHECK(cudaMemset(cu_loop_triangle_counts, 0, (new_num_bound_loops + 1) * sizeof(int)));
    compute_ear_clip_triangle_counts_kernel<<<new_num_bound_loops, 1>>>(
        this->edges.ptr,
        this->edges.size,
        cu_new_loop_boundaries_offset,
        cu_ordered_loop_vertices,
        cu_loop_work_indices,
        cu_loop_is_valid,
        new_num_bound_loops,
        cu_loop_triangle_counts
    );
    CUDA_CHECK(cudaGetLastError());

    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes,
        cu_loop_triangle_counts, cu_loop_triangle_offsets,
        new_num_bound_loops + 1
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_loop_triangle_counts, cu_loop_triangle_offsets,
        new_num_bound_loops + 1
    ));

    int new_num_faces_to_add = 0;
    CUDA_CHECK(cudaMemcpy(
        &new_num_faces_to_add,
        cu_loop_triangle_offsets + new_num_bound_loops,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    if (new_num_faces_to_add == 0) {
        CUDA_CHECK(cudaFree(cu_new_loop_boundaries));
        CUDA_CHECK(cudaFree(cu_new_loop_boundaries_cnt));
        CUDA_CHECK(cudaFree(cu_new_loop_boundaries_offset));
        CUDA_CHECK(cudaFree(cu_ordered_loop_vertices));
        CUDA_CHECK(cudaFree(cu_loop_is_valid));
        CUDA_CHECK(cudaFree(cu_loop_triangle_counts));
        CUDA_CHECK(cudaFree(cu_loop_triangle_offsets));
        CUDA_CHECK(cudaFree(cu_loop_work_indices));
        return;
    }

    int3* cu_added_faces;
    CUDA_CHECK(cudaMalloc(&cu_added_faces, new_num_faces_to_add * sizeof(int3)));
    triangulate_selected_boundary_loops_kernel<<<new_num_bound_loops, 1>>>(
        this->edges.ptr,
        this->edges.size,
        cu_new_loop_boundaries_offset,
        cu_ordered_loop_vertices,
        cu_loop_work_indices,
        cu_loop_triangle_offsets,
        cu_loop_triangle_counts,
        cu_loop_is_valid,
        new_num_bound_loops,
        cu_added_faces
    );
    CUDA_CHECK(cudaGetLastError());

    this->faces.extend(new_num_faces_to_add);
    CUDA_CHECK(cudaMemcpy(
        this->faces.ptr + F,
        cu_added_faces,
        new_num_faces_to_add * sizeof(int3),
        cudaMemcpyDeviceToDevice
    ));

    CUDA_CHECK(cudaFree(cu_new_loop_boundaries));
    CUDA_CHECK(cudaFree(cu_new_loop_boundaries_cnt));
    CUDA_CHECK(cudaFree(cu_new_loop_boundaries_offset));
    CUDA_CHECK(cudaFree(cu_ordered_loop_vertices));
    CUDA_CHECK(cudaFree(cu_loop_is_valid));
    CUDA_CHECK(cudaFree(cu_loop_triangle_counts));
    CUDA_CHECK(cudaFree(cu_loop_triangle_offsets));
    CUDA_CHECK(cudaFree(cu_loop_work_indices));
    CUDA_CHECK(cudaFree(cu_added_faces));

    // Delete all cached info since mesh has changed
    this->clear_cache();
}


static __global__ void construct_vertex_adj_pairs_kernel(
    const int2* manifold_face_adj,
    const int3* faces,
    int2* vertex_adj_pairs,
    const size_t M
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M) return;

    const int2 adj_faces = manifold_face_adj[tid];
    const int3 face1 = faces[adj_faces.x];
    const int3 face2 = faces[adj_faces.y];

    const int v1[3] = {face1.x, face1.y, face1.z};

    int shared_local_indices1[2] = {0, 0};
    int shared_local_indices2[2] = {0, 0};
    int found_count = 0;

    for (int i = 0; i < 3; ++i) {
        if (v1[i] == face2.x) {
            shared_local_indices1[found_count] = i;
            shared_local_indices2[found_count] = 0;
            found_count++;
        } else if (v1[i] == face2.y) {
            shared_local_indices1[found_count] = i;
            shared_local_indices2[found_count] = 1;
            found_count++;
        } else if (v1[i] == face2.z) {
            shared_local_indices1[found_count] = i;
            shared_local_indices2[found_count] = 2;
            found_count++;
        }
        if (found_count == 2) {
            break;
        }
    }

    // Only process if we found exactly 2 shared vertices (valid manifold edge)
    if (found_count == 2) {
        vertex_adj_pairs[2 * tid + 0] = make_int2(
            3 * adj_faces.x + shared_local_indices1[0],
            3 * adj_faces.y + shared_local_indices2[0]
        );
        vertex_adj_pairs[2 * tid + 1] = make_int2(
            3 * adj_faces.x + shared_local_indices1[1],
            3 * adj_faces.y + shared_local_indices2[1]
        );
    } else {
        // Invalid edge, set to identity mapping
        vertex_adj_pairs[2 * tid + 0] = make_int2(3 * adj_faces.x, 3 * adj_faces.x);
        vertex_adj_pairs[2 * tid + 1] = make_int2(3 * adj_faces.y, 3 * adj_faces.y);
    }
}


static __global__ void index_vertice_kernel(
    const int* vertex_ids,
    const int3* faces,
    const float3* vertices,
    const size_t V,
    float3* new_vertices
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= V) return;
    const int vid = vertex_ids[tid];
    const int3 face = faces[vid / 3];
    const int f[3] = {face.x, face.y, face.z};
    new_vertices[tid] = vertices[f[vid % 3]];
}


void CuMesh::repair_non_manifold_edges(){
    // Always recompute manifold_face_adj to ensure it's up to date
    // especially after operations like simplify() that modify the mesh
    this->get_manifold_face_adjacency();

    size_t F = this->faces.size;
    size_t M = this->manifold_face_adj.size;

    // Construct vertex adjacency pairs with manifold edges
    int2* cu_vertex_adj_pairs;
    CUDA_CHECK(cudaMalloc(&cu_vertex_adj_pairs, 2*M*sizeof(int2)));
    construct_vertex_adj_pairs_kernel<<<(M+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->manifold_face_adj.ptr,
        this->faces.ptr,
        cu_vertex_adj_pairs,
        M
    );
    CUDA_CHECK(cudaGetLastError());

    // Iterative Hook and Compress
    int* cu_vertex_ids;
    CUDA_CHECK(cudaMalloc(&cu_vertex_ids, 3 * F * sizeof(int)));
    arange_kernel<<<(3*F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(cu_vertex_ids, 3 * F);
    CUDA_CHECK(cudaGetLastError());
    int* cu_end_flag; int h_end_flag;
    CUDA_CHECK(cudaMalloc(&cu_end_flag, sizeof(int)));
    do {
        h_end_flag = 1;
        CUDA_CHECK(cudaMemcpy(cu_end_flag, &h_end_flag, sizeof(int), cudaMemcpyHostToDevice));

        // Hook
        hook_edges_kernel<<<(2*M+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            cu_vertex_adj_pairs,
            2 * M,
            cu_vertex_ids,
            cu_end_flag
        );
        CUDA_CHECK(cudaGetLastError());

        // Compress
        compress_components_kernel<<<(3*F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            cu_vertex_ids,
            3 * F
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&h_end_flag, cu_end_flag, sizeof(int), cudaMemcpyDeviceToHost));
    } while (h_end_flag == 0);
    CUDA_CHECK(cudaFree(cu_end_flag));
    CUDA_CHECK(cudaFree(cu_vertex_adj_pairs));

    // Construct new faces
    int* cu_new_vertices_ids;
    CUDA_CHECK(cudaMalloc(&cu_new_vertices_ids, 3 * F * sizeof(int)));
    int new_V = compress_ids(cu_vertex_ids, 3 * F, this->cub_temp_storage, cu_new_vertices_ids);
    float3* cu_new_vertices;
    CUDA_CHECK(cudaMalloc(&cu_new_vertices, new_V * sizeof(float3)));
    index_vertice_kernel<<<(new_V+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_new_vertices_ids,
        this->faces.ptr,
        this->vertices.ptr,
        new_V,
        cu_new_vertices
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(cu_new_vertices_ids));
    this->vertices.resize(new_V);
    CUDA_CHECK(cudaMemcpy(this->vertices.ptr, cu_new_vertices, new_V * sizeof(float3), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(cu_new_vertices));
    this->faces.resize(F);
    copy_T_to_T3_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(cu_vertex_ids, F, this->faces.ptr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(cu_vertex_ids));

    // Delete all cached info since mesh has changed
    this->clear_cache();
}


/**
 * Mark faces to remove for non-manifold edges
 * For each non-manifold edge (shared by >2 faces), only keep the first 2 faces
 *
 * @param edge2face: edge to face adjacency
 * @param edge2face_offset: edge to face adjacency offset
 * @param edge2face_cnt: number of faces per edge
 * @param E: number of edges
 * @param face_keep_mask: output mask (1 = keep, 0 = remove)
 */
static __global__ void mark_non_manifold_faces_kernel(
    const int* edge2face,
    const int* edge2face_offset,
    const int* edge2face_cnt,
    const size_t E,
    uint8_t* face_keep_mask
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;

    // Only process non-manifold edges (cnt > 2)
    int cnt = edge2face_cnt[tid];
    if (cnt <= 2) return;

    // Mark faces beyond the first 2 for removal
    int start = edge2face_offset[tid];
    for (int i = 2; i < cnt; i++) {
        int face_idx = edge2face[start + i];
        face_keep_mask[face_idx] = 0;
    }
}


void CuMesh::remove_non_manifold_faces() {
    // Get edge-face adjacency information
    if (this->edge2face.is_empty() || this->edge2face_offset.is_empty()) {
        this->get_edge_face_adjacency();
    }

    size_t F = this->faces.size;
    size_t E = this->edges.size;

    if (F == 0 || E == 0) return;

    // Initialize face mask (1 = keep all faces initially)
    uint8_t* cu_face_keep_mask;
    CUDA_CHECK(cudaMalloc(&cu_face_keep_mask, F * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(cu_face_keep_mask, 1, F * sizeof(uint8_t)));

    // Mark faces on non-manifold edges for removal
    mark_non_manifold_faces_kernel<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->edge2face.ptr,
        this->edge2face_offset.ptr,
        this->edge2face_cnt.ptr,
        E,
        cu_face_keep_mask
    );
    CUDA_CHECK(cudaGetLastError());

    // Remove marked faces
    this->_remove_faces(cu_face_keep_mask);
    CUDA_CHECK(cudaFree(cu_face_keep_mask));

    // Clear cache since mesh has changed
    this->clear_cache();
}


struct GreaterThanOrEqualToOp {
    __device__ __forceinline__ bool operator()(const float& a, const float& b) const {
        return a >= b;
    }
};


void CuMesh::remove_small_connected_components(float min_area) {
    if (this->conn_comp_ids.is_empty()) {
        this->get_connected_components();
    }
    if (this->face_areas.is_empty()) {
        this->compute_face_areas();
    }
    size_t F = this->faces.size;
    if (F == 0) return;

    // 1. Sort face areas based on their connected component ID.
    // This groups all faces of the same component together.
    size_t temp_storage_bytes = 0;
    int *cu_sorted_conn_comp_ids;
    float *cu_sorted_face_areas;
    CUDA_CHECK(cudaMalloc(&cu_sorted_conn_comp_ids, F * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_sorted_face_areas, F * sizeof(float)));
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes,
        this->conn_comp_ids.ptr, cu_sorted_conn_comp_ids,
        this->face_areas.ptr, cu_sorted_face_areas,
        F
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        this->conn_comp_ids.ptr, cu_sorted_conn_comp_ids,
        this->face_areas.ptr, cu_sorted_face_areas,
        F
    ));

    // 2. Find unique components and get the number of faces in each.
    int* cu_conn_comp_num_faces;
    int* cu_num_conn_comps;
    int* cu_unique_conn_comp_ids; // Not needed, but we need to pass a valid pointer.
    CUDA_CHECK(cudaMalloc(&cu_conn_comp_num_faces, (this->num_conn_comps + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_num_conn_comps, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_unique_conn_comp_ids, (this->num_conn_comps + 1) * sizeof(int)));
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
        nullptr, temp_storage_bytes,
        cu_sorted_conn_comp_ids, cu_unique_conn_comp_ids,
        cu_conn_comp_num_faces, cu_num_conn_comps,
        F
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_sorted_conn_comp_ids, cu_unique_conn_comp_ids,
        cu_conn_comp_num_faces, cu_num_conn_comps,
        F
    ));
    int num_conn_comps;
    CUDA_CHECK(cudaMemcpy(&num_conn_comps, cu_num_conn_comps, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(cu_num_conn_comps));
    CUDA_CHECK(cudaFree(cu_sorted_conn_comp_ids));
    CUDA_CHECK(cudaFree(cu_unique_conn_comp_ids));

    // 3. Compute the total area for each connected component via segmented reduction.
    int* cu_conn_comp_offsets;
    CUDA_CHECK(cudaMalloc(&cu_conn_comp_offsets, (num_conn_comps + 1) * sizeof(int)));
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes,
        cu_conn_comp_num_faces, cu_conn_comp_offsets,
        num_conn_comps + 1
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_conn_comp_num_faces, cu_conn_comp_offsets,
        num_conn_comps + 1
    ));
    CUDA_CHECK(cudaFree(cu_conn_comp_num_faces));

    float *cu_conn_comp_areas;
    CUDA_CHECK(cudaMalloc(&cu_conn_comp_areas, num_conn_comps * sizeof(float)));
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(
        nullptr, temp_storage_bytes,
        cu_sorted_face_areas, cu_conn_comp_areas,
        num_conn_comps,
        cu_conn_comp_offsets,
        cu_conn_comp_offsets + 1
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_sorted_face_areas, cu_conn_comp_areas,
        num_conn_comps,
        cu_conn_comp_offsets,
        cu_conn_comp_offsets + 1
    ));
    CUDA_CHECK(cudaFree(cu_sorted_face_areas));
    CUDA_CHECK(cudaFree(cu_conn_comp_offsets));

    // 4. Create a "keep" mask for components with area >= min_area.
    uint8_t* cu_comp_keep_mask;
    CUDA_CHECK(cudaMalloc(&cu_comp_keep_mask, num_conn_comps * sizeof(uint8_t)));
    compare_kernel<<<(num_conn_comps+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_conn_comp_areas,
        min_area,
        num_conn_comps,
        GreaterThanOrEqualToOp(),
        cu_comp_keep_mask
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(cu_conn_comp_areas));

    // 5. Propagate the component "keep" mask to every face.
    uint8_t* cu_face_keep_mask;
    CUDA_CHECK(cudaMalloc(&cu_face_keep_mask, F * sizeof(uint8_t)));
    // Use an index_kernel (gather operation)
    index_kernel<<<(F + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_comp_keep_mask,      // Source array
        this->conn_comp_ids.ptr, // Indices to gather from
        F,
        cu_face_keep_mask       // Destination array
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(cu_comp_keep_mask));

    // 6. Select the faces to keep and update the mesh.
    this->_remove_faces(cu_face_keep_mask);
    CUDA_CHECK(cudaFree(cu_face_keep_mask));
}


static __global__ void hook_edges_with_orientation_kernel(
    const int2* adj,
    const uint8_t* flipped,
    const int M,
    int* conn_comp_ids,
    int* end_flag
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M) return;

    // get adjacent faces
    int f0 = adj[tid].x;
    int f1 = adj[tid].y;
    uint8_t is_flipped = flipped[tid];

    // union
    // find roots
    int root0 = conn_comp_ids[f0] >> 1;
    int flip0 = conn_comp_ids[f0] & 1;
    while (root0 != (conn_comp_ids[root0] >> 1)) {
        flip0 ^= conn_comp_ids[root0] & 1;
        root0 = conn_comp_ids[root0] >> 1;
    }
    int root1 = conn_comp_ids[f1] >> 1;
    int flip1 = conn_comp_ids[f1] & 1;
    while (root1 != (conn_comp_ids[root1] >> 1)) {
        flip1 ^= conn_comp_ids[root1] & 1;
        root1 = conn_comp_ids[root1] >> 1;
    }

    if (root0 == root1) return;

    int high = max(root0, root1);
    int low = min(root0, root1);
    atomicMin(&conn_comp_ids[high], (low << 1) | (is_flipped ^ flip0 ^ flip1));
    *end_flag = 0;
}


static __global__ void compress_components_with_orientation_kernel(
    int* conn_comp_ids,
    const int F
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;

    int p = conn_comp_ids[tid] >> 1;
    int f = conn_comp_ids[tid] & 1;
    while (p != (conn_comp_ids[p] >> 1)) {
        f ^= conn_comp_ids[p] & 1;
        p = conn_comp_ids[p] >> 1;
    }
    conn_comp_ids[tid] = (p << 1) | f;
}


static __global__ void get_flip_flags_kernel(
    const int2* manifold_face_adj,
    const int3* faces,
    const int M,
    uint8_t* flipped
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M) return;

    const int2 adj_faces = manifold_face_adj[tid];
    const int3 face1 = faces[adj_faces.x];
    const int3 face2 = faces[adj_faces.y];

    const int v1[3] = {face1.x, face1.y, face1.z};

    int shared_local_indices1[2];
    int shared_local_indices2[2];
    int found_count = 0;

    for (int i = 0; i < 3; ++i) {
        if (v1[i] == face2.x) {
            shared_local_indices1[found_count] = i;
            shared_local_indices2[found_count] = 0;
            found_count++;
        } else if (v1[i] == face2.y) {
            shared_local_indices1[found_count] = i;
            shared_local_indices2[found_count] = 1;
            found_count++;
        } else if (v1[i] == face2.z) {
            shared_local_indices1[found_count] = i;
            shared_local_indices2[found_count] = 2;
            found_count++;
        }
        if (found_count == 2) {
            break;
        }
    }

    int direction1 = (shared_local_indices1[1] - shared_local_indices1[0] + 3) % 3;
    int direction2 = (shared_local_indices2[1] - shared_local_indices2[0] + 3) % 3;
    flipped[tid] = (direction1 == direction2) ? 1 : 0;
}


static __global__ void inplace_flip_faces_with_flags_kernel(
    int3* faces,
    const int* conn_comp_with_flip,
    const int F
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;

    int is_flipped = conn_comp_with_flip[tid] & 1;
    if (is_flipped) {
        int3 face = faces[tid];
        faces[tid] = make_int3(face.x, face.z, face.y);
    }
}


void CuMesh::unify_face_orientations() {
    if (this->manifold_face_adj.is_empty()) {
        this->get_manifold_face_adjacency();
    }

    // 1. Compute the flipped flag for each edge.
    uint8_t* cu_flipped;
    CUDA_CHECK(cudaMalloc(&cu_flipped, this->manifold_face_adj.size * sizeof(uint8_t)));
    get_flip_flags_kernel<<<(this->manifold_face_adj.size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->manifold_face_adj.ptr,
        this->faces.ptr,
        this->manifold_face_adj.size,
        cu_flipped
    );
    CUDA_CHECK(cudaGetLastError());

    // 2. Hook edges with flipped flag.
    int* conn_comp_with_flip;
    CUDA_CHECK(cudaMalloc(&conn_comp_with_flip, this->faces.size * sizeof(int)));
    arange_kernel<<<(this->faces.size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(conn_comp_with_flip, this->faces.size, 2);
    CUDA_CHECK(cudaGetLastError());
    int* cu_end_flag; int h_end_flag;
    CUDA_CHECK(cudaMalloc(&cu_end_flag, sizeof(int)));
    do {
        h_end_flag = 1;
        CUDA_CHECK(cudaMemcpy(cu_end_flag, &h_end_flag, sizeof(int), cudaMemcpyHostToDevice));

        // Hook
        hook_edges_with_orientation_kernel<<<(this->manifold_face_adj.size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            this->manifold_face_adj.ptr,
            cu_flipped,
            this->manifold_face_adj.size,
            conn_comp_with_flip,
            cu_end_flag
        );
        CUDA_CHECK(cudaGetLastError());

        // Compress
        compress_components_with_orientation_kernel<<<(this->faces.size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            conn_comp_with_flip,
            this->faces.size
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&h_end_flag, cu_end_flag, sizeof(int), cudaMemcpyDeviceToHost));
    } while (h_end_flag == 0);
    CUDA_CHECK(cudaFree(cu_end_flag));

    // 3. Flip the orientation of the faces.
    inplace_flip_faces_with_flags_kernel<<<(this->faces.size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->faces.ptr,
        conn_comp_with_flip,
        this->faces.size
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(cu_flipped));
    CUDA_CHECK(cudaFree(conn_comp_with_flip));
}


} // namespace cumesh
