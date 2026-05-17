#include <torch/extension.h>
#include "hash/api.h"
#include "cumesh.h"
#include "remesh/api.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Hash functions
    m.def("hashmap_insert_cuda", &cumesh::hashmap_insert_cuda);
    m.def("hashmap_lookup_cuda", &cumesh::hashmap_lookup_cuda);
    m.def("hashmap_insert_3d_cuda", &cumesh::hashmap_insert_3d_cuda);
    m.def("hashmap_lookup_3d_cuda", &cumesh::hashmap_lookup_3d_cuda);
    m.def("hashmap_insert_3d_idx_as_val_cuda", &cumesh::hashmap_insert_3d_idx_as_val_cuda);

    /* CUMESH */
    py::class_<cumesh::CuMesh>(m, "CuMesh")
        .def(py::init<>())
        .def("num_vertices", &cumesh::CuMesh::num_vertices)
        .def("num_faces", &cumesh::CuMesh::num_faces)
        .def("num_edges", &cumesh::CuMesh::num_edges)
        .def("num_boundaries", &cumesh::CuMesh::num_boundaries)
        .def("num_conneted_components", &cumesh::CuMesh::num_conneted_components)
        .def("num_boundary_conneted_components", &cumesh::CuMesh::num_boundary_conneted_components)
        .def("num_boundary_loops", &cumesh::CuMesh::num_boundary_loops)
        .def("clear_cache", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); self.clear_cache(); })
        .def("init", &cumesh::CuMesh::init)
        .def("read", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); return self.read(); })
        .def("read_face_normals", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); return self.read_face_normals(); })
        .def("read_vertex_normals", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); return self.read_vertex_normals(); })
        .def("read_edges", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); return self.read_edges(); })
        .def("read_boundaries", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); return self.read_boundaries(); })
        .def("read_manifold_face_adjacency", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); return self.read_manifold_face_adjacency(); })
        .def("read_manifold_boundary_adjacency", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); return self.read_manifold_boundary_adjacency(); })
        .def("read_connected_components", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); return self.read_connected_components(); })
        .def("read_boundary_connected_components", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); return self.read_boundary_connected_components(); })
        .def("read_boundary_loops", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); return self.read_boundary_loops(); })
        .def("read_all_cache", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); return self.read_all_cache(); })
        .def("compute_face_normals", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); self.compute_face_normals(); })
        .def("compute_vertex_normals", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); self.compute_vertex_normals(); })
        .def("get_vertex_face_adjacency", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); self.get_vertex_face_adjacency(); })
        .def("get_edges", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); self.get_edges(); })
        .def("get_edge_face_adjacency", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); self.get_edge_face_adjacency(); })
        .def("get_vertex_edge_adjacency", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); self.get_vertex_edge_adjacency(); })
        .def("get_boundary_info", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); self.get_boundary_info(); })
        .def("get_vertex_boundary_adjacency", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); self.get_vertex_boundary_adjacency(); })
        .def("get_vertex_is_manifold", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); self.get_vertex_is_manifold(); })
        .def("get_manifold_face_adjacency", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); self.get_manifold_face_adjacency(); })
        .def("get_manifold_boundary_adjacency", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); self.get_manifold_boundary_adjacency(); })
        .def("get_connected_components", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); self.get_connected_components(); })
        .def("get_boundary_connected_components", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); self.get_boundary_connected_components(); })
        .def("get_boundary_loops", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); self.get_boundary_loops(); })
        .def("remove_faces", [](cumesh::CuMesh& self, torch::Tensor& face_mask) { auto device_guard = self.guard_device(); self.remove_faces(face_mask); })
        .def("remove_unreferenced_vertices", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); self.remove_unreferenced_vertices(); })
        .def("remove_duplicate_faces", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); self.remove_duplicate_faces(); })
        .def("remove_degenerate_faces", [](cumesh::CuMesh& self, float abs_thresh, float rel_thresh) { auto device_guard = self.guard_device(); self.remove_degenerate_faces(abs_thresh, rel_thresh); })
        .def("fill_holes", [](cumesh::CuMesh& self, float max_hole_perimeter) { auto device_guard = self.guard_device(); self.fill_holes(max_hole_perimeter); })
        .def("repair_non_manifold_edges", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); self.repair_non_manifold_edges(); })
        .def("remove_non_manifold_faces", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); self.remove_non_manifold_faces(); })
        .def("remove_small_connected_components", [](cumesh::CuMesh& self, float min_area) { auto device_guard = self.guard_device(); self.remove_small_connected_components(min_area); })
        .def("unify_face_orientations", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); self.unify_face_orientations(); })
        .def("simplify_step", [](cumesh::CuMesh& self, float lambda_edge_length, float lambda_skinny, float threshold, bool timing) { auto device_guard = self.guard_device(); return self.simplify_step(lambda_edge_length, lambda_skinny, threshold, timing); })
        .def("compute_charts", [](cumesh::CuMesh& self, float threshold_cone_half_angle_rad, int refine_iterations, int global_iterations, float smooth_strength, float area_penalty_weight, float perimeter_area_ratio_weight) { auto device_guard = self.guard_device(); self.compute_charts(threshold_cone_half_angle_rad, refine_iterations, global_iterations, smooth_strength, area_penalty_weight, perimeter_area_ratio_weight); })
        .def("read_atlas_charts", [](cumesh::CuMesh& self) { auto device_guard = self.guard_device(); return self.read_atlas_charts(); });

    // Remeshing functions
    m.def("get_sparse_voxel_grid_active_vertices", &cumesh::get_sparse_voxel_grid_active_vertices);
    m.def("simple_dual_contour", &cumesh::simple_dual_contour);
}
