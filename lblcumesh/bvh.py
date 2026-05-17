import torch

# CUDA extension
from . import _cubvh as _backend

_sdf_mode_to_id = {
    'watertight': 0,
    'raystab': 1,
}

def _cuda_device(*tensors, device=None):
    devices = [tensor.device for tensor in tensors if torch.is_tensor(tensor) and tensor.is_cuda]
    assert all(device == devices[0] for device in devices), "CUDA tensors must be on the same device"
    if device is not None:
        device = torch.device(device)
        assert device.type == "cuda" and device.index is not None, device
        if devices:
            assert devices[0] == device, f"CUDA tensors must be on {device}, got {devices[0]}"
        return device
    assert devices, "CPU-only cuBVH construction requires an explicit CUDA device"
    return devices[0]

def _to_device(tensor, device, dtype):
    assert torch.is_tensor(tensor)
    assert not tensor.is_cuda or tensor.device == device
    if not tensor.is_cuda:
        tensor = tensor.to(device)
    return tensor.to(dtype=dtype).contiguous()

class cuBVH():
    def __init__(self, vertices, triangles, device=None):
        # vertices: np.ndarray, [N, 3]
        # triangles: np.ndarray, [M, 3]

        self.device = _cuda_device(vertices, triangles, device=device)
        if torch.is_tensor(vertices): vertices = vertices.cpu().numpy()
        if torch.is_tensor(triangles): triangles = triangles.cpu().numpy()

        # check inputs
        assert triangles.shape[0] > 8, "BVH needs at least 8 triangles."
        
        # implementation
        with torch.cuda.device(self.device):
            self.impl = _backend.create_cuBVH(vertices, triangles, self.device.index)

    def ray_trace(self, rays_o, rays_d):
        # rays_o: torch.Tensor, float, [N, 3]
        # rays_d: torch.Tensor, float, [N, 3]

        rays_o = _to_device(rays_o, self.device, torch.float32)
        rays_d = _to_device(rays_d, self.device, torch.float32)

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)

        N = rays_o.shape[0]

        # init output buffers
        positions = torch.empty(N, 3, dtype=torch.float32, device=rays_o.device)
        face_id = torch.empty(N, dtype=torch.int64, device=rays_o.device)
        depth = torch.empty(N, dtype=torch.float32, device=rays_o.device)
        
        with torch.cuda.device(self.device):
            self.impl.ray_trace(rays_o, rays_d, positions, face_id, depth) # [N, 3]

        positions = positions.view(*prefix, 3)
        face_id = face_id.view(*prefix)
        depth = depth.view(*prefix)

        return positions, face_id, depth

    def unsigned_distance(self, positions, return_uvw=False):
        # positions: torch.Tensor, float, [N, 3]

        positions = _to_device(positions, self.device, torch.float32)

        prefix = positions.shape[:-1]
        positions = positions.view(-1, 3)

        N = positions.shape[0]

        # init output buffers
        distances = torch.empty(N, dtype=torch.float32, device=positions.device)
        face_id = torch.empty(N, dtype=torch.int64, device=positions.device)

        if return_uvw:
            uvw = torch.empty(N, 3, dtype=torch.float32, device=positions.device)
        else:
            uvw = None
        
        with torch.cuda.device(self.device):
            self.impl.unsigned_distance(positions, distances, face_id, uvw) # [N, 3]

        distances = distances.view(*prefix)
        face_id = face_id.view(*prefix)
        if uvw is not None:
            uvw = uvw.view(*prefix, 3)

        return distances, face_id, uvw


    def signed_distance(self, positions, return_uvw=False, mode='watertight'):
        # positions: torch.Tensor, float, [N, 3]

        positions = _to_device(positions, self.device, torch.float32)

        prefix = positions.shape[:-1]
        positions = positions.view(-1, 3)

        N = positions.shape[0]

        # init output buffers
        distances = torch.empty(N, dtype=torch.float32, device=positions.device)
        face_id = torch.empty(N, dtype=torch.int64, device=positions.device)

        if return_uvw:
            uvw = torch.empty(N, 3, dtype=torch.float32, device=positions.device)
        else:
            uvw = None
        
        with torch.cuda.device(self.device):
            self.impl.signed_distance(positions, distances, face_id, uvw, _sdf_mode_to_id[mode]) # [N, 3]

        distances = distances.view(*prefix)
        face_id = face_id.view(*prefix)
        if uvw is not None:
            uvw = uvw.view(*prefix, 3)

        return distances, face_id, uvw
