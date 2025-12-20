import os
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageSequence, ImageOps
from pathlib import Path
import numpy as np
import json
import trimesh as Trimesh
from tqdm import tqdm
import cumesh

import folder_paths

import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar, common_upscale
import comfy.utils

script_directory = os.path.dirname(os.path.abspath(__file__))
comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]
    
tensor2pil = transforms.ToPILImage()  
    
def convert_tensor_images_to_pil(images):
    pil_array = []
    
    for image in images:
        pil_array.append(tensor2pil(image))
        
    return pil_array     

def TrimeshToCuMesh(trimesh):
    vertices = torch.from_numpy(trimesh.vertices).float()
    faces = torch.from_numpy(trimesh.faces).int()
    
    vertices = vertices.cuda()
    faces = faces.cuda()        
    
    mesh = cumesh.CuMesh()
    
    mesh.init(vertices, faces)
    return mesh

class CuMeshUVUnWrap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("TRIMESH", )
    RETURN_NAMES = ("trimesh", )
    FUNCTION = "process"
    CATEGORY = "CuMeshWrapper"

    def process(self, trimesh):
        cumesh = TrimeshToCuMesh(trimesh)
        new_vertices, new_faces, uv = cumesh.uv_unwrap(verbose=True)
        trimesh.vertices = new_vertices.cpu().numpy()
        trimesh.faces = new_faces.cpu().numpy()
        trimesh.visual.uv = uv.cpu().numpy()
        
        return (trimesh,)
        
class CuMeshRemesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "scale": ("FLOAT",{"default":1.0,"min":0.01,"max":9.99,"step":0.01}),
                "resolution": ([128,256,512,1024,2048],{"default":1024}),
                "band": ("FLOAT", {"default":1.0,"min":0.1,"max":9.9,"step":0.1}),
                "project_back": ("FLOAT", {"default":0.9,"min":0.1,"max":9.9,"step":0.1}),
            },
        }

    RETURN_TYPES = ("TRIMESH", )
    RETURN_NAMES = ("trimesh", )
    FUNCTION = "process"
    CATEGORY = "CuMeshWrapper"

    def process(self, trimesh, scale, resolution, band, project_back):
        vertices = torch.from_numpy(trimesh.vertices).float()
        faces = torch.from_numpy(trimesh.faces).int()
        print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

        vertices = vertices.cuda()
        faces = faces.cuda()
        
        aabb_max = vertices.max(dim=0)[0]
        aabb_min = vertices.min(dim=0)[0]
        center = (aabb_max + aabb_min) / 2
        scale = (aabb_max - aabb_min).max().item()
        print(f"Center: {center}, Scale: {scale}")

        print(f"Building BVH for current mesh...")
        bvh = cuMesh.cuBVH(vertices, faces)

        new_vertices, new_faces = cumesh.remeshing.remesh_narrow_band_dc(
            vertices, faces,
            center = center,
            scale = scale,
            resolution = resolution,
            band = band,
            project_back = project_back,
            verbose = True,
            bvh = bvh
        )
        
        trimesh.vertices = new_vertices.cpu().numpy()
        trimesh.faces = new_faces.cpu().numpy()
        
        return (trimesh,)
        
class CuMeshSimplify:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "target_faces": ("INT",{"default":200000,"min":100,"max":2000000}),
            },
        }

    RETURN_TYPES = ("TRIMESH", )
    RETURN_NAMES = ("trimesh", )
    FUNCTION = "process"
    CATEGORY = "CuMeshWrapper"

    def process(self, trimesh, target_faces):
        mesh = TrimeshToCuMesh(trimesh)
        mesh.simplify(target_faces, verbose=True)
        new_vertices, new_faces = mesh.read()
        
        trimesh.vertices = new_vertices.cpu().numpy()
        trimesh.faces = new_faces.cpu().numpy()        
        
        return (trimesh,)   

class CuMeshFillHoles:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "max_hole_perimeter": ("FLOAT",{"default":0.1,"min":0.001,"max":100.000}),
            },
        }

    RETURN_TYPES = ("TRIMESH", )
    RETURN_NAMES = ("trimesh", )
    FUNCTION = "process"
    CATEGORY = "CuMeshWrapper"

    def process(self, trimesh, max_hole_perimeter):
        mesh = TrimeshToCuMesh(trimesh)
        mesh.fill_holes(max_hole_perimeter=0.1)
        new_vertices, new_faces = mesh.read()
        
        trimesh.vertices = new_vertices.cpu().numpy()
        trimesh.faces = new_faces.cpu().numpy()        
        
        return (trimesh,)   

class CuMeshLoadMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "glb_path": ("STRING", {"default": "", "tooltip": "The glb path with mesh to load."}), 
            }
        }
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    OUTPUT_TOOLTIPS = ("The glb model with mesh to texturize.",)
    
    FUNCTION = "load"
    CATEGORY = "CuMeshWrapper"
    DESCRIPTION = "Loads a glb model from the given path."

    def load(self, glb_path):

        if not os.path.exists(glb_path):
            glb_path = os.path.join(folder_paths.get_input_directory(), glb_path)
        
        trimesh = Trimesh.load(glb_path, force="mesh")
        
        return (trimesh,)

class CuMeshExportMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "filename_prefix": ("STRING", {"default": "3D/Hy3D"}),
                "file_format": (["glb", "obj", "ply", "stl", "3mf", "dae"],),
            },
            "optional": {
                "save_file": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "process"
    CATEGORY = "CuMeshWrapper"
    OUTPUT_NODE = True

    def process(self, trimesh, filename_prefix, file_format, save_file=True):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        output_glb_path = Path(full_output_folder, f'{filename}_{counter:05}_.{file_format}')
        output_glb_path.parent.mkdir(exist_ok=True)
        if save_file:
            trimesh.export(output_glb_path, file_type=file_format)
            relative_path = Path(subfolder) / f'{filename}_{counter:05}_.{file_format}'
        else:
            temp_file = Path(full_output_folder, f'hy3dtemp_.{file_format}')
            trimesh.export(temp_file, file_type=file_format)
            relative_path = Path(subfolder) / f'hy3dtemp_.{file_format}'
        
        return (str(relative_path), )        
        
class CuMeshPostProcessMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "fill_holes": ("BOOLEAN",{"default":True}),
                "max_hole_perimeter": ("FLOAT",{"default":0.1,"min":0.001,"max":100.000}),
                "remove_duplicate_faces": ("BOOLEAN",{"default":True}),
                "repair_non_manifold_edges": ("BOOLEAN", {"default":True}),
                "remove_non_manifold_faces": ("BOOLEAN", {"default":True}),
                "remove_small_connected_components": ("BOOLEAN", {"default":True}),
                "remove_small_connected_components_size": ("FLOAT", {"default":0.00001,"min":0.00001,"max":9.99999,"step":0.00001}),
            },
        }

    RETURN_TYPES = ("TRIMESH", )
    RETURN_NAMES = ("trimesh", )
    FUNCTION = "process"
    CATEGORY = "CuMeshWrapper"

    def process(self, trimesh, fill_holes, max_hole_perimeter, remove_duplicate_faces, repair_non_manifold_edges, remove_non_manifold_faces, remove_small_connected_components, remove_small_connected_components_size):
        print(f"Original mesh: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")
        
        mesh = TrimeshToCuMesh(trimesh)
        
        if fill_holes:
            print('Filling holes ...')
            mesh.fill_holes(max_hole_perimeter=0.1)
            
        if remove_duplicate_faces:
            print('Removing duplicate faces ...')
            mesh.remove_duplicate_faces()
            
        if repair_non_manifold_edges:
            print('Repairing non manifold edges ...')
            mesh.repair_non_manifold_edges()
            
        if remove_non_manifold_faces:
            print('Removing non manifold faces ...')
            mesh.remove_non_manifold_faces()
            
        if remove_small_connected_components:
            print('Removing small connected components ...')
            mesh.remove_small_connected_components(remove_small_connected_components_size)
        
        mesh.unify_face_orientations()            
        
        new_vertices, new_faces = mesh.read()
        
        print(f"After post processing: {len(new_vertices)} vertices, {len(new_faces)} faces")
        
        trimesh.vertices = new_vertices.cpu().numpy()
        trimesh.faces = new_faces.cpu().numpy()        
        
        return (trimesh,)           

NODE_CLASS_MAPPINGS = {
    "CuMeshUVUnWrap": CuMeshUVUnWrap,
    "CuMeshRemesh": CuMeshRemesh,
    "CuMeshSimplify": CuMeshSimplify,
    "CuMeshFillHoles": CuMeshFillHoles,
    "CuMeshLoadMesh": CuMeshLoadMesh,
    "CuMeshExportMesh": CuMeshExportMesh,
    "CuMeshPostProcessMesh": CuMeshPostProcessMesh,
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "CuMeshUVUnWrap": "CuMesh - UV UnWrap",
    "CuMeshRemesh": "CuMesh - Remesh",
    "CuMeshSimplify": "CuMesh - Simplify",
    "CuMeshFillHoles": "CuMesh - Fill Holes",
    "CuMeshLoadMesh": "CuMesh - Load Mesh",
    "CuMeshExportMesh": "CuMesh - Export Mesh",
    "CuMeshPostProcessMesh": "CuMesh - PostProcess Mesh",
    }
