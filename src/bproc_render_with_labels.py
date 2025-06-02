import blenderproc as bproc
import os
import bpy
import numpy as np
from mathutils import Matrix, Euler
from sklearn.neighbors import KDTree
import random
import json
import cv2

def generate_spaced_positions(num_objects, min_distance=1.0, max_attempts=100):
    positions = []
    bounds = [-2, 2]
    for _ in range(num_objects):
        attempts = 0
        while attempts < max_attempts:
            new_pos = np.random.uniform(bounds[0], bounds[1], size=3)
            new_pos[2] = 0
            if len(positions) == 0:
                positions.append(new_pos)
                break
            tree = KDTree(np.array(positions))
            dist, _ = tree.query([new_pos], k=1)
            if dist[0][0] >= min_distance:
                positions.append(new_pos)
                break
            attempts += 1
        if attempts >= max_attempts:
            new_pos = np.random.uniform(bounds[0], bounds[1], size=3)
            new_pos[2] = 0
            positions.append(new_pos)
    return positions

def create_vertex_color_material():
    mat = bproc.material.create("vertex_color_mat")
    mat_blender = mat.blender_obj
    node_tree = mat_blender.node_tree
    bsdf_node = next((n for n in node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if not bsdf_node:
        raise RuntimeError("No Principled BSDF node found")
    vc_node = node_tree.nodes.new(type="ShaderNodeVertexColor")
    vc_node.layer_name = "Col"
    node_tree.links.new(vc_node.outputs["Color"], bsdf_node.inputs["Base Color"])
    return mat

def generate_camera_poses(num_angles=10):
    poses = []
    base_distance = 12
    base_height = 3
    for i in range(num_angles):
        angle = 2 * np.pi * i / num_angles + np.random.uniform(-0.2, 0.2)
        distance = base_distance * np.random.uniform(0.9, 1.1)
        height = base_height * np.random.uniform(0.9, 1.1)
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        z = height
        target = [np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3), 0]
        cam_location = [x, y, z]
        cam_rot_matrix = bproc.camera.rotation_from_forward_vec(np.array(target) - np.array(cam_location))
        pose = Matrix.Translation(cam_location) @ Matrix(cam_rot_matrix).to_4x4()
        poses.append(pose)
    return poses

def setup_scene(scene_idx):
    bproc.clean_up()

    hdri_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../hdri"))
    hdri_files = [f for f in os.listdir(hdri_dir) if f.lower().endswith(".hdr")]
    chosen_hdri = random.choice(hdri_files)
    hdri_path = os.path.join(hdri_dir, chosen_hdri)

    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")
    if bpy.context.scene.world.node_tree is None:
        bpy.context.scene.world.use_nodes = True

    bproc.world.set_world_background_hdr_img(hdri_path)

    plane = bproc.object.create_primitive("PLANE", scale=[5, 5, 1])
    plane.set_location([0, 0, -3])
    plane.enable_rigidbody(active=False)

    textures_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../textures"))
    texture_folders = [f for f in os.listdir(textures_root) if os.path.isdir(os.path.join(textures_root, f))]
    chosen_folder = random.choice(texture_folders)
    chosen_path = os.path.join(textures_root, chosen_folder)
    texture_file = next((f for f in os.listdir(chosen_path) if "_Color" in f and f.lower().endswith(".jpg")), None)
    texture_path = os.path.join(chosen_path, texture_file)

    mat = bproc.material.create("textured_plane_mat")
    mat_blender = mat.blender_obj
    nodes = mat_blender.node_tree.nodes
    links = mat_blender.node_tree.links
    tex_image_node = nodes.new(type='ShaderNodeTexImage')
    tex_image_node.image = bpy.data.images.load(texture_path)
    bsdf_node = next((node for node in nodes if node.type == 'BSDF_PRINCIPLED'), None)
    if bsdf_node is None:
        raise RuntimeError("No Principled BSDF node found in material")
    if len(plane.blender_obj.data.materials) == 0:
        plane.blender_obj.data.materials.append(None)
    links.new(tex_image_node.outputs['Color'], bsdf_node.inputs['Base Color'])
    plane.set_material(0, mat)

    ply_folder = os.path.join(os.path.dirname(__file__), "../ply")
    objects = []
    ply_files = [f for f in os.listdir(ply_folder) if f.lower().endswith(".ply")]
    positions = generate_spaced_positions(len(ply_files), min_distance=1.2)

    for i, fname in enumerate(ply_files):
        path = os.path.join(ply_folder, fname)
        class_name = os.path.splitext(fname)[0]
        objs = bproc.loader.load_obj(path)
        for obj in objs:
            obj.set_location(positions[i])
            obj.set_rotation_euler(Euler(np.random.uniform(0, 2 * np.pi, size=3)))
            material = create_vertex_color_material()
            obj.set_material(0, material)
            obj.enable_rigidbody(active=True)
            obj.set_cp("class_name", class_name)
            obj.blender_obj.pass_index = random.randint(1, 10000)
            objects.append(obj)

    bproc.object.simulate_physics_and_fix_final_poses(4, 20, 1)
    return objects

def render_scene(output_dir, scene_idx, angle_idx):
    bproc.renderer.set_output_format("PNG")
    bproc.renderer.enable_segmentation_output(map_by="instance")
    bproc.renderer.set_max_amount_of_samples(100)
    bproc.renderer.set_light_bounces(diffuse_bounces=3, glossy_bounces=3)

    data = bproc.renderer.render()
    frame_dir = os.path.join(output_dir, f"frame_{angle_idx:02d}")
    os.makedirs(frame_dir, exist_ok=True)

    bproc.writer.write_hdf5(frame_dir, data)

    img_bgr = (data["colors"][0]).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(frame_dir, "rgb.png"), img_rgb)

    seg = data["instance_segmaps"][0]
    seg_vis = (seg.astype(np.float32) / seg.max() * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(frame_dir, "segmentation.png"), seg_vis)

    # ✅ Mapeo por instance_id con validación
    mapping = {}
    for obj in bproc.object.get_all_mesh_objects():
        instance_id = obj.blender_obj.pass_index
        if obj.has_cp("class_name"):
            class_name = obj.get_cp("class_name")
            if class_name and instance_id != 0:
                mapping[str(instance_id)] = class_name

    json_path = os.path.join(frame_dir, "instance_class_map.json")
    with open(json_path, "w") as f:
        json.dump(mapping, f)

if __name__ == "__main__":
    bproc.init()
    for scene_idx in range(10):
        print(f"\n🔧 Generando escena {scene_idx + 1}/10")
        bproc.utility.reset_keyframes()
        setup_scene(scene_idx)
        camera_poses = generate_camera_poses(10)
        scene_dir = os.path.join("output_scenes", f"scene_{scene_idx:02d}")
        os.makedirs(scene_dir, exist_ok=True)
        for angle_idx, pose in enumerate(camera_poses):
            print(f"  🎥 Ángulo {angle_idx + 1}/10")
            bproc.utility.reset_keyframes()
            bproc.camera.add_camera_pose(pose)
            render_scene(scene_dir, scene_idx, angle_idx)

    print("\n✅ Todas las escenas fueron generadas correctamente.")
