import blenderproc as bproc
import os
import numpy as np
from mathutils import Matrix, Euler
import cv2
import bpy
import random

# ---------- CONFIGURACIONES ----------
OBJ_FOLDER = os.path.join(os.path.dirname(__file__), "../obj")
HDRI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../hdri"))
TEXTURE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../textures"))
OUTPUT_DIR = "video_output"
NUM_PARTS = 3  # Número de escenas distintas

# ---------- FUNCIONES AUXILIARES ----------

def setup_scene(texture_name, hdri_name):
    # HDRI
    hdri_path = os.path.join(HDRI_DIR, hdri_name)
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")
    world = bpy.context.scene.world
    if world.node_tree is None:
        world.use_nodes = True
    bproc.world.set_world_background_hdr_img(hdri_path)

    # Piso con textura
    plane = bproc.object.create_primitive("PLANE", scale=[6, 6, 1])
    plane.set_location([0, 0, -4])
    plane.set_cp("category_id", 0)
    plane.enable_rigidbody(active=False, collision_shape="MESH")

    texture_path = os.path.join(TEXTURE_ROOT, texture_name)
    mat = bproc.material.create("textured_plane_mat")
    nodes = mat.blender_obj.node_tree.nodes
    links = mat.blender_obj.node_tree.links
    tex_node = nodes.new(type='ShaderNodeTexImage')
    tex_node.image = bpy.data.images.load(texture_path)
    bsdf_node = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED')
    links.new(tex_node.outputs['Color'], bsdf_node.inputs['Base Color'])

    if len(plane.blender_obj.data.materials) == 0:
        plane.blender_obj.data.materials.append(None)
    plane.set_material(0, mat)

    # Luz
    sun = bproc.types.Light()
    sun.set_type("SUN")
    sun.set_location([
        6 + np.random.uniform(-1, 1),
        -6 + np.random.uniform(-1, 1),
        8 + np.random.uniform(-1, 1)
    ])
    sun.set_energy(3 + np.random.uniform(-1, 1))


def load_and_place_objects(obj_folder):
    objects = []
    class_id = 1
    model_files = sorted([f for f in os.listdir(obj_folder) if f.lower().endswith(".obj")])

    for fname in model_files:
        obj_path = os.path.join(obj_folder, fname)
        base_objs = bproc.loader.load_obj(obj_path)
        for base_obj in base_objs:
            for _ in range(np.random.randint(5, 7)):
                obj = base_obj.duplicate()
                obj.set_location(np.random.uniform(-5, 5, size=3))
                obj.set_rotation_euler(Euler(np.random.uniform(-np.pi / 4, np.pi / 4, size=3)))
                obj.set_cp("category_id", class_id)
                obj.enable_rigidbody(active=True)
                objects.append(obj)
        for base_obj in base_objs:
            base_obj.delete()
        class_id += 1
    return objects


def generate_orbit_camera_poses(num_frames=60, radius=15, height=5):
    poses = []
    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height
        location = np.array([x, y, z])
        target = np.array([0.0, 0.0, -4.0])
        rot_matrix = bproc.camera.rotation_from_forward_vec(target - location)
        pose = Matrix.Translation(location) @ Matrix(rot_matrix).to_4x4()
        poses.append(pose)
    return poses


def gen_frames_and_video(data, output_dir, part_idx, fps=10):
    img_dir = os.path.join(output_dir, f"frames_part_{part_idx}")
    os.makedirs(img_dir, exist_ok=True)
    frame_paths = []

    for i, rgb in enumerate(data["colors"]):
        img_path = os.path.join(img_dir, f"frame_{i:04d}.png")
        cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        frame_paths.append(img_path)

    # Guardar video
    if frame_paths:
        h, w, _ = cv2.imread(frame_paths[0]).shape
        video_path = os.path.join(output_dir, f"demo_part_{part_idx}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        for path in frame_paths:
            out.write(cv2.imread(path))
        out.release()
        print(f"🎞️ Parte {part_idx} guardada: {video_path}")


def concat_videos(video_files, output_path):
    if not video_files:
        print("⚠️ No hay videos para concatenar.")
        return

    cap = cv2.VideoCapture(video_files[0])
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for vf in video_files:
        cap = cv2.VideoCapture(vf)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()

    out.release()
    print(f"✅ Video final unido: {output_path}")


# ---------- PIPELINE PRINCIPAL ----------

def main():
    hdri_files = [f for f in os.listdir(HDRI_DIR) if f.endswith(".exr")]
    texture_files = [os.path.join(d, f) for d in os.listdir(TEXTURE_ROOT)
                     for f in os.listdir(os.path.join(TEXTURE_ROOT, d))
                     if f.endswith((".jpg", ".png"))]
    
    bproc.init()

    for part_idx in range(NUM_PARTS):
        print(f"\n--- Escena {part_idx + 1} ---")
        bproc.clean_up()

        # Elegir HDRI y textura aleatorios
        chosen_hdri = random.choice(hdri_files)
        chosen_texture = random.choice(texture_files)

        setup_scene(texture_name=chosen_texture, hdri_name=chosen_hdri)
        load_and_place_objects(OBJ_FOLDER)

        poses = generate_orbit_camera_poses(num_frames=90)
        for pose in poses:
            bproc.camera.add_camera_pose(pose)

        bproc.renderer.set_output_format("PNG")
        bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])
        bproc.renderer.set_max_amount_of_samples(100)
        bproc.renderer.set_light_bounces(diffuse_bounces=3, glossy_bounces=3)

        bproc.object.simulate_physics(min_simulation_time=1, max_simulation_time=15,
                                      check_object_interval=1, substeps_per_frame=24)

        data = bproc.renderer.render()
        gen_frames_and_video(data, OUTPUT_DIR, part_idx)

    # Unir videos
    video_parts = [os.path.join(OUTPUT_DIR, f"demo_part_{i}.mp4") for i in range(NUM_PARTS)]
    final_output_path = os.path.join(OUTPUT_DIR, "video_final.mp4")
    concat_videos(video_parts, final_output_path)


if __name__ == "__main__":
    main()
