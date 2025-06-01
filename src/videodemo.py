import blenderproc as bproc
import os
import numpy as np
from mathutils import Matrix, Euler
import cv2
import json

# ---------- CONFIGURACIONES GLOBALES ----------
PLY_FOLDER = os.path.join(os.path.dirname(__file__), "../ply/bimbo")
OUTPUT_DIR = "output_ply_vertex"

# ---------- FUNCIONES AUXILIARES ----------

def create_vertex_color_material():
    mat = bproc.material.create("vertex_color_mat")
    node_tree = mat.blender_obj.node_tree
    bsdf_node = next(n for n in node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    vc_node = node_tree.nodes.new(type="ShaderNodeVertexColor")
    vc_node.layer_name = "Col"
    node_tree.links.new(vc_node.outputs["Color"], bsdf_node.inputs["Base Color"])
    return mat

def load_and_place_objects(ply_folder, material):
    objects = []
    category_map = {}
    class_id = 1

    model_files = sorted([f for f in os.listdir(ply_folder) if f.lower().endswith(".ply")])
    for fname in model_files:
        path = os.path.join(ply_folder, fname)
        base_objs = bproc.loader.load_obj(path)

        for base_obj in base_objs:
            for _ in range(np.random.randint(2, 5)):
                obj = base_obj.duplicate()
                obj.set_location(np.random.uniform(-4, 4, size=3))
                obj.set_rotation_euler(Euler(np.random.uniform(-np.pi / 4, np.pi / 4, size=3)))
                obj.set_material(0, material)
                obj.set_cp("category_id", class_id)
                obj.enable_rigidbody(active=True)
                objects.append(obj)

        category_map[class_id] = os.path.splitext(fname)[0]
        class_id += 1

        for base_obj in base_objs:
            base_obj.delete()

    return objects, category_map

def generate_orbit_camera_poses(num_frames=60, radius=15, height=5):
    poses = []
    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height

        location = np.array([x, y, z])
        target = np.array([0.0, 0.0, 0.0])
        rot_matrix = bproc.camera.rotation_from_forward_vec(target - location)
        pose = Matrix.Translation(location) @ Matrix(rot_matrix).to_4x4()
        poses.append(pose)
    return poses


def gen_frames_and_video(data, output_dir, fps=10, video_name="demo_video.mp4"):
    img_dir = os.path.join(output_dir, "frames")
    os.makedirs(img_dir, exist_ok=True)

    frame_paths = []

    # Guardar imágenes
    for i, rgb in enumerate(data["colors"]):
        img_path = os.path.join(img_dir, f"frame_{i:04d}.png")
        cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        frame_paths.append(img_path)

    # Crear video
    if frame_paths:
        h, w, _ = cv2.imread(frame_paths[0]).shape
        video_path = os.path.join(output_dir, video_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

        for path in frame_paths:
            frame = cv2.imread(path)
            out.write(frame)

        out.release()
        print(f"Video guardado en: {video_path}")



# ---------- PIPELINE PRINCIPAL ----------

def main():
    bproc.init()

    # Crear escena básica
    plane = bproc.object.create_primitive("PLANE", scale=[5, 5, 1])
    plane.set_location([0, 0, -3])
    plane.set_cp("category_id", 0)
    plane.enable_rigidbody(active=False, collision_shape="MESH")

    # Luz
    light = bproc.types.Light()
    light.set_type("SUN")
    light.set_location([5, -5, 5])
    light.set_energy(5)

    # Material y objetos
    mat = create_vertex_color_material()
    _, category_id_to_name = load_and_place_objects(PLY_FOLDER, mat)

    # Cámaras en órbita suave
    camera_poses = generate_orbit_camera_poses(num_frames=90)
    for pose in camera_poses:
        bproc.camera.add_camera_pose(pose)


    # Render settings
    bproc.renderer.set_output_format("PNG")
    bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])
    bproc.renderer.set_max_amount_of_samples(100)
    bproc.renderer.set_light_bounces(diffuse_bounces=3, glossy_bounces=3)

    # Simulación de física
    bproc.object.simulate_physics(min_simulation_time=1, max_simulation_time=15, check_object_interval=1, substeps_per_frame=24)
    # Renderizar múltiples frames
    #bproc.utility.set_keyframe_render_interval(frame_start=0, frame_end=25) #frames a simular con fisicas
    # Renderizar escena
    data = bproc.renderer.render()
    gen_frames_and_video(data, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
