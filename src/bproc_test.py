import blenderproc as bproc
import bpy
import os
import numpy as np
from mathutils import Matrix, Euler, Vector
import cv2
import json
import random

# ---------- CONFIGURACIONES ----------
OBJ_FOLDER = os.path.join(os.path.dirname(__file__), "../obj")
OUTPUT_DIR = "output_dataset"
COCO_DIR = os.path.join(OUTPUT_DIR, "coco_data")
IMG_OUTPUT_DIR = os.path.join(COCO_DIR, "imgs")
MASK_OUTPUT_DIR = os.path.join(COCO_DIR, "masks")
NUM_SCENES = 1
NUM_CAMERAS = 5

os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)

# ---------- FUNCIONES AUXILIARES ----------

def setup_scene():

    # 1. HDRI Background Setup
    # ---------------------------
    hdri_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../hdri"))
    hdri_files = [f for f in os.listdir(hdri_dir) if f.lower().endswith(".exr")]

    if not hdri_files:
        raise FileNotFoundError("No HDR files found in hdri directory")

    chosen_hdri = random.choice(hdri_files)
    hdri_path = os.path.join(hdri_dir, chosen_hdri)
    print(f"Using HDRI: {chosen_hdri}")

    # Ensure the scene has a world and that it uses nodes
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")
    world = bpy.context.scene.world
    if world.node_tree is None:
        world.use_nodes = True

    # Apply HDRI background
    bproc.world.set_world_background_hdr_img(hdri_path)

    # 2. Floor setup
    #-------------------------
    plane = bproc.object.create_primitive("PLANE", scale=[5, 5, 1])
    plane.set_location([0, 0, -4])
    plane.set_cp("category_id", 0)
    plane.enable_rigidbody(active=False, collision_shape="MESH")
    #Textura
    textures_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../textures"))
    #texture_folders = [f for f in os.listdir(textures_root) if os.path.isdir(os.path.join(textures_root, f))]
    chosen_folder = 'carpet' #random.choice(texture_folders)
    chosen_path = os.path.join(textures_root, chosen_folder)
    #lista de texturas
    textures = [t for t in os.listdir(chosen_path) if os.path.isfile(os.path.join(chosen_path, t))]
    texture_file = random.choice(textures)

    texture_path = os.path.join(chosen_path, texture_file)
    print(f"Using texture: {texture_path}")

    # Create material
    mat = bproc.material.create("textured_plane_mat")
    mat_blender = mat.blender_obj
    nodes = mat_blender.node_tree.nodes
    links = mat_blender.node_tree.links

    tex_image_node = nodes.new(type='ShaderNodeTexImage')
    tex_image_node.image = bpy.data.images.load(texture_path)

    bsdf_node = next((node for node in nodes if node.type == 'BSDF_PRINCIPLED'), None)
    if bsdf_node is None:
        raise RuntimeError("No Principled BSDF node found in material")

    links.new(tex_image_node.outputs['Color'], bsdf_node.inputs['Base Color'])

    # Ensure material slot exists and assign material
    if len(plane.blender_obj.data.materials) == 0:
        plane.blender_obj.data.materials.append(None)
    plane.set_material(0, mat)
    
    #3. Light setup
    #--------------------------------
    sun = bproc.types.Light()
    sun.set_type("SUN")
    sun.set_location([
        6 + np.random.uniform(-1, 1),
        -6 + np.random.uniform(-1, 1),
        8 + np.random.uniform(-1, 1)
    ])
    sun.set_energy(3 + np.random.uniform(-1, 1))


# deprecated, now using img textures
def create_vertex_color_material():
    mat = bproc.material.create("vertex_color_mat")
    nt = mat.blender_obj.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    vc_node = nt.nodes.new(type="ShaderNodeVertexColor")
    vc_node.layer_name = "Col"
    nt.links.new(vc_node.outputs["Color"], bsdf.inputs["Base Color"])
    return mat

def deform_faces_randomly(obj, strength=0.1, probability=0.1):
    mesh = obj.blender_obj.data
    if not mesh:
        return

    bpy.context.view_layer.objects.active = obj.blender_obj
    bpy.ops.object.mode_set(mode='OBJECT')

    # Para cada cara
    for face in mesh.polygons:
        if np.random.rand() < probability:
            # Obtener el centroide de la cara
            center = sum((mesh.vertices[i].co for i in face.vertices), Vector()) / len(face.vertices)

            # Dirección de deformación: hacia adentro (normal negativa) + ruido
            deformation_dir = -face.normal + Vector(np.random.uniform(-0.1, 0.1, 3))
            deformation_vec = Vector(deformation_dir).normalized() * strength

            # Mover todos los vértices de la cara
            for i in face.vertices:
                mesh.vertices[i].co += deformation_vec



def load_and_place_objects(obj_folder):
    objects = []
    category_map = {}
    class_id = 1

    model_files = sorted([f for f in os.listdir(obj_folder) if f.lower().endswith(".obj")])
    for fname in model_files:
        obj_path = os.path.join(obj_folder, fname)
        obj_name = os.path.splitext(fname)[0]

        # Cargar el .obj con su .mtl y textura
        base_objs = bproc.loader.load_obj(obj_path)

        for base_obj in base_objs:
            for _ in range(np.random.randint(2, 5)):
                obj = base_obj.duplicate()
                obj.set_location(np.random.uniform(-3, 3, size=3))
                obj.set_rotation_euler(Euler(np.random.uniform(-np.pi / 4, np.pi / 4, size=3)))
                obj.set_cp("category_id", class_id)
                #deform_faces_randomly(obj, 0.01, 0.01)
                obj.enable_rigidbody(active=True)
                objects.append(obj)

        category_map[class_id] = obj_name
        class_id += 1

        # Elimina el original para evitar duplicados
        for base_obj in base_objs:
            base_obj.delete()

    return objects, category_map


def generate_camera_poses(num_angles=10):
    poses = []
    for i in range(num_angles):
        angle = 2 * np.pi * i / num_angles + np.random.uniform(-0.2, 0.2)
        distance = 4 * np.random.uniform(0.9, 1.1)
        height = 1 * np.random.uniform(0.9, 1.1)
        x, y = distance * np.cos(angle), distance * np.sin(angle)
        z = height
        cam_location = [x, y, z]
        target = [np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3), -4]
        cam_rot_matrix = bproc.camera.rotation_from_forward_vec(np.array(target) - np.array(cam_location))
        pose = Matrix.Translation(cam_location) @ Matrix(cam_rot_matrix).to_4x4()
        poses.append(pose)
    return poses

def write_coco_annotations(all_data, category_id_to_name):
    annotations, images, categories = [], [], []
    image_id, annotation_id = 1, 1
    instance_counts_global = {}
    category_ids = set()

    for frame_idx, (rgb_img, seg_img, attr_map) in enumerate(zip(all_data["colors"], all_data["instance_segmaps"], all_data["instance_attribute_maps"])):
        img_name = f"img_{frame_idx:05d}.jpg"
        img_rgb = cv2.cvtColor(rgb_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(IMG_OUTPUT_DIR, img_name), img_rgb)

        instance_counts_per_class = {}
        for instance_id in np.unique(seg_img):
            if instance_id == 0:
                continue
            attr = next((a for a in attr_map if a["idx"] == instance_id), None)
            if attr is None:
                continue
            category_id = attr["category_id"]
            category_ids.add(category_id)
            instance_counts_per_class.setdefault(category_id, 0)
            instance_counts_global.setdefault(category_id, 0)

            mask = (seg_img == instance_id).astype(np.uint8) * 255
            mask_name = f"img_{frame_idx:05d}_obj{category_id}_{instance_counts_per_class[category_id]:04d}.png"
            #cv2.imwrite(os.path.join(MASK_OUTPUT_DIR, mask_name), mask)

            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())

            # Encontrar contornos para segmentación COCO
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segmentation = []
            for contour in contours:
                seg = contour.flatten().tolist()
                if len(seg) >= 6:
                    segmentation.append(seg)

            annotations.append({
                "instance_id": annotation_id,
                "image_id": image_id,
                "class_id": category_id,
                "bbox": [x, y, w, h],
                "area": int(mask.sum() / 255),
                "segmentation": segmentation,
                "iscrowd": 0
            })
            annotation_id += 1
            instance_counts_per_class[category_id] += 1
            instance_counts_global[category_id] += 1

        images.append({
            "id": image_id,
            "file_name": img_name,
            "height": seg_img.shape[0],
            "width": seg_img.shape[1]
        })
        image_id += 1

    for cid in sorted(category_ids):
        name = category_id_to_name.get(cid, f"class_{cid}")
        categories.append({"id": cid, "name": name})

    coco_output = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(os.path.join(COCO_DIR, "coco_annotations.json"), "w") as f:
        json.dump(coco_output, f, indent=4)
    print(f"Dataset completo guardado en: {COCO_DIR}")

# ---------- Main loop ----------

def main():
    bproc.init()

    all_data = {
        "colors": [],
        "instance_segmaps": [],
        "instance_attribute_maps": []
    }

    for scene_id in range(NUM_SCENES):
        print(f"Generando escena {scene_id + 1}/{NUM_SCENES}")
        bproc.clean_up()

        # Escena base
        setup_scene()

        #mat = create_vertex_color_material()
        objects, category_id_to_name = load_and_place_objects(OBJ_FOLDER)

        cam_poses = generate_camera_poses(NUM_CAMERAS)
        for i, pose in enumerate(cam_poses):
            bproc.camera.add_camera_pose(pose, frame=15 + i)

        bproc.renderer.set_output_format("PNG")
        bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])
        bproc.renderer.set_max_amount_of_samples(80)
        bproc.renderer.set_light_bounces(diffuse_bounces=2, glossy_bounces=2)

        #nos interesa cuando ya estan en el suelo
        #simulate_physics_fix_final_poses
        bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=1, max_simulation_time=10, check_object_interval=1, substeps_per_frame=24)
        bproc.utility.set_keyframe_render_interval(frame_start=15, frame_end=15 + NUM_CAMERAS)
        data = bproc.renderer.render()

        # Acumular datos
        for k in all_data:
            all_data[k].extend(data[k])

    write_coco_annotations(all_data, category_id_to_name)

if __name__ == "__main__":
    main()
