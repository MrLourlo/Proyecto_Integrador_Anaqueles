import blenderproc as bproc
import os
import numpy as np
from mathutils import Matrix, Euler
import cv2
import json

# Inicializa BlenderProc
bproc.init()

# Configuraciones
ply_folder = os.path.join(os.path.dirname(__file__), "../ply/bimbo")
output_dir = "output_ply_vertex"
coco_dir = os.path.join(output_dir, "coco_data")
img_output_dir = os.path.join(coco_dir, "imgs")
mask_output_dir = os.path.join(coco_dir, "masks")
os.makedirs(img_output_dir, exist_ok=True)
os.makedirs(mask_output_dir, exist_ok=True)

# Crear plano base
plane = bproc.object.create_primitive("PLANE", scale=[5, 5, 1])
plane.set_location([0, 0, -3])
plane.set_cp("category_id", 0)
#el suelo solo actua como un obstaculo, por lo que sus fisicas son pasivas
plane.enable_rigidbody(active=False, collision_shape="MESH")

# Material con vertex colors
def create_vertex_color_material():
    mat = bproc.material.create("vertex_color_mat")
    node_tree = mat.blender_obj.node_tree
    bsdf_node = next(n for n in node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    vc_node = node_tree.nodes.new(type="ShaderNodeVertexColor")
    vc_node.layer_name = "Col"
    node_tree.links.new(vc_node.outputs["Color"], bsdf_node.inputs["Base Color"])
    return mat

# Cargar objetos y generar instancias
objects = []
class_id = 1
category_id_to_name = {}
mat = create_vertex_color_material()
model_files = sorted([f for f in os.listdir(ply_folder) if f.lower().endswith(".ply")])
for fname in model_files:
    path = os.path.join(ply_folder, fname)
    base_objs = bproc.loader.load_obj(path) #carga los modelos originales en escena
    for base_obj in base_objs:
        for _ in range(np.random.randint(1, 4)): #clonamos instancias de un objeto original
            obj = base_obj.duplicate()
            obj.set_location(np.random.uniform(-3, 3, size=3))
            obj.set_rotation_euler(Euler(np.random.uniform(-np.pi/4, np.pi/4, size=3)))
            obj.set_material(0, mat)
            obj.set_cp("category_id", class_id)
            obj.enable_rigidbody(active=True) # habilitar fisicas
            objects.append(obj)
    category_id_to_name[class_id] = os.path.splitext(fname)[0]  # nombre del .ply sin extensión
    class_id += 1

    # Eliminar objetos base originales
    for base_obj in base_objs:
        base_obj.delete()

# Luz
light = bproc.types.Light()
light.set_type("SUN")
light.set_location([5, -5, 5])
light.set_energy(5)

# Cámara
cam_location = [0.0, -12.0, 6.0]
cam_target = [0, -2.0, 0]
cam_rot_matrix = bproc.camera.rotation_from_forward_vec(np.array(cam_target) - np.array(cam_location))
cam_pose = Matrix.Translation(cam_location) @ Matrix(cam_rot_matrix).to_4x4()
bproc.camera.add_camera_pose(cam_pose)

# Render settings
bproc.renderer.set_output_format("PNG")
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])
bproc.renderer.set_max_amount_of_samples(100)
bproc.renderer.set_light_bounces(diffuse_bounces=3, glossy_bounces=3)
# Run the simulation with physics
bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=1, max_simulation_time=20, check_object_interval=1)

# Renderizar
data = bproc.renderer.render()

# write the data to a .hdf5 container
bproc.writer.write_hdf5(output_dir, data)

# This will make the renderer render the first n frames of the simulation
#bproc.utility.set_keyframe_render_interval(frame_end=216)

# Guardar imagen RGB
img_bgr = data["colors"][0].astype(np.uint8)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_name = "img_0001.jpg"
cv2.imwrite(os.path.join(img_output_dir, img_name), img_rgb)
cv2.imwrite(os.path.join(output_dir, "rgb.png"), img_rgb)

# Segmentación
seg = data["instance_segmaps"][0]
seg_vis = (seg.astype(np.float32) / seg.max() * 255).astype(np.uint8)
cv2.imwrite(os.path.join(output_dir, "segmented_img.png"), seg)
cv2.imwrite(os.path.join(output_dir, "colored_seg.png"), seg_vis)

# Atributos de instancia
attr_maps = data["instance_attribute_maps"][0]

# Generar anotaciones
annotations = []
categories = []
image_id = 1
annotation_id = 1
category_ids = set()
instance_counts_per_class = {}

for instance_id in np.unique(seg):
    if instance_id == 0:
        continue
    attr = next((a for a in attr_maps if a["idx"] == instance_id), None)
    if attr is None:
        continue
    category_id = attr["category_id"]
    category_ids.add(category_id)

    # Crear máscara binaria
    mask = (seg == instance_id).astype(np.uint8) * 255
    # Inicializa contador por clase si no existe
    if category_id not in instance_counts_per_class:
        instance_counts_per_class[category_id] = 0

    # Número de instancia para esta clase
    instance_num = instance_counts_per_class[category_id]
    mask_name = f"img_0001_obj{category_id}_{instance_num:04d}.png"

    # Incrementa contador por clase
    instance_counts_per_class[category_id] += 1
    cv2.imwrite(os.path.join(mask_output_dir, mask_name), mask)

    # Calcular bbox
    ys, xs = np.where(mask > 0)
    x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())

    annotations.append({
        "intance_id": annotation_id,
        "image_id": image_id,
        "class_id": category_id,
        "bbox": [x, y, w, h],
        "area": int(mask.sum() / 255)
    })
    annotation_id += 1

# Crear clases
for cid in sorted(category_ids):
    name = category_id_to_name.get(cid, f"class_{cid}")
    categories.append({"id": cid, "name": name})

# Guardar archivo COCO
coco_output = {
    "images": [{
        "id": image_id,
        "file_name": img_name,
        "height": seg.shape[0],
        "width": seg.shape[1]
    }],
    "annotations": annotations,
    "classes": categories
}
with open(os.path.join(coco_dir, "coco_annotations.json"), "w") as f:
    json.dump(coco_output, f, indent=4)

print(f"Dataset generado exitosamente en {output_dir}")
