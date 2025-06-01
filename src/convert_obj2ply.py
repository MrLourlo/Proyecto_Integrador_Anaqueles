def convert_obj_with_vertex_colors_to_ply(obj_path, ply_path):
    vertices = []
    faces = []

    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                if len(parts) >= 7:
                    x, y, z = map(float, parts[1:4])
                    r, g, b = map(float, parts[4:7])
                    # Convert r/g/b from 0–1 to 0–255
                    r, g, b = int(r * 255), int(g * 255), int(b * 255)
                    vertices.append((x, y, z, r, g, b))
            elif line.startswith('f '):
                parts = line.strip().split()
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(face)

    with open(ply_path, 'w') as f:
        # Write PLY header
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(vertices)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write(f'element face {len(faces)}\n')
        f.write('property list uchar int vertex_indices\n')
        f.write('end_header\n')

        # Write vertices
        for v in vertices:
            f.write(f'{v[0]} {v[1]} {v[2]} {v[3]} {v[4]} {v[5]}\n')

        # Write faces
        for face in faces:
            f.write(f'{len(face)} {" ".join(map(str, face))}\n')

    print(f"Successfully converted to: {ply_path}")


# Main
model_obj = '../obj/hotnuts.obj'
model_ply = '../ply/barcel/hotnuts.ply'
convert_obj_with_vertex_colors_to_ply(model_obj, model_ply)
