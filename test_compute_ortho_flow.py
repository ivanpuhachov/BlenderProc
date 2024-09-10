import blenderproc as bproc
import argparse
from blenderproc.python.types.MeshObjectUtility import MeshObject, create_primitive
import bpy

"""
Follows blenderproc examples
examples/advanced/optical_flow/main.py

Run with: 
blenderproc run test_compute_ortho_flow.py 

Visualize with:
blenderproc vis hdf5 output/2.hdf5

"""

parser = argparse.ArgumentParser()
parser.add_argument('camera', nargs='?', default="camera_my", help="Path to the camera file")
# parser.add_argument('scene', nargs='?', default="examples/resources/scene.obj", help="Path to the scene.obj file")
parser.add_argument('output_dir', nargs='?', default="output", help="Path to where the final files, will be saved")
args = parser.parse_args()

bproc.init()

# load the objects into the scene
# objs = bproc.loader.load_obj(args.scene)
cube = create_primitive("CUBE")

cheetah_image = bpy.data.images.load(filepath=str("cheetah_image.jpeg"))
materials = bproc.material.collect_all()
print(materials)
# mat.set_principled_shader_value("Base Color", cheetah_image)

mat = bproc.material.create(name="cheetah_material")
mat.set_principled_shader_value("Base Color", cheetah_image)

cube.add_material(mat)

# define a light and set its location and energy level
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([0, -5, 5])
light.set_energy(5000)

# define the camera intrinsics
bproc.camera.set_resolution(512, 512)
bproc.camera.make_ortho_camera()

# read the camera positions file and convert into homogeneous camera-world transformation
with open(args.camera, "r") as f:
    for line in f.readlines():
        line = [float(x) for x in line.split()]
        position, euler_rotation = line[:3], line[3:6]
        matrix_world = bproc.math.build_transformation_mat(position, euler_rotation)
        bproc.camera.add_camera_pose(matrix_world)

# render the whole pipeline
data = bproc.renderer.render()

# Render the optical flow (forward and backward) for all frames
data.update(bproc.renderer.render_optical_flow(get_backward_flow=True, get_forward_flow=True, blender_image_coordinate_style=False))

# write the data to a .hdf5 container
bproc.writer.write_hdf5(args.output_dir, data)
