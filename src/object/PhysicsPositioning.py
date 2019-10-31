import mathutils
import bpy

from src.main.Module import Module
from time import time

class PhysicsPositioning(Module):
    """ Performs physics simulation in the scene, assigns new poses for all objects that participated.

    .. csv-table::
       :header: "Parameter", "Description"

       "simulation_iterations", "For how many iterations the simulation should be computed until the new object positions should be read."
       "object_stopped_location_threshold", "The maximum difference per coordinate in the location vector that is allowed, such that an object is still recognized as 'stopped moving'."
       "object_stopped_rotation_threshold", "The maximum difference per coordinate in the rotation euler vector that is allowed. such that an object is still recognized as 'stopped moving'."
       "min_simulation_iterations", "The minimum number of iterations to simulate."
       "simulation_iterations_increase_step", "The value with which the simulation iterations should be increased until all objects have stopped moving."
       "max_simulation_iterations", "The maximum number of iterations to simulate."
    """

    def __init__(self, config):
        Module.__init__(self, config)
        self.object_stopped_location_threshold = self.config.get_float("object_stopped_location_threshold", 0.01)
        self.object_stopped_rotation_threshold = self.config.get_float("object_stopped_rotation_threshold", 0.01)

    def run(self):
        """ Performs physics simulation in the scene. """

        # Enable physics for all objects
        self._add_rigidbody()

        # Run simulation and use the position of the objects at the end of the simulation as new initial position.
        obj_poses = self._do_simulation()
        self._set_pose(obj_poses)

        # Disable physics for all objects
        self._remove_rigidbody()

    def _add_rigidbody(self):
        """ Adds a rigidbody element to all mesh objects and sets their type depending on the custom property "physics". """
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                bpy.context.view_layer.objects.active = obj
                bpy.ops.rigidbody.object_add()
                obj.rigid_body.type = obj["physics"].upper()
                # TODO: Configure this per object. MESH is very slow but sometimes necessary.
                obj.rigid_body.collision_shape = "MESH"

    def _remove_rigidbody(self):
        """ Removes the rigidbody element from all mesh objects. """
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                bpy.context.view_layer.objects.active = obj
                bpy.ops.rigidbody.object_remove()

    def _do_simulation(self):
        """ Perform the simulation.

        This method bakes the simulation for the configured number of iterations and returns all object positions at the last frame.

        :return: Dict of form {obj_name:{'location':[x, y, z], 'rotation':[x_rot, y_rot, z_rot]}}.
        """
        # Run simulation
        point_cache = bpy.context.scene.rigidbody_world.point_cache
        point_cache.frame_start = 1

        min_simulation_iterations = self.config.get_int("min_simulation_iterations", 100)
        max_simulation_iterations = self.config.get_int("max_simulation_iterations", 1000)
        simulation_iterations_increase_step = self.config.get_int("simulation_iterations_increase_step", 100)

        if min_simulation_iterations >= max_simulation_iterations:
            raise Exception("max_simulation_iterations has to be bigger than min_simulation_iterations")

        # Run simulation starting from min to max in the configured steps
        for simulation_iterations in range(min_simulation_iterations, max_simulation_iterations, simulation_iterations_increase_step):
            print("Running simulation up to frame " + str(simulation_iterations))

            # Simulate current interval
            point_cache.frame_end = simulation_iterations
            bpy.ops.ptcache.bake({"point_cache": point_cache}, bake=True)

            # Go to second last frame and get poses
            bpy.context.scene.frame_set(simulation_iterations - 1)
            second_last_frame_poses = self._get_pose()

            # Go to last frame of simulation and get poses
            bpy.context.scene.frame_set(simulation_iterations)
            last_frame_poses = self._get_pose()

            # Free bake (this will not completely remove the simulation cache, so further simulations can reuse the already calculated frames)
            bpy.ops.ptcache.free_bake({"point_cache": point_cache})

            # If objects have stopped moving between the last two frames, then stop here
            if self._have_objects_stopped_moving(second_last_frame_poses, last_frame_poses):
                print("Objects have stopped moving after " + str(simulation_iterations) + " iterations")
                break
            elif simulation_iterations + simulation_iterations_increase_step >= max_simulation_iterations:
                print("Stopping simulation as configured max_simulation_iterations has been reached")

        return last_frame_poses

    def _get_pose(self):
        """Returns position and rotation values of all objects in the scene with ACTIVE rigid_body type.

        :return: Dict of form {obj_name:{'location':[x, y, z], 'rotation':[x_rot, y_rot, z_rot]}}.
        """
        objects_poses = {}
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH' and obj.rigid_body.type == 'ACTIVE':
                location = bpy.context.scene.objects[obj.name].matrix_world.translation
                rotation = mathutils.Vector(bpy.context.scene.objects[obj.name].matrix_world.to_euler())
                objects_poses.update({obj.name: {'location': location, 'rotation': rotation}})

        return objects_poses

    def _set_pose(self, pose_dict):
        """ Sets location and rotation properties of objects.

        :param pose_dict: Dict of form {obj_name:{'location':[x, y, z], 'rotation':[x_rot, y_rot, z_rot]}}.
        """
        for obj_name in pose_dict:
            bpy.context.scene.objects[obj_name].location = pose_dict[obj_name]['location']
            bpy.context.scene.objects[obj_name].rotation_euler = pose_dict[obj_name]['rotation']


    def _have_objects_stopped_moving(self, last_poses, new_poses):
        """ Check if the difference between the two given poses per object is smaller than the configured threshold.

        :param last_poses: Dict of form {obj_name:{'location':[x, y, z], 'rotation':[x_rot, y_rot, z_rot]}}.
        :param new_poses: Dict of form {obj_name:{'location':[x, y, z], 'rotation':[x_rot, y_rot, z_rot]}}.
        :return: True, if no objects are moving anymore.
        """
        stopped = True
        for obj_name in last_poses:
            # Check location difference
            location_diff = last_poses[obj_name]['location'] - new_poses[obj_name]['location']
            for i in range(3):
                if location_diff[i] > self.object_stopped_location_threshold:
                    stopped = False

            # Check rotation difference
            rotation_diff = last_poses[obj_name]['rotation'] - new_poses[obj_name]['rotation']
            for i in range(3):
                if rotation_diff[i] > self.object_stopped_rotation_threshold:
                    stopped = False

            if not stopped:
                break

        return stopped