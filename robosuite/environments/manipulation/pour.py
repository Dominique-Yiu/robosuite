from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, MugObject, BallObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat
import robosuite.utils.transform_utils as T


class Pour(SingleArmEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    #  TODO:
    def reward(self, action=None):
        return 0.0
        # reward = 0.0

        # if self._check_success():
        #     reward = 2.0
        # elif self.reward_shaping:
        #     pass

        # # Scale reward if requested
        # if self.reward_scale is not None:
        #     reward *= self.reward_scale / 2.0

        # return reward

    # TODO:
    def _load_model(self):
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        balls = [
            BallObject(
                name=f"sphere{idx}",
                size=[0.01],
                rgba=[1, 0, 0, 1],
                material=redwood,
            )
            for idx in range(np.power(2, 3))
        ]
        self.liquid = balls

        # 2 cups
        mugs = [
            MugObject(
                name=f"mug_{idx}",
            )
            for idx in range(2)
        ]
        self.glass = mugs

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.glass + self.liquid,
        )

        self._get_placement_initializer()

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds
        """
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.glass,
                x_range=[-0.2, 0.2],
                y_range=[0.05, 0.2],
                rotation=None,
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.00,
            )
        )

    # TODO:
    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        self.glass_body_id = {}
        # self.liquid_body_id = {}

        for obj in self.glass:
            self.glass_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)

        # for obj in self.liquid:
        #     self.liquid_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)

    # TODO:
    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # reset obj sensor mappings
            self.object_id_to_sensors = {}

            # for conversion to relative gripper frame
            @sensor(modality=modality)
            def world_pose_in_gripper(obs_cache):
                return (
                    T.pose_inv(T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"])))
                    if f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache
                    else np.eye(4)
                )

            sensors = [world_pose_in_gripper]
            names = ["world_pose_in_gripper"]
            enableds = [True]
            actives = [False]

            for i, obj in enumerate(self.glass):
                obj_sensors, obj_sensor_names = self._create_obj_sensors(obj.name, modality=modality)
                sensors += obj_sensors
                names += obj_sensor_names
                enableds += [True] * len(obj_sensors)
                actives += [True] * len(obj_sensors)

            # Create observables
            for name, s, enabled, active in zip(names, sensors, enableds, actives):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    enabled=enabled,
                    active=active,
                )
        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            obj_name (str): Name of object to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.glass_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.glass_body_id[obj_name]], to="xyzw")

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            # Immediately return default value if cache is empty
            if any(
                [name not in obs_cache for name in [f"{obj_name}_pos", f"{obj_name}_quat", "world_pose_in_gripper"]]
            ):
                return np.zeros(3)
            obj_pose = T.pose2mat((obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"]))
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache["world_pose_in_gripper"])
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return (
                obs_cache[f"{obj_name}_to_{pf}eef_quat"] if f"{obj_name}_to_{pf}eef_quat" in obs_cache else np.zeros(4)
            )

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [f"{obj_name}_pos", f"{obj_name}_quat", f"{obj_name}_to_{pf}eef_pos", f"{obj_name}_to_{pf}eef_quat"]

        return sensors, names

    # TODO:
    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for idx, (obj_pos, obj_quat, obj) in enumerate(object_placements.values()):
                if idx == 0:
                    centers = calculate_sphere_centers(2, 0.015, center=np.array(obj_pos) + np.array([0, 0, 0.02]))
                    for i, center in enumerate(centers):
                        self.sim.data.set_joint_qpos(
                            self.liquid[i].joints[0], np.concatenate([np.array(center), np.array(obj_quat)])
                        )
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    # TODO: target is not self.cube
    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        # if vis_settings["grippers"]:
        #     self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube)

    # TODO
    def _check_success(self):
        pass
        # cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        # target_pos = self.model.mujoco_arena.target_cylinder.get("pos")
        # target_pos = [float(num) for num in target_pos.split()]
        # target_pos = np.array(target_pos) + self.model.mujoco_arena.table_offset
        # dist = np.linalg.norm(cube_pos - target_pos)

        # return dist < 0.01


def calculate_sphere_centers(cube_dimension, sphere_diameter, center=(0, 0, 0)):
    # Calculate the half of the sphere diameter
    radius = sphere_diameter / 2

    # Calculate the total length of the cube's side
    cube_side = cube_dimension * sphere_diameter

    # Find the start point (bottom-left-front corner of the cube)
    start_x = center[0] - (cube_side / 2) + radius
    start_y = center[1] - (cube_side / 2) + radius
    start_z = center[2] - (cube_side / 2) + radius

    # Calculate the centers of the spheres
    centers = []
    for i in range(cube_dimension):
        for j in range(cube_dimension):
            for k in range(cube_dimension):
                centers.append(
                    (start_x + i * sphere_diameter, start_y + j * sphere_diameter, start_z + k * sphere_diameter)
                )

    return centers
