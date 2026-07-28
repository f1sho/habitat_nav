import habitat_sim
import numpy as np


class HabitatEnv:
    """Create a Habitat-Sim navigation environment with aligned RGB-D sensors."""

    def __init__(
        self,
        scene_path,
        navmesh_path,
        enable_depth=False,
        camera_height=1.5,
        camera_pitch_deg=15.0,
    ):
        self.scene_path = scene_path
        self.navmesh_path = navmesh_path
        self.enable_depth = enable_depth
        self.camera_height = float(camera_height)
        self.camera_pitch_deg = float(camera_pitch_deg)

        if self.camera_height <= 0.0:
            raise ValueError("camera_height must be greater than zero.")
        if not 0.0 <= self.camera_pitch_deg < 90.0:
            raise ValueError(
                "camera_pitch_deg must be in the range [0, 90)."
            )

        self.sim = self._init_sim()
        self._load_navmesh()

    def _init_sim(self):
        """Initialize Habitat-Sim and attach aligned RGB and depth cameras."""
        print(f"Loading scene: {self.scene_path}")
        print(
            "Camera configuration: "
            f"height={self.camera_height:.2f}m, "
            f"downward pitch={self.camera_pitch_deg:.1f}deg"
        )

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = self.scene_path

        agent_cfg = habitat_sim.agent.AgentConfiguration()

        # Habitat-Sim applies sensor orientation as X, Y, and Z rotations in
        # radians. A negative X rotation points the camera downward.
        downward_pitch_rad = float(
            -np.deg2rad(self.camera_pitch_deg)
        )
        sensor_orientation = [
            downward_pitch_rad,
            0.0,
            0.0,
        ]

        # Attach an RGB color sensor to the robot.
        rgb_sensor = habitat_sim.CameraSensorSpec()
        rgb_sensor.uuid = "color_sensor"
        rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        rgb_sensor.resolution = [480, 640]
        rgb_sensor.position = [0.0, self.camera_height, 0.0]
        rgb_sensor.orientation = sensor_orientation
        rgb_sensor.hfov = 90.0

        sensor_specs = [rgb_sensor]

        if self.enable_depth:
            # Match every depth-camera parameter to the RGB camera so depth
            # pixels remain aligned with segmentation polygons.
            depth_sensor = habitat_sim.CameraSensorSpec()
            depth_sensor.uuid = "depth_sensor"
            depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            depth_sensor.resolution = [480, 640]
            depth_sensor.position = [0.0, self.camera_height, 0.0]
            depth_sensor.orientation = sensor_orientation
            depth_sensor.hfov = 90.0

            sensor_specs.append(depth_sensor)

        # Attach all configured sensors to the same agent.
        agent_cfg.sensor_specifications = sensor_specs

        # Create a clean MetadataMediator for the default dataset.
        mm = habitat_sim.metadata.MetadataMediator()
        stage_manager = mm.stage_template_manager

        # Register the PLY file without forcing a rotation because the mesh in
        # the Habitat folder is already correctly oriented with Y as up.
        template = stage_manager.create_new_template(
            self.scene_path,
            True,
        )
        stage_manager.register_template(
            template,
            self.scene_path,
        )

        cfg = habitat_sim.Configuration(
            sim_cfg,
            [agent_cfg],
        )
        cfg.metadata_mediator = mm

        return habitat_sim.Simulator(cfg)

    def _load_navmesh(self):
        """Load the precomputed navigation mesh."""
        print(
            "Loading precomputed navmesh: "
            f"{self.navmesh_path}"
        )
        loaded = self.sim.pathfinder.load_nav_mesh(
            self.navmesh_path
        )
        print(f"Navmesh loaded: {loaded}")

        if not loaded:
            raise RuntimeError("Failed to load navmesh!")

    def get_observations(self):
        """Return observations from all enabled sensors."""
        return self.sim.get_sensor_observations()

    def step(self, action):
        """Apply one discrete navigation action."""
        return self.sim.step(action)

    def close(self):
        """Close the Habitat-Sim instance."""
        self.sim.close()
