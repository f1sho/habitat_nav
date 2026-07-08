import habitat_sim
import numpy as np

class HabitatEnv:
    def __init__(self, scene_path, navmesh_path):
        self.scene_path = scene_path
        self.navmesh_path = navmesh_path
        self.sim = self._init_sim()
        self._load_navmesh()

    def _init_sim(self):
        print(f"Loading scene: {self.scene_path}")
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = self.scene_path

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        
        # Attach an RGB color sensor to the robot
        rgb_sensor = habitat_sim.CameraSensorSpec()
        rgb_sensor.uuid = "color_sensor"
        rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor.resolution = [480, 640] 
        rgb_sensor.position = [0.0, 1.5, 0.0] 

        # --- MODIFICATION START ---
        # Context: Removing the physical depth sensor to enforce purely monocular vision.
        # Changes: Commented out all depth sensor configurations.
        
        # Attach a Depth sensor to the robot (DISABLED FOR PURE MONOCULAR TEST)
        # depth_sensor = habitat_sim.CameraSensorSpec()
        # depth_sensor.uuid = "depth_sensor"
        # depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
        # depth_sensor.resolution = [480, 640]
        # depth_sensor.position = [0.0, 1.5, 0.0]

        # Only pass the rgb_sensor to the agent specification
        agent_cfg.sensor_specifications = [rgb_sensor]
        # --- MODIFICATION END ---

        # Create a clean MetadataMediator for the default dataset
        mm = habitat_sim.metadata.MetadataMediator()
        stage_manager = mm.stage_template_manager
        
        # Register the ply file as a valid stage WITHOUT forcing any rotation.
        # The mesh in the habitat/ folder is already correctly oriented (Y-up).
        template = stage_manager.create_new_template(self.scene_path, True)
        stage_manager.register_template(template, self.scene_path)

        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        cfg.metadata_mediator = mm
        
        return habitat_sim.Simulator(cfg)

    def _load_navmesh(self):
        # Load the original precomputed navmesh since the scene is now properly oriented
        print(f"Loading precomputed navmesh: {self.navmesh_path}")
        loaded = self.sim.pathfinder.load_nav_mesh(self.navmesh_path)
        print(f"Navmesh loaded: {loaded}")
        if not loaded:
            raise RuntimeError("Failed to load navmesh!")

    def get_observations(self):
        return self.sim.get_sensor_observations()

    def step(self, action):
        return self.sim.step(action)

    def close(self):
        self.sim.close()