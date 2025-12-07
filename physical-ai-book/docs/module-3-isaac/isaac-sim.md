---
sidebar_position: 2
---

# Isaac Sim: Advanced Physics Simulation

## Learning Objectives
- Master Isaac Sim for high-fidelity humanoid robot simulation
- Understand USD (Universal Scene Description) for scene creation
- Configure advanced physics properties for humanoid robots
- Generate synthetic training data for AI models
- Implement domain randomization techniques

## Intuition

Isaac Sim is like a virtual physics laboratory powered by NVIDIA's GPU technology, where you can create incredibly realistic robot simulations. Think of it as a combination of the most advanced video game engine and a physics laboratory, where every detail of the robot's interaction with its environment is calculated with extreme accuracy. For humanoid robots, this means you can simulate complex behaviors like walking, balancing, and manipulation with physics accuracy that closely matches the real world.

## Concept

Isaac Sim is built on NVIDIA Omniverse and uses the Universal Scene Description (USD) format for scene representation. It provides:
- **PhysX Physics Engine**: For accurate contact simulation and dynamics
- **RTX Rendering**: For photorealistic visuals and synthetic data generation
- **Omniverse Connectors**: For integration with other 3D tools
- **ROS 2 Bridge**: For robotics application development

## USD Scene Creation

USD (Universal Scene Description) is the foundation of Isaac Sim scenes:

```python title="usd_scene_creation.py"
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.prims import create_primitive, get_prim_at_path
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.carb import wait_stage_loading
import numpy as np

class IsaacSimScene:
    def __init__(self):
        # Initialize world with 1 meter = 1 stage unit
        self.world = World(stage_units_in_meters=1.0)
        self.setup_scene()

    def setup_scene(self):
        """Create a complete scene with humanoid robot"""
        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Add textured ground plane
        from omni.isaac.core.objects import GroundPlane
        self.ground_plane = self.world.scene.add(
            GroundPlane(
                prim_path="/World/defaultGroundPlane",
                name="default_ground_plane",
                size=1000.0,
                height=0.0,
                visible=True
            )
        )

        # Add lighting
        self.setup_lighting()

        # Add humanoid robot
        self.add_humanoid_robot()

        # Add environment objects
        self.add_environment_objects()

    def setup_lighting(self):
        """Setup realistic lighting for the scene"""
        from omni.isaac.core.utils.prims import create_prim
        from pxr import Gf

        # Create dome light for ambient lighting
        create_prim(
            prim_path="/World/DomeLight",
            prim_type="DomeLight",
            position=np.array([0, 0, 0]),
            attributes={"inputs:color": Gf.Vec3f(0.2, 0.2, 0.2)}
        )

        # Add directional light for shadows
        create_prim(
            prim_path="/World/DirectionalLight",
            prim_type="DistantLight",
            position=np.array([0, 0, 10]),
            rotation=np.array([0, 45, 0]),
            attributes={
                "inputs:color": Gf.Vec3f(0.8, 0.8, 0.8),
                "inputs:intensity": 3000
            }
        )

    def add_humanoid_robot(self):
        """Add a humanoid robot to the scene"""
        assets_root_path = get_assets_root_path()
        if assets_root_path is not None:
            # Use a basic articulated robot as placeholder
            # In practice, you'd use a more complex humanoid model
            add_reference_to_stage(
                usd_path=f"{assets_root_path}/Isaac/Robots/Franka/franka.usd",
                prim_path="/World/Robot"
            )
        else:
            # Create a simple articulated robot for demonstration
            self.create_simple_articulated_robot()

    def create_simple_articulated_robot(self):
        """Create a simple articulated robot"""
        # Create base
        base = create_primitive(
            prim_path="/World/Robot/base",
            prim_type="Cylinder",
            position=np.array([0, 0, 0.5]),
            scale=np.array([0.2, 0.2, 0.2]),
            orientation=np.array([1, 0, 0, 0])
        )

        # Create additional links and joints would go here
        # This is a simplified example - full humanoid would have many more parts

    def add_environment_objects(self):
        """Add objects to create a realistic environment"""
        from omni.isaac.core.objects import DynamicCuboid

        # Add some objects for interaction
        self.box1 = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/box1",
                name="box1",
                position=np.array([1.0, 0.0, 0.5]),
                size=0.2,
                mass=0.5
            )
        )

        self.box2 = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/box2",
                name="box2",
                position=np.array([-1.0, 0.5, 0.5]),
                size=0.15,
                mass=0.3
            )
        )

def main():
    # Initialize Isaac Sim
    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({"headless": False})

    try:
        # Create the scene
        scene = IsaacSimScene()

        # Reset the world
        scene.world.reset()

        # Run simulation for a number of steps
        for i in range(500):
            scene.world.step(render=True)

            if i % 100 == 0:
                print(f"Simulation step: {i}")

    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
```

## Physics Configuration for Humanoid Robots

Proper physics configuration is crucial for stable humanoid simulation:

```python title="physics_configuration.py"
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import UsdPhysics, PhysicsSchemaTools
import numpy as np

class PhysicsConfigurator:
    def __init__(self, stage):
        self.stage = stage

    def configure_humanoid_physics(self, robot_path):
        """Configure physics properties for humanoid robot"""
        # Get the robot prim
        robot_prim = get_prim_at_path(robot_path)

        # Configure articulation root
        self._configure_articulation_root(robot_path)

        # Configure individual links
        self._configure_links(robot_path)

        # Configure joints for stability
        self._configure_joints(robot_path)

    def _configure_articulation_root(self, robot_path):
        """Configure the articulation root for the robot"""
        from omni.isaac.core.utils.prims import set_targets

        # Set up articulation root
        robot_prim = get_prim_at_path(robot_path)

        # Add articulation root API
        UsdPhysics.ArticulationRootAPI.Apply(robot_prim)

    def _configure_links(self, robot_path):
        """Configure physics properties for robot links"""
        # Example: Configure a specific link
        link_path = f"{robot_path}/torso_link"
        link_prim = get_prim_at_path(link_path)

        if link_prim:
            # Apply rigid body API
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(link_prim)
            rigid_body_api.CreateEnableGyroscopicForcesAttr(True)

            # Set mass properties
            mass_api = UsdPhysics.MassAPI.Apply(link_prim)
            mass_api.CreateMassAttr(10.0)  # 10kg

    def _configure_joints(self, robot_path):
        """Configure joint properties for stability"""
        # Example: Configure hip joint
        joint_path = f"{robot_path}/hip_joint"
        joint_prim = get_prim_at_path(joint_path)

        if joint_prim:
            # Apply joint API
            joint_api = UsdPhysics.JointAPI.Apply(joint_prim)

            # Configure joint limits
            joint_api.CreateLowerLimitAttr(-1.57)  # -90 degrees
            joint_api.CreateUpperLimitAttr(1.57)   # 90 degrees

            # Configure drive for active joints
            drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            drive_api.CreateStiffnessAttr(1000.0)
            drive_api.CreateDampingAttr(100.0)

    def setup_contact_properties(self):
        """Configure contact properties for stable interactions"""
        # Set global contact properties
        scene_prim = get_prim_at_path("/World")

        # Configure default contact properties
        UsdPhysics.SceneAPI.Apply(scene_prim)
        scene_physics = UsdPhysics.SceneAPI.Get(self.stage, scene_prim.GetPath())

        # Set contact offset and rest offset
        scene_physics.CreateContactOffsetThresholdAttr(0.001)
        scene_physics.CreateRestOffsetThresholdAttr(0.0)
```

## Sensor Simulation in Isaac Sim

Isaac Sim provides high-quality sensor simulation:

```python title="sensor_simulation.py"
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.prims import get_prim_at_path
import numpy as np

class IsaacSimSensors:
    def __init__(self, robot_prim_path):
        self.robot_prim_path = robot_prim_path
        self.cameras = []
        self.setup_sensors()

    def setup_sensors(self):
        """Setup various sensors for the robot"""
        # Add RGB camera
        self.add_camera_sensor(
            prim_path=f"{self.robot_prim_path}/head_camera",
            position=np.array([0.1, 0, 0.1]),
            orientation=np.array([0.707, 0, 0.707, 0])  # Look forward
        )

        # Add IMU sensor
        self.add_imu_sensor(
            prim_path=f"{self.robot_prim_path}/imu",
            position=np.array([0, 0, 0.2])  # In torso
        )

        # Add force/torque sensors at feet
        self.add_force_torque_sensors()

    def add_camera_sensor(self, prim_path, position, orientation):
        """Add RGB camera sensor"""
        camera = Camera(
            prim_path=prim_path,
            position=position,
            orientation=orientation
        )

        # Configure camera properties
        camera.focal_length = 24.0
        camera.focus_distance = 400.0
        camera.horizontal_aperture = 20.955
        camera.vertical_aperture = 15.2908

        # Set resolution
        camera.resolution = (640, 480)

        self.cameras.append(camera)

    def add_imu_sensor(self, prim_path, position):
        """Add IMU sensor"""
        # In Isaac Sim, IMU is typically simulated through the physics engine
        # and accessed via the robot's state
        pass

    def add_force_torque_sensors(self):
        """Add force/torque sensors to feet"""
        # Configure force/torque sensing at contact points
        # This would typically be done at joint level in Isaac Sim
        pass

    def get_camera_data(self, camera_idx=0):
        """Get data from camera sensor"""
        if camera_idx < len(self.cameras):
            # Capture RGB image
            rgb_image = self.cameras[camera_idx].get_rgb()

            # Capture depth image
            depth_image = self.cameras[camera_idx].get_depth()

            return {
                'rgb': rgb_image,
                'depth': depth_image,
                'timestamp': self.cameras[camera_idx].get_timestamp()
            }
        return None
```

## Domain Randomization for Robust Training

Domain randomization helps create robust AI models:

```python title="domain_randomization.py"
import numpy as np
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import Gf, UsdGeom
import random

class DomainRandomizer:
    def __init__(self, world):
        self.world = world
        self.randomization_ranges = {
            'lighting': {
                'intensity': (0.5, 2.0),
                'color_temperature': (0.8, 1.2)
            },
            'materials': {
                'roughness': (0.1, 0.9),
                'metallic': (0.0, 0.5),
                'specular': (0.0, 1.0)
            },
            'physics': {
                'friction': (0.3, 0.9),
                'restitution': (0.0, 0.2)
            },
            'textures': {
                'scale': (0.5, 2.0),
                'rotation': (0, 360)
            }
        }

    def randomize_lighting(self):
        """Randomize lighting conditions"""
        # Get all light prims in the scene
        light_prims = [prim for prim in self.world.scene.stage.Traverse()
                      if prim.GetTypeName() in ["DistantLight", "DomeLight", "SphereLight"]]

        for light_prim in light_prims:
            # Randomize intensity
            min_int, max_int = self.randomization_ranges['lighting']['intensity']
            new_intensity = random.uniform(min_int, max_int)
            light_prim.GetAttribute("inputs:intensity").Set(new_intensity)

            # Randomize color
            min_col, max_col = self.randomization_ranges['lighting']['color_temperature']
            color_scale = [
                random.uniform(min_col, max_col),
                random.uniform(min_col, max_col),
                random.uniform(min_col, max_col)
            ]
            light_prim.GetAttribute("inputs:color").Set(Gf.Vec3f(*color_scale))

    def randomize_materials(self):
        """Randomize material properties"""
        # Get all material prims
        material_prims = [prim for prim in self.world.scene.stage.Traverse()
                         if 'Material' in prim.GetTypeName()]

        for material_prim in material_prims:
            # Randomize roughness
            min_rough, max_rough = self.randomization_ranges['materials']['roughness']
            roughness = random.uniform(min_rough, max_rough)
            if material_prim.GetAttribute("inputs:roughness"):
                material_prim.GetAttribute("inputs:roughness").Set(roughness)

            # Randomize metallic
            min_metal, max_metal = self.randomization_ranges['materials']['metallic']
            metallic = random.uniform(min_metal, max_metal)
            if material_prim.GetAttribute("inputs:metallic"):
                material_prim.GetAttribute("inputs:metallic").Set(metallic)

    def randomize_physics_properties(self):
        """Randomize physics properties"""
        # Randomize friction for all rigid bodies
        for prim in self.world.scene.stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_api = UsdPhysics.RigidBodyAPI(prim)

                # Randomize friction
                min_fric, max_fric = self.randomization_ranges['physics']['friction']
                friction = random.uniform(min_fric, max_fric)

                # Apply to material if it exists
                material_rel = prim.GetRelationship("physics:material")
                if material_rel:
                    material_path = material_rel.GetTargets()[0] if material_rel.GetTargets() else None
                    if material_path:
                        material_prim = self.world.scene.stage.GetPrimAtPath(material_path)
                        if material_prim:
                            material_prim.GetAttribute("physics:staticFriction").Set(friction)
                            material_prim.GetAttribute("physics:dynamicFriction").Set(friction)

    def randomize_textures(self):
        """Randomize texture properties"""
        # Randomize texture scales and rotations
        texture_prims = [prim for prim in self.world.scene.stage.Traverse()
                        if 'Texture' in prim.GetName()]

        for texture_prim in texture_prims:
            # Randomize scale
            min_scale, max_scale = self.randomization_ranges['textures']['scale']
            scale_factor = random.uniform(min_scale, max_scale)

            # Apply scale to texture transform
            if texture_prim.GetAttribute("inputs:scale"):
                texture_prim.GetAttribute("inputs:scale").Set(Gf.Vec2f(scale_factor, scale_factor))

    def apply_randomization_step(self):
        """Apply domain randomization for current simulation step"""
        self.randomize_lighting()
        self.randomize_materials()
        self.randomize_physics_properties()
        self.randomize_textures()

        # Notify Isaac Sim of changes
        self.world.step(render=False)
```

## Synthetic Data Generation

Isaac Sim excels at generating synthetic training data:

```python title="synthetic_data_generation.py"
import omni
import numpy as np
from PIL import Image
import json
import os

class SyntheticDataGenerator:
    def __init__(self, output_dir="synthetic_data"):
        self.output_dir = output_dir
        self.scene_counter = 0

        # Create output directories
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/labels", exist_ok=True)
        os.makedirs(f"{output_dir}/depth", exist_ok=True)

    def capture_synthetic_data(self, camera_sensor, robot_state, scene_description):
        """Capture synthetic data with ground truth labels"""
        # Capture RGB image
        rgb_data = camera_sensor.get_rgb()
        rgb_image = Image.fromarray((rgb_data * 255).astype(np.uint8))

        # Capture depth image
        depth_data = camera_sensor.get_depth()
        depth_image = Image.fromarray((depth_data * 1000).astype(np.uint16))  # Scale for 16-bit

        # Generate semantic segmentation (simplified)
        segmentation = self.generate_segmentation(camera_sensor)
        seg_image = Image.fromarray(segmentation.astype(np.uint8))

        # Create ground truth labels
        labels = {
            "robot_pose": robot_state["pose"].tolist(),
            "joint_positions": robot_state["joint_positions"],
            "object_poses": scene_description["object_poses"],
            "camera_intrinsics": camera_sensor.get_intrinsics(),
            "timestamp": camera_sensor.get_timestamp()
        }

        # Save data
        image_name = f"scene_{self.scene_counter:06d}"

        rgb_image.save(f"{self.output_dir}/images/{image_name}.png")
        depth_image.save(f"{self.output_dir}/depth/{image_name}.png")
        seg_image.save(f"{self.output_dir}/labels/{image_name}_seg.png")

        with open(f"{self.output_dir}/labels/{image_name}.json", 'w') as f:
            json.dump(labels, f, indent=2)

        self.scene_counter += 1

        return {
            "image_path": f"{self.output_dir}/images/{image_name}.png",
            "labels_path": f"{self.output_dir}/labels/{image_name}.json",
            "scene_id": self.scene_counter - 1
        }

    def generate_segmentation(self, camera_sensor):
        """Generate semantic segmentation mask"""
        # In a real implementation, this would use Isaac Sim's segmentation capabilities
        # This is a simplified placeholder
        width, height = camera_sensor.resolution
        segmentation = np.zeros((height, width), dtype=np.uint8)

        # Add simple segmentation regions (in practice, Isaac Sim provides this)
        # This would identify different objects in the scene
        return segmentation

    def create_varied_scenes(self, scene_generator, num_scenes=1000):
        """Create varied scenes for diverse training data"""
        for i in range(num_scenes):
            # Randomize scene
            scene_description = scene_generator.randomize_scene()

            # Capture data from multiple viewpoints
            for view_angle in [0, 45, 90, 135, 180, 225, 270, 315]:
                scene_generator.set_camera_angle(view_angle)
                robot_state = scene_generator.get_robot_state()
                camera_data = scene_generator.get_camera_data()

                self.capture_synthetic_data(
                    camera_data,
                    robot_state,
                    scene_description
                )

                print(f"Generated scene {i*8 + int(view_angle/45)}/{num_scenes*8}")
```

## Performance Optimization

Optimizing Isaac Sim for real-time performance:

```python title="performance_optimization.py"
class PerformanceOptimizer:
    def __init__(self, world):
        self.world = world
        self.settings = {
            'max_substeps': 1,
            'min_frame_step_size': 1.0/240.0,  # 240 Hz physics
            'max_frame_step_size': 1.0/60.0,   # 60 Hz max
            'solver_position_iteration_count': 4,
            'solver_velocity_iteration_count': 1
        }

    def optimize_physics_settings(self):
        """Optimize physics settings for performance"""
        from omni.isaac.core.utils.stage import get_current_stage
        from pxr import UsdPhysics

        stage = get_current_stage()
        scene_prim = stage.GetPrimAtPath("/physicsScene")

        if scene_prim:
            phys_scene = UsdPhysics.SceneAPI.Get(stage, scene_prim.GetPath())

            # Set solver settings for performance
            phys_scene.CreateSolverPositionIterationCountAttr(self.settings['solver_position_iteration_count'])
            phys_scene.CreateSolverVelocityIterationCountAttr(self.settings['solver_velocity_iteration_count'])

            # Set substep settings
            phys_scene.CreateMaxSubStepsAttr(self.settings['max_substeps'])

    def optimize_rendering(self):
        """Optimize rendering settings"""
        # Reduce rendering quality for training speed
        carb_settings = omni.appwindow.get_default_app_window().get_carb_settings_interface()

        # Reduce MSAA
        carb_settings.set_int("rtx-hair:antialiasing:aaMode", 0)  # Off

        # Reduce ray tracing effects
        carb_settings.set_int("rtx-globalillumination:giMode", 0)  # No GI

    def selective_rendering(self, render_enabled=True):
        """Enable/disable rendering based on needs"""
        if not render_enabled:
            # For headless training, disable rendering
            self.world._world_settings.render = False
            self.world._world_settings.simulation_rendering_interval = 0
        else:
            # For visualization, enable rendering
            self.world._world_settings.render = True
            self.world._world_settings.simulation_rendering_interval = 1
```

## Exercises

1. Create a complex humanoid scene with multiple robots and interactive objects
2. Implement domain randomization for robust humanoid walking controllers
3. Generate synthetic dataset for computer vision tasks in humanoid robotics

## Summary

Isaac Sim provides state-of-the-art simulation capabilities for humanoid robotics, combining accurate physics simulation with photorealistic rendering. By mastering USD scene creation, physics configuration, and domain randomization techniques, you can create powerful simulation environments that enable the development of robust humanoid robot systems. The combination of realistic physics and synthetic data generation makes Isaac Sim an invaluable tool for advancing humanoid robotics research and development.