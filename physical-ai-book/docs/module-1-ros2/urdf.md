---
sidebar_position: 6
---

# Unified Robot Description Format (URDF)

## Learning Objectives
- Understand the structure and components of URDF files
- Learn how to describe humanoid robot kinematics using URDF
- Create and visualize simple humanoid robot models
- Integrate URDF with ROS 2 robot state publishing

## Intuition

URDF (Unified Robot Description Format) is like a digital blueprint for a robot. Just as architectural blueprints describe the structure of a building with dimensions, materials, and relationships between components, URDF describes the physical structure of a robot with links, joints, and their relationships. For a humanoid robot, URDF defines everything from the location of each joint to the physical properties of each body part.

## Concept

URDF is an XML-based format for representing a robot model. It describes the robot's physical structure including links (rigid parts), joints (connections between links), inertial properties, visual and collision representations, and transmission elements. URDF is used by ROS 2 tools for visualization, simulation, and kinematic calculations.

## Minimal Example

Here's a simplified URDF for a basic humanoid leg:

```xml title="humanoid_leg.urdf"
<?xml version="1.0"?>
<robot name="humanoid_leg">
  <!-- Base link (pelvis) -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.2 0.2 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.2 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Hip joint -->
  <joint name="hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="thigh_link"/>
    <origin xyz="0 0 -0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <!-- Thigh link -->
  <link name="thigh_link">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="3.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Knee joint -->
  <joint name="knee_joint" type="revolute">
    <parent link="thigh_link"/>
    <child link="shank_link"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.35" effort="100" velocity="1"/>
  </joint>

  <!-- Shank (lower leg) link -->
  <link name="shank_link">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.04"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>
</robot>
```

## Robot State Publishing

To visualize and work with URDF in ROS 2, you typically use the robot_state_publisher:

```python title="state_publisher_node.py"
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math

class StatePublisher(Node):
    def __init__(self):
        super().__init__('state_publisher')

        # Publisher for joint states
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        # Transform broadcaster for TF
        self.br = TransformBroadcaster(self)

        # Timer for publishing state
        self.timer = self.create_timer(0.05, self.publish_state)  # 20Hz

        # Initialize joint positions
        self.joint_names = ['hip_joint', 'knee_joint']
        self.joint_positions = [0.0, 0.0]

        self.get_logger().info('State Publisher initialized')

    def publish_state(self):
        # Create joint state message
        msg = JointState()
        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # Simulate moving joints
        time_sec = self.get_clock().now().nanoseconds / 1e9
        self.joint_positions[0] = math.sin(time_sec) * 0.5
        self.joint_positions[1] = math.cos(time_sec) * 0.5

        self.joint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    publisher = StatePublisher()

    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## URDF Components

### Links
- **Visual**: How the link appears in visualization
- **Collision**: How the link interacts in physics simulation
- **Inertial**: Physical properties for dynamics simulation

### Joints
- **Type**: revolute, continuous, prismatic, fixed, etc.
- **Limits**: Range of motion and physical constraints
- **Axis**: Direction of joint movement

### Materials
- **Color**: RGBA values for visualization
- **Textures**: Optional texture mapping

## Best Practices for Humanoid URDF

1. **Use Standard Naming Conventions**:
   - Joint names: `left_hip_yaw`, `right_knee`, etc.
   - Link names: `base_link`, `left_thigh`, `right_shank`, etc.

2. **Include Proper Inertial Properties**:
   - Accurate mass and inertia values are crucial for simulation
   - Use CAD software to calculate these values when possible

3. **Separate Visual and Collision Geometry**:
   - Visual: Detailed models for rendering
   - Collision: Simplified models for physics

4. **Use Xacro for Complex Models**:
   - Xacro allows macros and calculations to simplify complex URDFs

## Exercises

1. Create a URDF for a simple humanoid upper body (torso, arms, head)
2. Implement a joint state publisher that simulates walking motion
3. Design a URDF that includes sensors (IMU, cameras) in appropriate locations

## Summary

URDF provides the digital blueprint for robot models in ROS 2. For humanoid robotics, properly designed URDF files are essential for simulation, visualization, and kinematic calculations. Understanding URDF structure is crucial for working with humanoid robot models in both simulation and real-world applications.