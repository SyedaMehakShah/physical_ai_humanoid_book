---
sidebar_position: 2
---

# Gazebo Physics Simulation

## Learning Objectives
- Understand Gazebo's role in robotics simulation
- Learn to configure physics parameters for humanoid robots
- Create and customize simulation environments
- Integrate Gazebo with ROS 2 for robot simulation

## Intuition

Gazebo is like a physics laboratory in your computer where you can test robot behaviors under realistic physical conditions. Just as engineers test aircraft designs in wind tunnels, roboticists use Gazebo to test robot designs under various physical conditions - gravity, friction, collisions, and other forces. For humanoid robots, Gazebo is particularly valuable because it can accurately simulate the complex dynamics of walking, balancing, and interacting with the environment.

## Concept

Gazebo is a 3D dynamic simulator with accurate physics simulation based on ODE (Open Dynamics Engine), Bullet Physics, or DART. It provides realistic rendering, sensor simulation, and a plugin architecture that allows custom behaviors. For ROS 2 integration, Gazebo provides the `gazebo_ros_pkgs` that enable communication between simulated robots and ROS 2 nodes.

## Gazebo World File

Here's an example world file for a humanoid testing environment:

```xml title="humanoid_test.world"
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_test_world">
    <!-- Include ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Physics engine configuration -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Add obstacles for testing -->
    <model name="box_obstacle">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.5 0.5 1.0</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.5 0.5 1.0</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Add ramp for walking tests -->
    <model name="ramp">
      <pose>-2 0 0 0 0.3 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <mesh>
              <uri>model://ramp/meshes/ramp.dae</uri>
            </mesh>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <mesh>
              <uri>model://ramp/meshes/ramp.dae</uri>
            </mesh>
          </geometry>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## Launch File for Gazebo Simulation

```xml title="launch/humanoid_gazebo.launch.py">
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    ld = LaunchDescription()

    # Launch arguments
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='humanoid_test.world',
        description='Choose one of the world files from `/your_robot_gazebo/worlds`'
    )

    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('gazebo_ros'),
            '/launch/gazebo.launch.py'
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                get_package_share_directory('your_robot_gazebo'),
                'worlds',
                LaunchConfiguration('world')
            ])
        }.items()
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'humanoid_robot',
            '-x', '0', '-y', '0', '-z', '1.0'
        ],
        output='screen'
    )

    # Add actions to launch description
    ld.add_action(world_arg)
    ld.add_action(gazebo)
    ld.add_action(robot_state_publisher)
    ld.add_action(spawn_entity)

    return ld
```

## Physics Parameters for Humanoid Robots

For humanoid robots, special attention should be paid to:

1. **Time Step**: Smaller time steps for more accurate simulation
2. **Real-time Factor**: Balance between simulation speed and accuracy
3. **Contact Parameters**: Proper friction and restitution for stable walking
4. **Solver Parameters**: Iterations and tolerance for stable dynamics

```xml
<physics name="humanoid_physics" type="ode">
  <!-- Smaller time step for humanoid stability -->
  <max_step_size>0.001</max_step_size>
  <real_time_factor>0.5</real_time_factor>
  <real_time_update_rate>1000.0</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>

  <ode>
    <!-- Contact parameters for stable walking -->
    <solver>
      <type>quick</type>
      <iters>1000</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.000001</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

## Sensor Simulation in Gazebo

Gazebo can simulate various sensors commonly used in humanoid robots:

```xml title="humanoid_with_sensors.urdf.xacro">
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_sensors">
  <!-- IMU sensor -->
  <gazebo reference="torso_link">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <imu>
        <angular_velocity>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.02</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.02</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.02</stddev>
            </noise>
          </z>
        </angular_velocity>
        <linear_acceleration>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.17</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.17</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.17</stddev>
            </noise>
          </z>
        </linear_acceleration>
      </imu>
    </sensor>
  </gazebo>

  <!-- Camera sensor -->
  <gazebo reference="head_link">
    <sensor name="camera" type="camera">
      <camera>
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>10</far>
        </clip>
      </camera>
      <always_on>true</always_on>
      <update_rate>30</update_rate>
      <visualize>true</visualize>
    </sensor>
  </gazebo>
</robot>
```

## Best Practices for Humanoid Simulation

1. **Start Simple**: Begin with basic models and gradually add complexity
2. **Validate Physics**: Ensure simulation matches real-world behavior
3. **Use Appropriate Solvers**: Adjust solver parameters for humanoid stability
4. **Simulate Sensor Noise**: Include realistic noise models for robustness
5. **Test Edge Cases**: Simulate challenging scenarios safely

## Exercises

1. Create a Gazebo world with multiple humanoid robots interacting
2. Implement a balance controller that works in simulation
3. Add sensor noise models to make simulation more realistic

## Summary

Gazebo provides the physics foundation for humanoid robot simulation. By properly configuring physics parameters, sensors, and environments, you can create realistic simulation scenarios that closely match real-world robot behavior, enabling safe and effective development of humanoid robot systems.