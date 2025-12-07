---
sidebar_position: 1
---

# Tools and Technologies for Physical AI & Humanoid Robotics

## Learning Objectives
- Understand the essential tools and technologies for humanoid robotics development
- Learn to set up and configure development environments
- Master the use of simulation tools and frameworks
- Understand the integration of various tools in the development workflow

## Intuition

The tools and technologies for humanoid robotics are like the instruments in an orchestra - each serves a specific purpose, but they must work harmoniously together to create beautiful music. Just as a conductor coordinates musicians, a robotics engineer must coordinate various tools and technologies to create sophisticated humanoid robots. Having the right tools and knowing how to use them effectively is essential for success in this complex field.

## Essential Development Tools

### 1. ROS 2 (Robot Operating System 2)

ROS 2 is the middleware that connects all parts of your robot system:

```bash title="ROS 2 Installation and Setup"
# Install ROS 2 Humble Hawksbill (recommended for humanoid robotics)
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update

# Add ROS 2 GPG key and repository
sudo apt update && sudo apt install curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros-keyring.gpg | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt upgrade

# Install ROS 2 packages
sudo apt install ros-humble-desktop-full
sudo apt install python3-argcomplete python3-colcon-common-extensions

# Source ROS 2 environment
source /opt/ros/humble/setup.bash
```

```python title="ROS 2 Python Client Library (rclpy) Example"
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import String

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Publishers
        self.joint_pub = self.create_publisher(JointState, 'joint_commands', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)
        self.status_sub = self.create_subscription(
            String, 'robot_status', self.status_callback, 10)

        # Timer for control loop
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20Hz

        self.get_logger().info('Humanoid Controller initialized')

    def joint_callback(self, msg):
        """Process joint state messages"""
        self.get_logger().info(f'Received {len(msg.position)} joint positions')

    def status_callback(self, msg):
        """Process robot status messages"""
        self.get_logger().info(f'Robot status: {msg.data}')

    def control_loop(self):
        """Main control loop"""
        # Implement control logic here
        pass

def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2. Simulation Environments

#### Gazebo Simulation
Gazebo provides accurate physics simulation for humanoid robots:

```xml title="robot_model.gazebo.xacro"
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Gazebo-specific configurations -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid_robot</robotNamespace>
      <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
    </plugin>
  </gazebo>

  <!-- Link Gazebo materials -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
  </gazebo>

  <!-- Joint transmissions for Gazebo -->
  <transmission name="tran1">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="joint1">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="motor1">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>
</robot>
```

```bash title="Gazebo Launch Script"
#!/bin/bash

# Launch Gazebo with humanoid robot model
ros2 launch gazebo_ros gazebo.launch.py world:=my_humanoid_world.world &
sleep 5

# Spawn robot in Gazebo
ros2 run gazebo_ros spawn_entity.py -topic robot_description -entity humanoid_robot -x 0 -y 0 -z 1.0

# Launch robot controllers
ros2 launch my_robot_bringup robot.launch.py
```

#### Unity for High-Fidelity Simulation
Unity provides photorealistic rendering and human-robot interaction scenarios:

```csharp title="Unity ROS Connection Example"
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;

public class UnityRobotController : MonoBehaviour
{
    ROSConnection ros;
    string robotTopic = "unity_robot_position";

    void Start()
    {
        // Get ROS connection
        ros = ROSConnection.instance;

        // Subscribe to ROS topic
        ros.Subscribe<JointStateMsg>(robotTopic, OnJointStateReceived);
    }

    void OnJointStateReceived(JointStateMsg jointState)
    {
        // Update Unity robot model based on ROS joint states
        Debug.Log("Received joint states: " + jointState.position.Count);

        // Update robot visualization
        UpdateRobotVisualization(jointState);
    }

    void UpdateRobotVisualization(JointStateMsg jointState)
    {
        // Apply joint positions to Unity robot model
        // This would update the robot's visual representation
    }

    void Update()
    {
        // Send Unity robot position to ROS
        var robotPosition = new TransformMsg();
        robotPosition.translation = new Vector3Msg(transform.position.x,
                                                   transform.position.y,
                                                   transform.position.z);

        ros.Send("unity_robot_pose", robotPosition);
    }
}
```

### 3. NVIDIA Isaac Tools

#### Isaac Sim for Advanced Simulation
Isaac Sim provides GPU-accelerated simulation:

```python title="Isaac Sim Setup Example"
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path

class IsaacSimEnvironment:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.setup_environment()

    def setup_environment(self):
        """Setup Isaac Sim environment"""
        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Add humanoid robot
        assets_root_path = get_assets_root_path()
        if assets_root_path is not None:
            # Load robot from Isaac Sim assets
            add_reference_to_stage(
                usd_path=f"{assets_root_path}/Isaac/Robots/Humanoid/humanoid.usd",
                prim_path="/World/Humanoid"
            )

    def run_simulation(self):
        """Run the simulation"""
        self.world.reset()
        for i in range(1000):
            self.world.step(render=True)

            if i % 100 == 0:
                print(f"Simulation step: {i}")

# Example usage
if __name__ == "__main__":
    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({"headless": False})

    try:
        env = IsaacSimEnvironment()
        env.run_simulation()
    finally:
        simulation_app.close()
```

#### Isaac ROS for GPU-Accelerated Processing
Isaac ROS provides GPU-accelerated perception and processing:

```python title="Isaac ROS Pipeline Example"
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

class IsaacROSPerceptionNode(Node):
    def __init__(self):
        super().__init__('isaac_ros_perception')

        # Subscribe to camera image
        self.image_sub = self.create_subscription(
            Image, 'camera/image_rect_color', self.image_callback, 10)

        # Publish detections
        self.detection_pub = self.create_publisher(
            Detection2DArray, 'isaac_ros/detections', 10)

    def image_callback(self, msg):
        """Process image with Isaac ROS GPU acceleration"""
        # This would use Isaac ROS DNN packages
        # for GPU-accelerated processing
        pass

def main(args=None):
    rclpy.init(args=args)
    node = IsaacROSPerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Development Environment Setup

### 1. System Requirements and Setup

```bash title="System Setup Script"
#!/bin/bash

echo "Setting up development environment for Physical AI & Humanoid Robotics..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential development tools
sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-pip \
    python3-venv \
    wget \
    curl \
    gnupg \
    lsb-release \
    software-properties-common \
    vim \
    tmux \
    htop

# Install Python packages
pip3 install --upgrade pip
pip3 install numpy scipy matplotlib jupyter

# Install graphics drivers and CUDA (for GPU acceleration)
echo "Installing graphics drivers..."
sudo apt install -y nvidia-driver-535 nvidia-utils-535
sudo apt install -y nvidia-cuda-toolkit

# Install Unity Hub (for Unity development)
echo "Downloading Unity Hub..."
wget https://public-cdn.cloud.unity3d.com/hub/prod/UnityHub.AppImage
chmod +x UnityHub.AppImage
./UnityHub.AppImage --headless install

echo "Development environment setup complete!"
```

### 2. IDE and Editor Configuration

#### VS Code Configuration for Robotics Development
```json title=".vscode/settings.json"
{
    "python.defaultInterpreterPath": "/usr/bin/python3",
    "python.terminal.activateEnvironment": true,
    "editor.formatOnSave": true,
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Args": [
        "--max-line-length=120",
        "--ignore=E203,W503"
    ],
    "files.associations": {
        "*.msg": "python",
        "*.srv": "python",
        "*.action": "python"
    },
    "yaml.schemas": {
        "kubernetes": "/*.yaml"
    }
}
```

#### Extensions for Robotics Development
```json title=".vscode/extensions.json"
{
    "recommendations": [
        "ms-python.python",
        "ms-iot.vscode-ros",
        "redhat.vscode-yaml",
        "ms-vscode.cpptools",
        "smilerobotics.urdf",
        "ms-toolsai.jupyter",
        "ms-python.black-formatter",
        "charliermarsh.ruff"
    ]
}
```

## Version Control and Collaboration

### 1. Git Workflow for Robotics Projects

```bash title="Git Workflow Setup"
#!/bin/bash

# Initialize git repository
git init

# Create .gitignore for robotics projects
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.idea
.vscode/settings.json
.DS_Store
.pytest_cache/

# ROS specific
*.dae
*.stl
*.obj
*.fbx
build/
install/
log/

# Unity specific
Library/
Temp/
Obj/
Logs/
MemoryCaptures/
WebGLTemplates/
*.unityproj
*.userprefs
*.sln
*.csproj
*.pidb
*.user
*.userprefs
*.unity
*.asset
*.meta
*.prefab
*.mat
*.anim
*.controller
*.guiskin
*.fontsettings
*.giparams
*.noasset

# Simulation specific
*.world
*.world.bak
*.sdf
*.urdf
*.xacro

# Build artifacts
*.so
*.dll
*.dylib
build/
dist/
*.egg-info/
EOF

# Set up git hooks for code quality
mkdir -p .git/hooks
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/sh
# Pre-commit hook for code quality checks

echo "Running pre-commit checks..."

# Check Python syntax
python3 -m py_compile $(git diff --cached --name-only --diff-filter=ACMR | grep '\.py$')

if [ $? -ne 0 ]; then
    echo "Python syntax error detected. Commit aborted."
    exit 1
fi

echo "Pre-commit checks passed!"
exit 0
EOF

chmod +x .git/hooks/pre-commit

echo "Git workflow configured for robotics project!"
```

### 2. Documentation and Knowledge Management

#### MkDocs for Technical Documentation
```yaml title="mkdocs.yml"
site_name: Physical AI & Humanoid Robotics
nav:
  - Home: index.md
  - Getting Started: getting-started.md
  - ROS 2 Fundamentals: ros2-fundamentals.md
  - Simulation: simulation.md
  - Control Systems: control-systems.md
  - Safety: safety.md
  - Ethics: ethics.md
  - Resources: resources/
  - API Reference: api-reference.md

theme:
  name: material
  palette:
    primary: indigo
    accent: blue

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: google

markdown_extensions:
  - pymdownx.highlight
  - pymdownx.superfences
  - pymdownx.emoji:
      emoji_index: !!python/name:materialx.emoji.twemoji
      emoji_generator: !!python/name:materialx.emoji.to_svg
```

## Testing and Validation Tools

### 1. Unit Testing Framework

```python title="test_robot_controller.py"
import unittest
import numpy as np
from controller import RobotController

class TestRobotController(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.controller = RobotController()

    def test_joint_command_validation(self):
        """Test that joint commands are properly validated."""
        # Test valid joint positions
        valid_positions = [0.0, 0.5, -0.3]
        result = self.controller.validate_joint_commands(valid_positions)
        self.assertTrue(result['valid'])

        # Test invalid joint positions (exceed limits)
        invalid_positions = [10.0, -10.0, 5.0]  # Exceed typical joint limits
        result = self.controller.validate_joint_commands(invalid_positions)
        self.assertFalse(result['valid'])

    def test_trajectory_generation(self):
        """Test trajectory generation functionality."""
        start_pos = [0.0, 0.0, 0.0]
        end_pos = [1.0, 1.0, 1.0]
        trajectory = self.controller.generate_trajectory(start_pos, end_pos, duration=2.0)

        self.assertIsNotNone(trajectory)
        self.assertGreater(len(trajectory), 0)
        self.assertEqual(trajectory[0]['position'], start_pos)
        self.assertEqual(trajectory[-1]['position'], end_pos)

    def test_safety_limits(self):
        """Test safety limit enforcement."""
        # Test that commands exceeding safety limits are rejected
        dangerous_command = {
            'position': [100, 100, 100],  # Way beyond safe limits
            'velocity': [10, 10, 10],     # Too fast
            'effort': [1000, 1000, 1000]  # Too much force
        }

        safe_command = self.controller.enforce_safety_limits(dangerous_command)

        # Verify limits were applied
        self.assertLessEqual(max(safe_command['position']), 3.14)  # Reasonable joint limit
        self.assertLessEqual(max(safe_command['velocity']), 2.0)   # Reasonable velocity limit
        self.assertLessEqual(max(safe_command['effort']), 200.0)   # Reasonable effort limit

if __name__ == '__main__':
    unittest.main()
```

### 2. Simulation Testing Framework

```python title="simulation_test_framework.py"
import unittest
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool

class SimulationTestFramework(Node):
    def __init__(self):
        super().__init__('simulation_test_framework')

        # Test result tracking
        self.test_results = {}
        self.current_test = None

    def run_simulation_test(self, test_name, test_scenario):
        """Run a simulation-based test scenario."""
        self.current_test = test_name

        # Setup test environment
        self.setup_test_environment(test_scenario)

        # Execute test
        result = self.execute_test_scenario(test_scenario)

        # Record results
        self.test_results[test_name] = result

        # Cleanup
        self.cleanup_test_environment()

        return result

    def setup_test_environment(self, scenario):
        """Setup the simulation environment for the test."""
        # This would set up the specific conditions for the test
        # e.g., robot starting position, obstacles, etc.
        pass

    def execute_test_scenario(self, scenario):
        """Execute the test scenario in simulation."""
        # This would run the actual test in simulation
        # and collect results
        test_result = {
            'success': True,
            'metrics': {},
            'logs': [],
            'failures': []
        }

        # Example: Test robot navigation
        if scenario['type'] == 'navigation':
            test_result = self.test_navigation_scenario(scenario)
        elif scenario['type'] == 'manipulation':
            test_result = self.test_manipulation_scenario(scenario)
        elif scenario['type'] == 'interaction':
            test_result = self.test_interaction_scenario(scenario)

        return test_result

    def test_navigation_scenario(self, scenario):
        """Test robot navigation in simulation."""
        # Set up navigation goal
        goal_pose = scenario['goal_pose']

        # Send navigation command
        # Monitor robot progress
        # Evaluate success criteria

        return {
            'success': True,  # Simplified
            'metrics': {
                'time_to_goal': 15.0,
                'path_efficiency': 0.85,
                'collision_free': True
            },
            'logs': ['Navigation test completed successfully'],
            'failures': []
        }

    def test_manipulation_scenario(self, scenario):
        """Test robot manipulation in simulation."""
        # Set up manipulation task
        # Monitor robot's ability to grasp and manipulate objects
        # Evaluate success criteria

        return {
            'success': True,  # Simplified
            'metrics': {
                'grasp_success_rate': 0.95,
                'precision': 0.01,  # 1cm precision
                'task_completion_time': 10.0
            },
            'logs': ['Manipulation test completed successfully'],
            'failures': []
        }

    def test_interaction_scenario(self, scenario):
        """Test human-robot interaction in simulation."""
        # Set up HRI scenario
        # Monitor safety and comfort metrics
        # Evaluate interaction quality

        return {
            'success': True,  # Simplified
            'metrics': {
                'safety_compliance': 1.0,
                'user_comfort': 0.8,
                'interaction_success_rate': 0.9
            },
            'logs': ['Interaction test completed successfully'],
            'failures': []
        }

    def cleanup_test_environment(self):
        """Clean up the simulation environment after test."""
        # Reset simulation state
        # Clear any temporary objects
        pass

    def generate_test_report(self):
        """Generate a comprehensive test report."""
        report = {
            'timestamp': rclpy.clock.Clock().now().seconds_nanoseconds(),
            'total_tests': len(self.test_results),
            'successful_tests': sum(1 for r in self.test_results.values() if r['success']),
            'results': self.test_results,
            'summary': self.calculate_test_summary()
        }

        return report

    def calculate_test_summary(self):
        """Calculate overall test summary statistics."""
        if not self.test_results:
            return {}

        total = len(self.test_results)
        successful = sum(1 for r in self.test_results.values() if r['success'])

        return {
            'success_rate': successful / total,
            'total_tests': total,
            'passed_tests': successful,
            'failed_tests': total - successful
        }

def run_simulation_tests():
    """Run all simulation tests."""
    rclpy.init()

    test_framework = SimulationTestFramework()

    # Define test scenarios
    test_scenarios = [
        {
            'name': 'basic_navigation',
            'type': 'navigation',
            'goal_pose': [1.0, 1.0, 0.0]
        },
        {
            'name': 'object_manipulation',
            'type': 'manipulation',
            'object_pose': [0.5, 0.5, 0.1],
            'target_pose': [0.8, 0.8, 0.1]
        },
        {
            'name': 'safe_interaction',
            'type': 'interaction',
            'human_pose': [1.0, 0.0, 0.0],
            'interaction_distance': 0.8
        }
    ]

    # Run tests
    for scenario in test_scenarios:
        result = test_framework.run_simulation_test(scenario['name'], scenario)
        print(f"Test {scenario['name']}: {'PASSED' if result['success'] else 'FAILED'}")

    # Generate report
    report = test_framework.generate_test_report()
    print(f"\nTest Summary: {report['summary']['success_rate']:.1%} success rate")

    rclpy.shutdown()
    return report

if __name__ == '__main__':
    report = run_simulation_tests()
    print("Simulation tests completed!")
```

## Performance and Profiling Tools

### 1. System Performance Monitoring

```python title="performance_monitor.py"
import psutil
import time
import threading
from collections import deque
import json

class PerformanceMonitor:
    def __init__(self, update_interval=1.0):
        self.update_interval = update_interval
        self.monitoring = False
        self.performance_data = {
            'cpu_percent': deque(maxlen=100),
            'memory_percent': deque(maxlen=100),
            'disk_io': deque(maxlen=100),
            'network_io': deque(maxlen=100),
            'process_counts': deque(maxlen=100)
        }
        self.monitoring_thread = None

    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring = True
        self.monitoring_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()

    def monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            self.collect_performance_data()
            time.sleep(self.update_interval)

    def collect_performance_data(self):
        """Collect system performance data."""
        timestamp = time.time()

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.performance_data['cpu_percent'].append({
            'timestamp': timestamp,
            'value': cpu_percent
        })

        # Memory usage
        memory_info = psutil.virtual_memory()
        self.performance_data['memory_percent'].append({
            'timestamp': timestamp,
            'value': memory_info.percent
        })

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            self.performance_data['disk_io'].append({
                'timestamp': timestamp,
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes
            })

        # Network I/O
        net_io = psutil.net_io_counters()
        if net_io:
            self.performance_data['network_io'].append({
                'timestamp': timestamp,
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv
            })

        # Process count
        process_count = len(psutil.pids())
        self.performance_data['process_counts'].append({
            'timestamp': timestamp,
            'value': process_count
        })

    def get_current_performance(self):
        """Get current performance metrics."""
        current_data = {}
        for key, data_queue in self.performance_data.items():
            if data_queue:
                current_data[key] = data_queue[-1]  # Latest value
            else:
                current_data[key] = None

        return current_data

    def get_historical_performance(self, metric, duration_minutes=5):
        """Get historical performance data for a specific metric."""
        if metric not in self.performance_data:
            return []

        # Filter data for the requested duration
        cutoff_time = time.time() - (duration_minutes * 60)
        filtered_data = [
            entry for entry in self.performance_data[metric]
            if entry['timestamp'] >= cutoff_time
        ]

        return filtered_data

    def check_performance_alerts(self):
        """Check for performance alerts."""
        alerts = []

        # Check CPU usage
        if self.performance_data['cpu_percent']:
            current_cpu = self.performance_data['cpu_percent'][-1]['value']
            if current_cpu > 90:
                alerts.append({
                    'type': 'high_cpu',
                    'severity': 'warning',
                    'message': f'CPU usage is high: {current_cpu}%'
                })

        # Check memory usage
        if self.performance_data['memory_percent']:
            current_memory = self.performance_data['memory_percent'][-1]['value']
            if current_memory > 90:
                alerts.append({
                    'type': 'high_memory',
                    'severity': 'warning',
                    'message': f'Memory usage is high: {current_memory}%'
                })

        return alerts

    def generate_performance_report(self):
        """Generate a comprehensive performance report."""
        current_perf = self.get_current_performance()
        alerts = self.check_performance_alerts()

        report = {
            'timestamp': time.time(),
            'current_performance': current_perf,
            'alerts': alerts,
            'recommendations': self.generate_recommendations(alerts)
        }

        return report

    def generate_recommendations(self, alerts):
        """Generate recommendations based on alerts."""
        recommendations = []

        for alert in alerts:
            if alert['type'] == 'high_cpu':
                recommendations.append(
                    "Consider optimizing CPU-intensive processes or adding more computational resources."
                )
            elif alert['type'] == 'high_memory':
                recommendations.append(
                    "Check for memory leaks or consider increasing available RAM."
                )

        return recommendations

    def save_performance_data(self, filename):
        """Save performance data to file."""
        data_to_save = {}
        for key, data_queue in self.performance_data.items():
            data_to_save[key] = list(data_queue)

        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=2)

def main():
    """Main function to demonstrate performance monitoring."""
    monitor = PerformanceMonitor(update_interval=0.5)
    monitor.start_monitoring()

    print("Performance monitoring started. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(5)  # Print status every 5 seconds
            perf_data = monitor.get_current_performance()
            alerts = monitor.check_performance_alerts()

            print(f"CPU: {perf_data['cpu_percent']['value'] if perf_data['cpu_percent'] else 'N/A'}%, "
                  f"Memory: {perf_data['memory_percent']['value'] if perf_data['memory_percent'] else 'N/A'}%")

            if alerts:
                for alert in alerts:
                    print(f"⚠️  ALERT: {alert['message']}")

    except KeyboardInterrupt:
        monitor.stop_monitoring()
        print("\nPerformance monitoring stopped.")

        # Generate final report
        report = monitor.generate_performance_report()
        print(f"Performance report generated with {len(report['alerts'])} alerts.")

if __name__ == "__main__":
    main()
```

## Exercises

1. Set up a complete development environment with ROS 2, Gazebo, and Isaac Sim
2. Create a comprehensive test suite for a humanoid robot controller
3. Implement a performance monitoring system for real-time robot applications

## Summary

The tools and technologies for physical AI and humanoid robotics form a comprehensive ecosystem that enables the development, simulation, testing, and deployment of sophisticated robotic systems. From the foundational ROS 2 middleware to advanced simulation environments like Gazebo and Isaac Sim, and from development tools to testing frameworks, mastering these tools is essential for success in humanoid robotics. The key is to understand how these tools work together in a cohesive development workflow and to maintain proper configuration and version control practices for collaborative development.