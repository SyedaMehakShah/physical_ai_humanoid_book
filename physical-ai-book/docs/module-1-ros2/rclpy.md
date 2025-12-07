---
sidebar_position: 5
---

# Python ROS Client Library (rclpy)

## Learning Objectives
- Master the Python client library for ROS 2 (rclpy)
- Understand how to structure Python nodes for humanoid robotics
- Learn best practices for Python-based robot development
- Apply rclpy patterns to humanoid robot control systems

## Intuition

rclpy is the Python interface to ROS 2, like having a translator that allows Python programs to communicate with the ROS 2 middleware. It provides all the functionality needed to create nodes, publish/subscribe to topics, provide/use services, and manage parameters - all from Python code. For humanoid robotics, rclpy allows you to leverage Python's strengths in AI and data processing while connecting to the robot's control systems.

## Concept

rclpy is a Python package that provides Python bindings for the ROS 2 client library (rcl). It provides a Python API to interact with the ROS 2 ecosystem, allowing Python developers to create ROS 2 nodes, communicate with other nodes, and access ROS 2 features like parameters, services, and actions.

## Minimal Example

Here's a comprehensive example showing multiple rclpy patterns for humanoid control:

```python title="humanoid_controller.py"
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from std_srvs.srv import SetBool
import time

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # QoS profile for sensor data
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # Publishers
        self.joint_pub = self.create_publisher(JointState, 'joint_commands', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, 'controller_status', 10)

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, sensor_qos)

        # Services
        self.enable_service = self.create_service(
            SetBool, 'enable_controller', self.enable_callback)

        # Timers
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20Hz
        self.status_timer = self.create_timer(1.0, self.publish_status)  # 1Hz

        # Internal state
        self.enabled = False
        self.current_joint_states = JointState()

        self.get_logger().info('Humanoid Controller initialized')

    def joint_state_callback(self, msg):
        """Callback for receiving joint state updates"""
        self.current_joint_states = msg
        self.get_logger().debug(f'Received {len(msg.name)} joints')

    def enable_callback(self, request, response):
        """Service callback to enable/disable controller"""
        self.enabled = request.data
        response.success = True
        response.message = f'Controller {"enabled" if self.enabled else "disabled"}'
        self.get_logger().info(response.message)
        return response

    def control_loop(self):
        """Main control loop running at 20Hz"""
        if not self.enabled:
            return

        # In a real implementation, this would contain control algorithms
        # For now, we'll just publish a simple command
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = ['hip_joint', 'knee_joint', 'ankle_joint']
        joint_msg.position = [0.1, 0.0, -0.1]  # Simple test positions

        self.joint_pub.publish(joint_msg)

    def publish_status(self):
        """Publish controller status at 1Hz"""
        status_msg = String()
        status_msg.data = f'Controller: {"ENABLED" if self.enabled else "DISABLED"}'
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down Humanoid Controller...')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for rclpy

### 1. Node Structure
```python
class BestPracticeNode(Node):
    def __init__(self):
        super().__init__('best_practice_node')

        # Initialize publishers first
        self.pub = self.create_publisher(MsgType, 'topic', 10)

        # Then subscribers
        self.sub = self.create_subscription(MsgType, 'topic', self.cb, 10)

        # Then services/clients
        self.srv = self.create_service(SrvType, 'service', self.srv_cb)

        # Finally timers
        self.timer = self.create_timer(0.1, self.timer_cb)
```

### 2. Error Handling
```python
def safe_publisher(self, msg):
    try:
        self.publisher.publish(msg)
    except Exception as e:
        self.get_logger().error(f'Failed to publish message: {e}')
```

### 3. Parameter Management
```python
def __init__(self):
    super().__init__('param_node')

    # Declare parameters with defaults
    self.declare_parameter('control_rate', 50)  # Hz
    self.declare_parameter('safety_timeout', 1.0)  # seconds

    # Access parameters
    self.rate = self.get_parameter('control_rate').value
    self.timeout = self.get_parameter('safety_timeout').value
```

## Common Patterns in Humanoid Robotics

### 1. State Machine Pattern
```python
from enum import Enum

class RobotState(Enum):
    IDLE = 1
    WALKING = 2
    STANDING = 3
    EMERGENCY = 4

class StateMachineNode(Node):
    def __init__(self):
        super().__init__('state_machine')
        self.current_state = RobotState.IDLE
        # Implementation would include state transition logic
```

### 2. Safety Monitor Pattern
```python
class SafetyMonitor(Node):
    def __init__(self):
        super().__init__('safety_monitor')
        self.emergency_stop_pub = self.create_publisher(Bool, 'emergency_stop', 1)

    def check_safety(self, sensor_data):
        # Check for dangerous conditions
        if self.is_unsafe(sensor_data):
            self.trigger_emergency_stop()
```

## Exercises

1. Create a Python node that implements a PID controller for humanoid joint position
2. Implement a safety monitor that watches joint limits and triggers emergency stops
3. Design a parameter management system for different humanoid robot configurations

## Summary

rclpy provides the Python interface to ROS 2 that's essential for humanoid robotics development. With its intuitive API, Python's AI libraries can be seamlessly integrated with robot control systems, making it ideal for implementing intelligent humanoid behaviors.