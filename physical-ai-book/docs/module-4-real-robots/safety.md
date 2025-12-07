---
sidebar_position: 4
---

# Safety Protocols for Real Humanoid Robots

## Learning Objectives
- Understand comprehensive safety protocols for humanoid robot operation
- Implement multi-layered safety systems
- Learn emergency procedures and risk mitigation strategies
- Master safety validation and compliance procedures

## Intuition

Safety protocols for humanoid robots are like the safety systems in a nuclear power plant - multiple redundant systems that prevent catastrophic failures. Just as nuclear facilities have multiple backup systems and strict protocols, humanoid robots operating near humans need comprehensive safety measures to prevent injury to people and damage to property. The stakes are high because humanoid robots are powerful, complex machines that operate in human environments.

## Concept

Humanoid robot safety involves multiple layers of protection:
- **Physical Safety**: Preventing harm to humans and property
- **Operational Safety**: Ensuring safe robot operation
- **Cyber Safety**: Protecting against malicious control
- **Environmental Safety**: Managing robot impact on surroundings

## Multi-Layered Safety Architecture

### 1. Hardware Safety Layer
```python title="hardware_safety.py"
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, Temperature
from std_msgs.msg import Bool, Float64MultiArray
from geometry_msgs.msg import Vector3
import numpy as np

class HardwareSafetyLayer(Node):
    def __init__(self):
        super().__init__('hardware_safety_layer')

        # Safety limits
        self.safety_limits = {
            'joint_position': {'min': -3.14, 'max': 3.14},
            'joint_velocity': {'max': 5.0},
            'joint_effort': {'max': 200.0},
            'temperature': {'max': 70.0},
            'acceleration': {'max': 10.0},
            'power_consumption': {'max': 1000.0}  # watts
        }

        # Emergency stop state
        self.emergency_stop_active = False
        self.last_safe_time = self.get_clock().now()
        self.safety_violations = []

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.temp_sub = self.create_subscription(
            Temperature, '/temperature', self.temp_callback, 10)

        # Publishers
        self.emergency_stop_pub = self.create_publisher(Bool, '/emergency_stop', 10)
        self.velocity_limit_pub = self.create_publisher(Float64MultiArray, '/velocity_limits', 10)

        # Timer for safety monitoring (1000Hz for critical safety)
        self.safety_timer = self.create_timer(0.001, self.hardware_safety_check)

        self.get_logger().info('Hardware Safety Layer initialized')

    def joint_callback(self, msg):
        """Monitor joint states for safety violations"""
        for i, (pos, vel, effort) in enumerate(zip(msg.position, msg.velocity, msg.effort)):
            # Check position limits
            if (pos < self.safety_limits['joint_position']['min'] or
                pos > self.safety_limits['joint_position']['max']):
                self.safety_violations.append(f'Joint {i} position limit violated: {pos}')
                self.trigger_emergency_stop()

            # Check velocity limits
            if abs(vel) > self.safety_limits['joint_velocity']['max']:
                self.safety_violations.append(f'Joint {i} velocity limit violated: {vel}')
                self.trigger_emergency_stop()

            # Check effort limits
            if abs(effort) > self.safety_limits['joint_effort']['max']:
                self.safety_violations.append(f'Joint {i} effort limit violated: {effort}')
                self.trigger_emergency_stop()

    def imu_callback(self, msg):
        """Monitor IMU for balance and acceleration safety"""
        # Check linear acceleration limits
        lin_accel = np.sqrt(msg.linear_acceleration.x**2 +
                           msg.linear_acceleration.y**2 +
                           msg.linear_acceleration.z**2)

        if lin_accel > self.safety_limits['acceleration']['max']:
            self.safety_violations.append(f'Linear acceleration limit violated: {lin_accel}')
            self.trigger_emergency_stop()

        # Check angular velocity limits
        ang_vel = np.sqrt(msg.angular_velocity.x**2 +
                         msg.angular_velocity.y**2 +
                         msg.angular_velocity.z**2)

        if ang_vel > 5.0:  # 5 rad/s angular velocity limit
            self.safety_violations.append(f'Angular velocity limit violated: {ang_vel}')
            self.trigger_emergency_stop()

    def temp_callback(self, msg):
        """Monitor temperature for overheating"""
        if msg.temperature > self.safety_limits['temperature']['max']:
            self.safety_violations.append(f'Temperature limit violated: {msg.temperature}°C')
            self.trigger_emergency_stop()

    def hardware_safety_check(self):
        """Critical safety check running at 1000Hz"""
        if self.emergency_stop_active:
            # Ensure emergency stop is active
            self.publish_emergency_stop()
            return

        # Check for sensor timeouts
        current_time = self.get_clock().now()
        time_diff = (current_time.nanoseconds - self.last_safe_time.nanoseconds) / 1e9

        if time_diff > 0.1:  # 100ms timeout
            self.safety_violations.append(f'Sensor timeout: {time_diff}s')
            self.trigger_emergency_stop()

        # Update last safe time if no violations
        if not self.safety_violations:
            self.last_safe_time = current_time

    def trigger_emergency_stop(self):
        """Trigger emergency stop with hardware priority"""
        self.emergency_stop_active = True
        self.publish_emergency_stop()

        # Log safety violation
        for violation in self.safety_violations:
            self.get_logger().error(f'HARDWARE SAFETY VIOLATION: {violation}')

        self.safety_violations.clear()

    def publish_emergency_stop(self):
        """Publish emergency stop command"""
        stop_msg = Bool()
        stop_msg.data = True
        self.emergency_stop_pub.publish(stop_msg)

    def reset_safety(self):
        """Reset safety system (with proper verification)"""
        # This should only be called after confirming robot is safe
        if self.emergency_stop_active:
            # Verify robot is in safe state before reset
            current_state = self.get_robot_state()
            if self.is_robot_safe(current_state):
                self.emergency_stop_active = False
                self.get_logger().info('Safety system reset - robot is in safe state')
                return True
        return False

    def is_robot_safe(self, state):
        """Check if robot is in a safe state for reset"""
        # Verify all joints are in safe positions
        # Verify no excessive forces
        # Verify robot is not moving
        return True  # Placeholder - implement actual checks
```

### 2. Operational Safety Layer
```python title="operational_safety.py"
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, LaserScan, PointCloud2
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String
from visualization_msgs.msg import Marker
import numpy as np
from scipy.spatial import distance

class OperationalSafetyLayer(Node):
    def __init__(self):
        super().__init__('operational_safety')

        # Operational safety parameters
        self.safety_zones = {
            'human_free_zone': 2.0,    # 2m radius around humans
            'robot_free_zone': 0.5,    # 0.5m around obstacles
            'collision_threshold': 0.3 # 0.3m for collision detection
        }

        # Human detection and tracking
        self.humans = {}  # Dictionary of tracked humans
        self.robot_pose = None
        self.robot_velocity = None

        # Subscribers
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, '/robot_pose', self.pose_callback, 10)
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self.command_callback, 10)

        # Publishers
        self.safety_cmd_pub = self.create_publisher(Twist, '/safety_cmd_vel', 10)
        self.safety_viz_pub = self.create_publisher(Marker, '/safety_viz', 10)

        # Timer for operational safety checks
        self.op_timer = self.create_timer(0.1, self.operational_safety_check)

        self.get_logger().info('Operational Safety Layer initialized')

    def laser_callback(self, msg):
        """Process laser scan for obstacle detection"""
        # Detect humans and obstacles in scan
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

        # Filter valid ranges
        valid_ranges = ranges[(ranges > msg.range_min) & (ranges < msg.range_max)]

        # Detect clusters that might be humans or obstacles
        self.detect_human_clusters(ranges, angles, msg.angle_increment)

    def detect_human_clusters(self, ranges, angles, angle_inc):
        """Detect human-shaped clusters in laser scan"""
        # Convert to Cartesian coordinates
        x_coords = ranges * np.cos(angles)
        y_coords = ranges * np.sin(angles)

        # Simple clustering to find potential humans
        clusters = self.cluster_points(x_coords, y_coords)

        # Update human tracking
        for i, cluster in enumerate(clusters):
            if self.is_human_shape(cluster):
                # Update or create human tracker
                human_id = f'human_{i}'
                self.humans[human_id] = {
                    'position': np.mean(cluster, axis=0),
                    'last_seen': self.get_clock().now()
                }

    def cluster_points(self, x_coords, y_coords):
        """Simple clustering of points"""
        points = np.column_stack((x_coords, y_coords))
        clusters = []
        processed = set()

        for i, point in enumerate(points):
            if i in processed:
                continue

            # Find nearby points
            cluster = [point]
            for j, other_point in enumerate(points[i+1:], i+1):
                if distance.euclidean(point, other_point) < 0.5:  # 0.5m threshold
                    cluster.append(other_point)
                    processed.add(j)

            if len(cluster) > 3:  # Minimum cluster size
                clusters.append(np.array(cluster))

        return clusters

    def is_human_shape(self, cluster):
        """Check if cluster resembles human shape"""
        if len(cluster) < 5:
            return False

        # Calculate cluster dimensions
        x_range = np.max(cluster[:, 0]) - np.min(cluster[:, 0])
        y_range = np.max(cluster[:, 1]) - np.min(cluster[:, 1])

        # Human-like aspect ratio (taller than wide)
        aspect_ratio = max(x_range, y_range) / min(x_range, y_range)

        return 0.3 < aspect_ratio < 3.0 and 0.3 < max(x_range, y_range) < 1.0

    def pose_callback(self, msg):
        """Update robot pose"""
        self.robot_pose = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

    def command_callback(self, msg):
        """Monitor robot commands for safety"""
        self.robot_velocity = np.array([msg.linear.x, msg.linear.y, msg.linear.z])

    def operational_safety_check(self):
        """Check operational safety conditions"""
        if not self.robot_pose is not None:
            return

        # Check for humans in safety zones
        for human_id, human_data in self.humans.items():
            human_pos = human_data['position']
            dist_to_human = distance.euclidean(self.robot_pose[:2], human_pos)

            if dist_to_human < self.safety_zones['human_free_zone']:
                self.get_logger().warn(f'Human too close: {dist_to_human:.2f}m')

                # Generate safe command to move away
                safe_cmd = self.generate_safe_command(human_pos)
                self.safety_cmd_pub.publish(safe_cmd)

    def generate_safe_command(self, human_pos):
        """Generate command to move away from human"""
        cmd = Twist()

        # Calculate direction away from human
        robot_to_human = human_pos - self.robot_pose[:2]
        direction_away = -robot_to_human / np.linalg.norm(robot_to_human)

        # Generate command to move away
        cmd.linear.x = direction_away[0] * 0.1  # 0.1 m/s away
        cmd.linear.y = direction_away[1] * 0.1

        # Stop rotation to avoid unpredictable movement
        cmd.angular.z = 0.0

        return cmd
```

### 3. Cyber Safety Layer
```python title="cyber_safety.py"
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
from sensor_msgs.msg import JointState
import hashlib
import hmac
import time
from threading import Lock

class CyberSafetyLayer(Node):
    def __init__(self):
        super().__init__('cyber_safety')

        # Security parameters
        self.security_key = self.generate_security_key()
        self.command_history = []
        self.max_command_history = 100
        self.command_timeout = 5.0  # seconds
        self.security_lock = Lock()

        # Security monitoring
        self.expected_topics = [
            '/joint_states',
            '/cmd_vel',
            '/imu/data',
            '/scan'
        ]
        self.received_messages = {}
        self.security_violations = []

        # Subscribers for monitoring
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.secure_joint_callback, 10)
        self.cmd_sub = self.create_subscription(
            String, '/insecure_cmd', self.insecure_command_callback, 10)

        # Timer for security checks
        self.security_timer = self.create_timer(1.0, self.security_audit)

        self.get_logger().info('Cyber Safety Layer initialized')

    def generate_security_key(self):
        """Generate secure key for message authentication"""
        import secrets
        return secrets.token_bytes(32)  # 256-bit key

    def authenticate_message(self, message, signature):
        """Authenticate message using HMAC"""
        expected_sig = hmac.new(
            self.security_key,
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(expected_sig, signature)

    def secure_joint_callback(self, msg):
        """Secure callback with authentication"""
        # Verify message integrity
        if self.is_message_integrity_valid(msg):
            # Process message
            self.process_joint_state(msg)
        else:
            self.security_violations.append("Joint state message integrity failed")

    def is_message_integrity_valid(self, msg):
        """Check if message has not been tampered with"""
        # In practice, this would verify digital signatures
        # For simulation, return True
        return True

    def insecure_command_callback(self, msg):
        """Handle potentially insecure commands"""
        # Validate command before execution
        if self.validate_command(msg.data):
            # Add to secure command queue
            self.add_secure_command(msg.data)
        else:
            self.security_violations.append(f"Invalid command: {msg.data}")

    def validate_command(self, command_str):
        """Validate command format and content"""
        # Check for dangerous commands
        dangerous_patterns = [
            'rm -rf',
            'shutdown',
            'reboot',
            'emergency_stop',
            'kill'
        ]

        # Check for dangerous patterns
        for pattern in dangerous_patterns:
            if pattern in command_str.lower():
                return False

        # Check command length
        if len(command_str) > 1000:
            return False

        return True

    def add_secure_command(self, command):
        """Add command to secure execution queue"""
        with self.security_lock:
            cmd_entry = {
                'command': command,
                'timestamp': time.time(),
                'validated': True
            }

            self.command_history.append(cmd_entry)

            # Maintain history size
            if len(self.command_history) > self.max_command_history:
                self.command_history.pop(0)

    def security_audit(self):
        """Perform security audit"""
        # Check for unusual message patterns
        self.check_message_frequency()

        # Check for unauthorized topics
        self.check_authorized_topics()

        # Review security violations
        if self.security_violations:
            for violation in self.security_violations:
                self.get_logger().error(f'SECURITY VIOLATION: {violation}')
            self.security_violations.clear()

    def check_message_frequency(self):
        """Check for unusual message frequency (potential attack)"""
        # This would monitor message rates for anomalies
        pass

    def check_authorized_topics(self):
        """Check that only authorized topics are publishing"""
        # This would verify topic publishers are authorized
        pass
```

## Emergency Procedures

### 1. Emergency Stop System
```python title="emergency_stop_system.py"
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
import time
import threading

class EmergencyStopSystem(Node):
    def __init__(self):
        super().__init__('emergency_stop_system')

        # Emergency stop states
        self.emergency_stop_active = False
        self.software_stop = False
        self.hardware_stop = False
        self.manual_stop = False

        # Stop triggers
        self.stop_reasons = []
        self.emergency_stop_sub = self.create_subscription(
            Bool, '/emergency_stop', self.emergency_stop_callback, 10)
        self.manual_stop_sub = self.create_subscription(
            Bool, '/manual_emergency_stop', self.manual_stop_callback, 10)

        # Publishers for stopping all systems
        self.stop_publishers = {}
        self.setup_stop_publishers()

        # Timer for continuous monitoring
        self.emergency_timer = self.create_timer(0.01, self.emergency_monitor)

        # Hardware emergency stop interface
        self.hardware_interface = HardwareEmergencyInterface()

        self.get_logger().info('Emergency Stop System initialized')

    def setup_stop_publishers(self):
        """Setup publishers for all systems that need stopping"""
        stop_topics = [
            '/cmd_vel',
            '/joint_group_position_controller/commands',
            '/joint_group_velocity_controller/commands',
            '/joint_group_effort_controller/commands',
            '/navigation/cmd_vel',
            '/base_controller/cmd_vel'
        ]

        for topic in stop_topics:
            self.stop_publishers[topic] = self.create_publisher(
                Twist if 'cmd_vel' in topic else Bool, topic, 10
            )

    def emergency_stop_callback(self, msg):
        """Handle emergency stop from software"""
        if msg.data and not self.emergency_stop_active:
            self.software_stop = True
            self.trigger_emergency_stop("Software emergency stop")

    def manual_stop_callback(self, msg):
        """Handle manual emergency stop"""
        if msg.data and not self.emergency_stop_active:
            self.manual_stop = True
            self.trigger_emergency_stop("Manual emergency stop")

    def trigger_emergency_stop(self, reason):
        """Trigger emergency stop procedure"""
        self.emergency_stop_active = True
        self.stop_reasons.append(reason)

        self.get_logger().error(f'EMERGENCY STOP TRIGGERED: {reason}')

        # Stop all motion commands
        self.stop_all_motion()

        # Activate hardware emergency stop
        self.hardware_interface.activate_emergency_stop()

        # Log emergency event
        self.log_emergency_event(reason)

    def stop_all_motion(self):
        """Send stop commands to all motion systems"""
        # Send zero velocity commands
        stop_cmd = Twist()

        for topic, publisher in self.stop_publishers.items():
            try:
                if 'cmd_vel' in topic:
                    publisher.publish(stop_cmd)
                else:
                    # For joint controllers, send zero position/effort
                    zero_msg = Bool()
                    zero_msg.data = False
                    publisher.publish(zero_msg)
            except Exception as e:
                self.get_logger().error(f'Failed to stop {topic}: {e}')

    def emergency_monitor(self):
        """Continuous monitoring for emergency conditions"""
        if not self.emergency_stop_active:
            # Check for dangerous conditions
            if self.detect_dangerous_condition():
                self.trigger_emergency_stop("Dangerous condition detected")

    def detect_dangerous_condition(self):
        """Detect dangerous operational conditions"""
        # Check for excessive forces, velocities, temperatures
        # Check for human proximity
        # Check for system failures
        return False  # Placeholder

    def log_emergency_event(self, reason):
        """Log emergency event with timestamp"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] EMERGENCY STOP: {reason}"

        # Write to emergency log file
        with open('/var/log/robot_emergency.log', 'a') as log_file:
            log_file.write(log_entry + '\n')

    def reset_emergency_stop(self):
        """Reset emergency stop (only after safety verification)"""
        if self.emergency_stop_active:
            # Verify it's safe to reset
            if self.is_safe_to_reset():
                self.emergency_stop_active = False
                self.software_stop = False
                self.manual_stop = False
                self.hardware_stop = False
                self.stop_reasons.clear()

                # Reset hardware emergency stop
                self.hardware_interface.deactivate_emergency_stop()

                self.get_logger().info('Emergency stop reset - system cleared')
                return True

        return False

    def is_safe_to_reset(self):
        """Check if it's safe to reset emergency stop"""
        # Verify robot is stationary
        # Verify no humans in danger zone
        # Verify no ongoing safety violations
        return True  # Placeholder

class HardwareEmergencyInterface:
    """Interface to hardware emergency stop system"""
    def __init__(self):
        # Initialize hardware interface
        self.hardware_connected = True

    def activate_emergency_stop(self):
        """Activate hardware emergency stop"""
        if self.hardware_connected:
            # Send signal to hardware emergency stop circuit
            print("Hardware emergency stop activated")
            # This would interface with physical emergency stop hardware

    def deactivate_emergency_stop(self):
        """Deactivate hardware emergency stop"""
        if self.hardware_connected:
            # Send signal to release hardware emergency stop
            print("Hardware emergency stop deactivated")
```

### 2. Risk Assessment and Mitigation
```python title="risk_assessment.py"
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List

class RiskLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class RiskType(Enum):
    COLLISION = "collision"
    STABILITY = "stability"
    ELECTRICAL = "electrical"
    THERMAL = "thermal"
    MECHANICAL = "mechanical"
    CYBER = "cyber"

@dataclass
class RiskAssessment:
    risk_type: RiskType
    risk_level: RiskLevel
    probability: float  # 0.0 to 1.0
    severity: float     # 0.0 to 1.0
    mitigation_score: float  # 0.0 to 1.0 (1.0 = fully mitigated)
    description: str

class RiskAssessmentSystem:
    def __init__(self):
        self.risks = []
        self.mitigation_strategies = {}
        self.risk_thresholds = {
            RiskLevel.LOW: 0.2,
            RiskLevel.MEDIUM: 0.4,
            RiskLevel.HIGH: 0.7,
            RiskLevel.CRITICAL: 0.9
        }

    def assess_current_risks(self, robot_state, environment_state):
        """Assess current risks based on robot and environment state"""
        self.risks = []

        # Assess collision risk
        collision_risk = self.assess_collision_risk(robot_state, environment_state)
        self.risks.append(collision_risk)

        # Assess stability risk
        stability_risk = self.assess_stability_risk(robot_state)
        self.risks.append(stability_risk)

        # Assess thermal risk
        thermal_risk = self.assess_thermal_risk(robot_state)
        self.risks.append(thermal_risk)

        # Assess electrical risk
        electrical_risk = self.assess_electrical_risk(robot_state)
        self.risks.append(electrical_risk)

        return self.risks

    def assess_collision_risk(self, robot_state, environment_state):
        """Assess collision risk"""
        # Calculate distance to nearest obstacles/humans
        min_distance = float('inf')

        if 'obstacles' in environment_state:
            for obstacle in environment_state['obstacles']:
                dist = self.calculate_distance(robot_state['position'], obstacle['position'])
                min_distance = min(min_distance, dist)

        if 'humans' in environment_state:
            for human in environment_state['humans']:
                dist = self.calculate_distance(robot_state['position'], human['position'])
                min_distance = min(min_distance, dist)

        # Calculate collision probability based on distance and velocity
        velocity_magnitude = np.linalg.norm(robot_state['velocity'])
        probability = max(0.0, 1.0 - min_distance / (velocity_magnitude + 0.1))

        # Severity increases with robot size and speed
        severity = min(1.0, velocity_magnitude / 2.0)

        # Mitigation: proximity sensors and collision avoidance
        mitigation_score = 0.8  # Good mitigation with sensors

        return RiskAssessment(
            risk_type=RiskType.COLLISION,
            risk_level=self.calculate_risk_level(probability, severity, mitigation_score),
            probability=probability,
            severity=severity,
            mitigation_score=mitigation_score,
            description=f"Collision risk: {probability:.2f} probability, {severity:.2f} severity"
        )

    def assess_stability_risk(self, robot_state):
        """Assess stability risk"""
        # Calculate center of mass position relative to support polygon
        com_position = robot_state['center_of_mass']
        support_polygon = robot_state['support_polygon']

        # Check if CoM is within support polygon
        if self.is_point_in_polygon(com_position, support_polygon):
            stability_factor = 0.1
        else:
            # Calculate distance outside support polygon
            distance_outside = self.distance_to_polygon(com_position, support_polygon)
            stability_factor = min(1.0, distance_outside * 5.0)  # Scale factor

        # Consider angular velocity for dynamic stability
        angular_velocity = np.linalg.norm(robot_state['angular_velocity'])
        probability = min(1.0, stability_factor + angular_velocity * 0.1)

        # Severity of fall increases with height and mass
        severity = min(1.0, robot_state['height'] * 0.5 + robot_state['mass'] * 0.01)

        # Mitigation: balance control and fall prevention
        mitigation_score = 0.7

        return RiskAssessment(
            risk_type=RiskType.STABILITY,
            risk_level=self.calculate_risk_level(probability, severity, mitigation_score),
            probability=probability,
            severity=severity,
            mitigation_score=mitigation_score,
            description=f"Stability risk: {probability:.2f} probability, {severity:.2f} severity"
        )

    def assess_thermal_risk(self, robot_state):
        """Assess thermal risk from overheating"""
        max_temp = max(robot_state['joint_temperatures']) if robot_state['joint_temperatures'] else 25.0

        # Calculate probability of thermal damage
        if max_temp > 80:  # Critical temperature
            probability = 0.9
        elif max_temp > 60:  # Warning temperature
            probability = 0.6
        else:
            probability = 0.1

        # Severity of thermal damage
        severity = min(1.0, (max_temp - 20) / 100.0)

        # Mitigation: cooling systems and thermal monitoring
        mitigation_score = 0.8

        return RiskAssessment(
            risk_type=RiskType.THERMAL,
            risk_level=self.calculate_risk_level(probability, severity, mitigation_score),
            probability=probability,
            severity=severity,
            mitigation_score=mitigation_score,
            description=f"Thermal risk: {probability:.2f} probability, {severity:.2f} severity"
        )

    def assess_electrical_risk(self, robot_state):
        """Assess electrical risk"""
        # Check for overcurrent, short circuits, etc.
        max_current = max(robot_state['motor_currents']) if robot_state['motor_currents'] else 0.0

        if max_current > 50.0:  # Overcurrent threshold
            probability = 0.9
        elif max_current > 30.0:  # Warning threshold
            probability = 0.6
        else:
            probability = 0.1

        # Severity of electrical failure
        severity = min(1.0, max_current / 100.0)

        # Mitigation: current limiting and circuit protection
        mitigation_score = 0.9

        return RiskAssessment(
            risk_type=RiskType.ELECTRICAL,
            risk_level=self.calculate_risk_level(probability, severity, mitigation_score),
            probability=probability,
            severity=severity,
            mitigation_score=mitigation_score,
            description=f"Electrical risk: {probability:.2f} probability, {severity:.2f} severity"
        )

    def calculate_risk_level(self, probability, severity, mitigation):
        """Calculate overall risk level"""
        # Risk = Probability * Severity * (1 - Mitigation)
        raw_risk = probability * severity * (1 - mitigation)

        if raw_risk >= self.risk_thresholds[RiskLevel.CRITICAL]:
            return RiskLevel.CRITICAL
        elif raw_risk >= self.risk_thresholds[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        elif raw_risk >= self.risk_thresholds[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two points"""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def is_point_in_polygon(self, point, polygon):
        """Check if point is inside polygon using ray casting"""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def distance_to_polygon(self, point, polygon):
        """Calculate minimum distance from point to polygon"""
        min_dist = float('inf')
        px, py = point

        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]

            # Calculate distance to line segment
            dist = self.distance_point_to_segment(px, py, p1[0], p1[1], p2[0], p2[1])
            min_dist = min(min_dist, dist)

        return min_dist

    def distance_point_to_segment(self, px, py, x1, y1, x2, y2):
        """Calculate distance from point to line segment"""
        A = px - x1
        B = py - y1
        C = x2 - x1
        D = y2 - y1

        dot = A * C + B * D
        len_sq = C * C + D * D

        if len_sq == 0:
            return np.sqrt(A * A + B * B)

        param = dot / len_sq

        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx = x1 + param * C
            yy = y1 + param * D

        dx = px - xx
        dy = py - yy
        return np.sqrt(dx * dx + dy * dy)

    def get_highest_risk(self):
        """Get the highest risk in the current assessment"""
        if not self.risks:
            return None

        return max(self.risks, key=lambda r: r.probability * r.severity * (1 - r.mitigation_score))

    def recommend_actions(self):
        """Recommend safety actions based on current risks"""
        actions = []

        for risk in self.risks:
            if risk.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                if risk.risk_type == RiskType.COLLISION:
                    actions.append("Reduce speed and increase safety margins")
                elif risk.risk_type == RiskType.STABILITY:
                    actions.append("Move to stable position and reduce motion")
                elif risk.risk_type == RiskType.THERMAL:
                    actions.append("Reduce activity and check cooling systems")
                elif risk.risk_type == RiskType.ELECTRICAL:
                    actions.append("Check power systems and reduce load")

        return actions
```

## Safety Validation and Compliance

### 1. Safety Testing Framework
```python title="safety_testing.py"
import unittest
import numpy as np
from typing import Dict, Any
import time

class SafetyTestCase(unittest.TestCase):
    """Base class for safety test cases"""

    def setUp(self):
        """Setup for safety tests"""
        self.safety_system = SafetySystemMock()
        self.test_results = []

    def test_emergency_stop_functionality(self):
        """Test that emergency stop works correctly"""
        # Trigger emergency stop
        self.safety_system.trigger_emergency_stop()

        # Verify all motion stops
        self.assertTrue(self.safety_system.emergency_stop_active)
        self.assertEqual(self.safety_system.get_robot_velocity(), 0.0)

        # Verify safety state
        self.assertEqual(self.safety_system.get_safety_state(), 'EMERGENCY_STOPPED')

    def test_collision_avoidance(self):
        """Test collision avoidance system"""
        # Set up scenario with obstacle
        self.safety_system.set_obstacle_distance(0.2)  # 20cm obstacle

        # Verify collision avoidance activates
        safe_command = self.safety_system.get_safe_command()
        self.assertLess(abs(safe_command.linear.x), 0.1)  # Should slow down

    def test_joint_limits(self):
        """Test joint limit enforcement"""
        # Command movement beyond joint limits
        test_command = {'position': [5.0, 0, 0, 0, 0, 0]}  # Beyond limit

        # Verify command is limited
        limited_command = self.safety_system.apply_joint_limits(test_command)
        self.assertLessEqual(limited_command['position'][0], 3.14)  # Within limit

    def test_balance_stability(self):
        """Test balance stability monitoring"""
        # Set up unstable configuration
        self.safety_system.set_com_outside_support(0.3)  # 30cm outside

        # Verify stability system activates
        stability_ok = self.safety_system.check_balance_stability()
        self.assertFalse(stability_ok)

class SafetyValidationFramework:
    def __init__(self):
        self.test_suite = unittest.TestSuite()
        self.results = None

    def add_test(self, test_class):
        """Add a test class to the suite"""
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_class)
        self.test_suite.addTest(suite)

    def run_validation(self):
        """Run all safety validations"""
        runner = unittest.TextTestRunner(verbosity=2)
        self.results = runner.run(self.test_suite)

        return {
            'total_tests': self.results.testsRun,
            'passed': self.results.testsRun - len(self.results.failures) - len(self.results.errors),
            'failed': len(self.results.failures),
            'errors': len(self.results.errors),
            'success_rate': (self.results.testsRun - len(self.results.failures) - len(self.results.errors)) / self.results.testsRun if self.results.testsRun > 0 else 0
        }

    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        if not self.results:
            return "No validation results available"

        report = f"""
        SAFETY VALIDATION REPORT
        ========================
        Total Tests: {self.results.testsRun}
        Passed: {self.results.testsRun - len(self.results.failures) - len(self.results.errors)}
        Failed: {len(self.results.failures)}
        Errors: {len(self.results.errors)}
        Success Rate: {(self.results.testsRun - len(self.results.failures) - len(self.results.errors)) / self.results.testsRun * 100:.1f}%

        Failures:
        """

        for failure in self.results.failures:
            report += f"  - {failure[0]}: {failure[1]}\n"

        report += "\nErrors:\n"
        for error in self.results.errors:
            report += f"  - {error[0]}: {error[1]}\n"

        return report

class SafetySystemMock:
    """Mock safety system for testing"""
    def __init__(self):
        self.emergency_stop_active = False
        self.safety_state = 'NORMAL'
        self.obstacle_distance = float('inf')
        self.com_outside_support = 0.0

    def trigger_emergency_stop(self):
        self.emergency_stop_active = True
        self.safety_state = 'EMERGENCY_STOPPED'

    def get_robot_velocity(self):
        return 0.0 if self.emergency_stop_active else 1.0

    def get_safety_state(self):
        return self.safety_state

    def set_obstacle_distance(self, distance):
        self.obstacle_distance = distance

    def get_safe_command(self):
        from geometry_msgs.msg import Twist
        cmd = Twist()
        if self.obstacle_distance < 0.5:
            cmd.linear.x = min(0.1, self.obstacle_distance * 0.5)  # Slow approach
        else:
            cmd.linear.x = 0.5  # Normal speed
        return cmd

    def apply_joint_limits(self, command):
        # Apply position limits
        limited_command = command.copy()
        for i, pos in enumerate(command['position']):
            limited_command['position'][i] = np.clip(pos, -3.14, 3.14)
        return limited_command

    def set_com_outside_support(self, distance):
        self.com_outside_support = distance

    def check_balance_stability(self):
        return self.com_outside_support < 0.1  # Stable if within 10cm

def run_safety_validation():
    """Run comprehensive safety validation"""
    print("Starting safety validation...")

    framework = SafetyValidationFramework()
    framework.add_test(SafetyTestCase)

    results = framework.run_validation()
    report = framework.generate_validation_report()

    print(report)

    # Check if validation passed
    if results['success_rate'] >= 0.95:  # 95% success rate required
        print("✅ Safety validation PASSED")
        return True
    else:
        print("❌ Safety validation FAILED")
        return False

if __name__ == "__main__":
    run_safety_validation()
```

## Exercises

1. Implement a comprehensive safety system for a specific humanoid robot
2. Design and execute safety validation tests for robot deployment
3. Create an emergency response procedure for different failure scenarios

## Summary

Safety protocols for humanoid robots require a multi-layered approach with hardware, operational, and cyber safety systems. The key components include emergency stop systems, risk assessment procedures, and comprehensive validation frameworks. By implementing these safety measures and maintaining rigorous testing protocols, we can ensure safe operation of humanoid robots in human environments. Remember that safety is not optional but essential for responsible robotics development.