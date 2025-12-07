---
sidebar_position: 1
---

# Module 4: From Simulation to Real Humanoids

## Learning Objectives
- Understand the challenges and techniques of sim-to-real transfer
- Learn to identify and address differences between simulation and reality
- Gain experience with real humanoid robot platforms and their interfaces
- Understand safety considerations and protocols for real robot operation
- Learn calibration and system identification techniques for real robots

## Intuition

Moving from simulation to real humanoid robots is like transitioning from flight simulation to actual flight. While simulators provide a safe environment to develop and test your control algorithms, real robots introduce unpredictable factors like sensor noise, actuator delays, mechanical wear, and environmental uncertainties. The goal is to develop robust control systems that can handle these real-world challenges while maintaining the performance achieved in simulation.

## Concept

Sim-to-real transfer involves bridging the "reality gap" between simulation and real-world performance:
- **System Identification**: Understanding real robot dynamics
- **Sensor Calibration**: Aligning simulated and real sensor data
- **Controller Adaptation**: Adjusting control parameters for real hardware
- **Safety Protocols**: Ensuring safe operation during transition

## The Reality Gap

The reality gap consists of several factors that differ between simulation and reality:

### 1. Physical Differences
- **Actuator Dynamics**: Real actuators have delays, backlash, and nonlinearities
- **Sensor Noise**: Real sensors have noise, bias, and drift
- **Mechanical Imperfections**: Joint friction, link flexibility, and manufacturing tolerances
- **Environmental Factors**: Temperature, humidity, and surface conditions

### 2. Temporal Differences
- **Timing**: Real-time constraints and communication delays
- **Synchronization**: Clock drift between different systems
- **Latency**: Sensor processing and actuator response delays

### 3. Modeling Differences
- **Parameter Uncertainty**: Inaccurate mass, inertia, or friction models
- **Unmodeled Dynamics**: Flexible links, gear backlash, motor dynamics
- **Contact Models**: Simplified contact physics in simulation

## System Identification for Real Humanoids

Understanding your real robot's characteristics:

```python title="system_identification.py"
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray
from control_msgs.msg import JointTrajectoryControllerState

class SystemIdentificationNode(Node):
    def __init__(self):
        super().__init__('system_identification')

        # Data collection variables
        self.joint_states = []
        self.commands = []
        self.timestamps = []
        self.identification_active = False

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)

        # Publishers for excitation signals
        self.command_pub = self.create_publisher(
            Float64MultiArray, '/joint_group_position_controller/commands', 10)

        # Timer for identification process
        self.ident_timer = self.create_timer(0.01, self.identification_loop)

        self.get_logger().info('System Identification Node initialized')

    def start_identification(self):
        """Start system identification process"""
        self.identification_active = True
        self.joint_states = []
        self.commands = []
        self.timestamps = []

        # Apply excitation signals to identify system
        self.apply_excitation_signal()

    def apply_excitation_signal(self):
        """Apply known input signals to identify system"""
        # Apply chirp signal (swept sine) to excite all frequencies
        t = np.linspace(0, 10, 1000)  # 10 seconds of excitation
        f0, f1 = 0.1, 5.0  # Frequency range
        chirp_signal = np.sin(2 * np.pi * f0 * t + (f1-f0)/t[-1] * t**2)

        # Store for analysis
        self.excitation_signal = chirp_signal

    def joint_callback(self, msg):
        """Process joint state data"""
        if self.identification_active:
            self.joint_states.append({
                'position': np.array(msg.position),
                'velocity': np.array(msg.velocity),
                'effort': np.array(msg.effort)
            })
            self.timestamps.append(self.get_clock().now().nanoseconds / 1e9)

    def imu_callback(self, msg):
        """Process IMU data for balance identification"""
        # Collect IMU data for balance and orientation analysis
        pass

    def identification_loop(self):
        """Main identification loop"""
        if not self.identification_active:
            return

        # Apply current excitation signal
        if hasattr(self, 'excitation_signal'):
            current_idx = min(len(self.timestamps), len(self.excitation_signal))
            if current_idx < len(self.excitation_signal):
                cmd_msg = Float64MultiArray()
                cmd_msg.data = [self.excitation_signal[current_idx]] * len(self.joint_states[-1]['position']) if self.joint_states else [0.0] * 6
                self.command_pub.publish(cmd_msg)

        # Stop after collection period
        if len(self.timestamps) > 1000:  # Collected enough data
            self.identification_active = False
            self.analyze_system()

    def analyze_system(self):
        """Analyze collected data to identify system parameters"""
        if not self.joint_states:
            return

        # Convert to numpy arrays
        positions = np.array([state['position'] for state in self.joint_states])
        velocities = np.array([state['velocity'] for state in self.joint_states])
        efforts = np.array([state['effort'] for state in self.joint_states])
        times = np.array(self.timestamps)

        # Perform system identification
        for joint_idx in range(positions.shape[1]):
            self.identify_single_joint(joint_idx, times, positions[:, joint_idx], velocities[:, joint_idx], efforts[:, joint_idx])

    def identify_single_joint(self, joint_idx, time, position, velocity, effort):
        """Identify parameters for a single joint"""
        # Estimate transfer function from effort to position
        # This is a simplified example - in practice, use more sophisticated methods

        # Calculate frequency response
        dt = np.mean(np.diff(time))
        freqs, h = signal.welch(effort, fs=1/dt, nperseg=min(len(effort), 256))

        # Fit transfer function model
        # For a simple second-order system: H(s) = K / (s^2 + 2*zeta*wn*s + wn^2)
        # Estimate parameters: K (gain), wn (natural frequency), zeta (damping)

        # Print results
        self.get_logger().info(f'Joint {joint_idx} identified parameters:')
        self.get_logger().info(f'  Natural frequency: ~2-10 rad/s (typical for actuators)')
        self.get_logger().info(f'  Damping ratio: ~0.1-0.7 (depends on control)')

def main(args=None):
    rclpy.init(args=args)
    ident_node = SystemIdentificationNode()

    # Start identification after a delay to allow connections
    import time
    time.sleep(2)
    ident_node.start_identification()

    try:
        rclpy.spin(ident_node)
    except KeyboardInterrupt:
        pass
    finally:
        ident_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sensor Calibration and Alignment

Calibrating sensors to match simulation models:

```python title="sensor_calibration.py"
import numpy as np
from scipy.optimize import minimize
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, Temperature, FluidPressure
from geometry_msgs.msg import Vector3Stamped

class SensorCalibrationNode(Node):
    def __init__(self):
        super().__init__('sensor_calibration')

        # Calibration parameters
        self.joint_bias = np.zeros(6)  # Joint position bias
        self.imu_bias = {'gyro': np.zeros(3), 'accel': np.zeros(3)}  # IMU bias
        self.calibration_complete = False

        # Data storage for calibration
        self.calibration_data = {
            'joint_positions': [],
            'imu_readings': [],
            'reference_positions': []  # Known positions for calibration
        }

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)

        # Timer for calibration process
        self.cal_timer = self.create_timer(0.1, self.calibration_loop)

        self.get_logger().info('Sensor Calibration Node initialized')

    def joint_callback(self, msg):
        """Process joint state data"""
        if not self.calibration_complete:
            self.calibration_data['joint_positions'].append(np.array(msg.position))

    def imu_callback(self, msg):
        """Process IMU data"""
        if not self.calibration_complete:
            imu_reading = np.array([
                msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
            ])
            self.calibration_data['imu_readings'].append(imu_reading)

    def collect_calibration_data(self, reference_positions):
        """Collect calibration data at known positions"""
        self.calibration_data['reference_positions'] = reference_positions

        # Move robot to each reference position and collect data
        # This would typically involve commanding the robot to specific poses
        pass

    def calibrate_joints(self):
        """Calibrate joint position sensors"""
        if len(self.calibration_data['joint_positions']) < len(self.calibration_data['reference_positions']):
            return False

        # Calculate bias between measured and reference positions
        measured = np.array(self.calibration_data['joint_positions'])
        reference = np.array(self.calibration_data['reference_positions'])

        # Calculate bias for each joint
        self.joint_bias = np.mean(measured - reference, axis=0)

        self.get_logger().info(f'Joint calibration bias: {self.joint_bias}')
        return True

    def calibrate_imu(self):
        """Calibrate IMU sensors"""
        # IMU calibration requires the robot to be in known orientations
        # and static positions

        # For gyroscope: average readings when robot is static
        # For accelerometer: readings when robot is level and static
        if len(self.calibration_data['imu_readings']) > 100:
            imu_array = np.array(self.calibration_data['imu_readings'])

            # Gyroscope bias (should read 0 when static)
            gyro_readings = imu_array[:, :3]
            self.imu_bias['gyro'] = np.mean(gyro_readings, axis=0)

            # Accelerometer bias (should read [0, 0, 9.81] when level)
            accel_readings = imu_array[:, 3:]
            expected_gravity = np.array([0, 0, 9.81])
            self.imu_bias['accel'] = expected_gravity - np.mean(accel_readings, axis=0)

            self.get_logger().info(f'IMU calibration - Gyro bias: {self.imu_bias["gyro"]}')
            self.get_logger().info(f'IMU calibration - Accel bias: {self.imu_bias["accel"]}')

    def apply_calibration(self, raw_joint_state, raw_imu_data):
        """Apply calibration to raw sensor data"""
        calibrated_data = {}

        # Apply joint calibration
        calibrated_data['joint_position'] = raw_joint_state.position - self.joint_bias

        # Apply IMU calibration
        calibrated_data['gyro'] = np.array([
            raw_imu_data.angular_velocity.x,
            raw_imu_data.angular_velocity.y,
            raw_imu_data.angular_velocity.z
        ]) - self.imu_bias['gyro']

        calibrated_data['accel'] = np.array([
            raw_imu_data.linear_acceleration.x,
            raw_imu_data.linear_acceleration.y,
            raw_imu_data.linear_acceleration.z
        ]) - self.imu_bias['accel']

        return calibrated_data

    def calibration_loop(self):
        """Main calibration loop"""
        if not self.calibration_complete and len(self.calibration_data['joint_positions']) > 100:
            # Perform calibration
            joint_cal_success = self.calibrate_joints()
            self.calibrate_imu()

            if joint_cal_success:
                self.calibration_complete = True
                self.get_logger().info('Sensor calibration completed successfully')

def main(args=None):
    rclpy.init(args=args)
    cal_node = SensorCalibrationNode()

    try:
        rclpy.spin(cal_node)
    except KeyboardInterrupt:
        pass
    finally:
        cal_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Safety Protocols and Emergency Procedures

Essential safety measures for real humanoid operation:

```python title="safety_system.py"
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, Temperature
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Bool, Float64MultiArray
from builtin_interfaces.msg import Duration
import numpy as np

class SafetySystemNode(Node):
    def __init__(self):
        super().__init__('safety_system')

        # Safety parameters
        self.safety_limits = {
            'joint_position': {'min': -3.14, 'max': 3.14},  # radians
            'joint_velocity': {'max': 5.0},  # rad/s
            'joint_effort': {'max': 100.0},  # Nm
            'imu_orientation': {'max_tilt': 0.5},  # rad from upright
            'temperature': {'max': 60.0}  # degrees Celsius
        }

        # Safety state
        self.emergency_stop = False
        self.safety_violations = []
        self.last_safe_time = self.get_clock().now()

        # Subscribers for monitoring
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.temp_sub = self.create_subscription(
            Temperature, '/temperature', self.temp_callback, 10)

        # Publishers for safety commands
        self.emergency_stop_pub = self.create_publisher(Bool, '/emergency_stop', 10)
        self.velocity_limit_pub = self.create_publisher(Float64MultiArray, '/velocity_limits', 10)

        # Timer for safety monitoring
        self.safety_timer = self.create_timer(0.01, self.safety_check)  # 100Hz

        # Emergency stop subscriber
        self.emergency_stop_sub = self.create_subscription(
            Bool, '/emergency_stop_request', self.emergency_stop_callback, 10)

        self.get_logger().info('Safety System initialized')

    def joint_callback(self, msg):
        """Monitor joint states for safety violations"""
        for i, pos in enumerate(msg.position):
            if (pos < self.safety_limits['joint_position']['min'] or
                pos > self.safety_limits['joint_position']['max']):
                self.safety_violations.append(f'Joint {i} position limit exceeded: {pos}')

        for i, vel in enumerate(msg.velocity):
            if abs(vel) > self.safety_limits['joint_velocity']['max']:
                self.safety_violations.append(f'Joint {i} velocity limit exceeded: {vel}')

        for i, effort in enumerate(msg.effort):
            if abs(effort) > self.safety_limits['joint_effort']['max']:
                self.safety_violations.append(f'Joint {i} effort limit exceeded: {effort}')

    def imu_callback(self, msg):
        """Monitor IMU for balance violations"""
        # Convert quaternion to Euler angles to check tilt
        quat = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        euler = self.quaternion_to_euler(quat)

        # Check if robot is tilting too much
        tilt_angle = np.sqrt(euler[0]**2 + euler[1]**2)  # Roll and pitch
        if tilt_angle > self.safety_limits['imu_orientation']['max_tilt']:
            self.safety_violations.append(f'Balancing limit exceeded: tilt={tilt_angle}')

    def temp_callback(self, msg):
        """Monitor temperature for overheating"""
        if msg.temperature > self.safety_limits['temperature']['max']:
            self.safety_violations.append(f'Temperature limit exceeded: {msg.temperature}°C')

    def quaternion_to_euler(self, q):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        import math
        w, x, y, z = q

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return [roll, pitch, yaw]

    def safety_check(self):
        """Main safety monitoring function"""
        if self.emergency_stop:
            # Publish emergency stop command
            stop_msg = Bool()
            stop_msg.data = True
            self.emergency_stop_pub.publish(stop_msg)
            return

        # Check for safety violations
        if self.safety_violations:
            # Log violations
            for violation in self.safety_violations:
                self.get_logger().error(f'Safety violation: {violation}')

            # Trigger emergency stop
            self.trigger_emergency_stop()
            self.safety_violations.clear()

        # Update safe time if no violations
        if not self.safety_violations:
            self.last_safe_time = self.get_clock().now()

        # Check for timeout (no sensor data)
        current_time = self.get_clock().now()
        if (current_time.nanoseconds - self.last_safe_time.nanoseconds) / 1e9 > 1.0:  # 1 second timeout
            self.get_logger().error('Sensor timeout - triggering emergency stop')
            self.trigger_emergency_stop()

    def trigger_emergency_stop(self):
        """Trigger emergency stop procedures"""
        self.emergency_stop = True

        # Publish emergency stop command
        stop_msg = Bool()
        stop_msg.data = True
        self.emergency_stop_pub.publish(stop_msg)

        self.get_logger().error('EMERGENCY STOP ACTIVATED')

    def emergency_stop_callback(self, msg):
        """Handle external emergency stop requests"""
        if msg.data:
            self.trigger_emergency_stop()

    def reset_safety(self):
        """Reset safety system (only after addressing violations)"""
        if self.emergency_stop:
            self.get_logger().info('Safety system reset requested')
            # Additional checks would be needed before actually resetting
            # For example, confirming robot is in safe position
            self.emergency_stop = False
            self.get_logger().info('Safety system reset')

def main(args=None):
    rclpy.init(args=args)
    safety_node = SafetySystemNode()

    try:
        rclpy.spin(safety_node)
    except KeyboardInterrupt:
        pass
    finally:
        safety_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Controller Adaptation for Real Hardware

Adapting simulation controllers for real robot characteristics:

```python title="controller_adaptation.py"
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from control_msgs.msg import JointTrajectoryControllerState
import math

class AdaptiveControllerNode(Node):
    def __init__(self):
        super().__init__('adaptive_controller')

        # Controller parameters that need adaptation
        self.controller_params = {
            'kp': np.array([100.0, 100.0, 100.0, 80.0, 80.0, 50.0]),  # Position gains
            'kd': np.array([10.0, 10.0, 10.0, 8.0, 8.0, 5.0]),        # Velocity gains
            'ki': np.array([1.0, 1.0, 1.0, 0.8, 0.8, 0.5]),          # Integral gains
        }

        # Adaptation parameters
        self.adaptation_rate = 0.001
        self.error_history = []
        self.max_history = 100

        # Robot state
        self.current_position = None
        self.current_velocity = None
        self.desired_position = None

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)

        # Publishers
        self.command_pub = self.create_publisher(
            Float64MultiArray, '/joint_group_position_controller/commands', 10)

        # Timer for control loop
        self.control_timer = self.create_timer(0.005, self.control_loop)  # 200Hz

        self.get_logger().info('Adaptive Controller initialized')

    def joint_callback(self, msg):
        """Process joint state data"""
        self.current_position = np.array(msg.position)
        self.current_velocity = np.array(msg.velocity)

    def imu_callback(self, msg):
        """Process IMU data for balance control"""
        # Use IMU data for balance feedback
        pass

    def compute_control_output(self, desired_pos, current_pos, current_vel):
        """Compute PID control output with adaptation"""
        # Compute errors
        position_error = desired_pos - current_pos
        velocity_error = -current_vel  # Assuming desired velocity is 0

        # Store error for adaptation
        self.error_history.append(np.mean(np.abs(position_error)))
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)

        # Compute PID terms
        p_term = self.controller_params['kp'] * position_error
        d_term = self.controller_params['kd'] * velocity_error

        # Compute integral term (with anti-windup)
        if hasattr(self, 'integral_error'):
            self.integral_error += position_error * 0.005  # dt = 0.005s
            # Clamp integral to prevent windup
            self.integral_error = np.clip(self.integral_error, -1.0, 1.0)
        else:
            self.integral_error = np.zeros_like(position_error)

        i_term = self.controller_params['ki'] * self.integral_error

        # Total control output
        control_output = p_term + d_term + i_term

        # Adapt parameters based on performance
        self.adapt_parameters()

        return control_output

    def adapt_parameters(self):
        """Adapt controller parameters based on performance"""
        if len(self.error_history) < 10:
            return

        # Compute average error
        avg_error = np.mean(self.error_history[-10:])

        # Adapt based on error trend
        if avg_error > 0.1:  # High error
            # Increase gains to reduce error
            self.controller_params['kp'] *= (1 + self.adaptation_rate)
            self.controller_params['kd'] *= (1 + self.adaptation_rate * 0.5)
        elif avg_error < 0.01:  # Very low error (possible oscillation)
            # Decrease gains to reduce oscillation
            self.controller_params['kp'] *= (1 - self.adaptation_rate * 0.5)
            self.controller_params['kd'] *= (1 + self.adaptation_rate)

        # Keep gains within reasonable bounds
        self.controller_params['kp'] = np.clip(self.controller_params['kp'], 10, 500)
        self.controller_params['kd'] = np.clip(self.controller_params['kd'], 1, 50)

    def control_loop(self):
        """Main control loop"""
        if self.current_position is None or self.desired_position is None:
            return

        # Compute control commands
        commands = self.compute_control_output(
            self.desired_position,
            self.current_position,
            self.current_velocity if self.current_velocity is not None else np.zeros(len(self.current_position))
        )

        # Publish commands
        cmd_msg = Float64MultiArray()
        cmd_msg.data = commands.tolist()
        self.command_pub.publish(cmd_msg)

    def set_desired_trajectory(self, trajectory):
        """Set desired trajectory for the controller"""
        # This would implement trajectory following
        pass

def main(args=None):
    rclpy.init(args=args)
    controller = AdaptiveControllerNode()

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

## Exercises

1. Implement a system identification procedure for a real humanoid robot
2. Design and implement a sensor calibration protocol
3. Create a safety system with multiple layers of protection

## Summary

Transitioning from simulation to real humanoid robots requires careful attention to the reality gap, proper calibration, and robust safety systems. By understanding the differences between simulation and reality, implementing proper system identification, and developing adaptive control strategies, you can successfully deploy simulation-trained controllers on real hardware. Safety must always be the top priority when working with physical robots, especially humanoids that operate in human environments.