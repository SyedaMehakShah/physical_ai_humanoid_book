---
sidebar_position: 6
---

# Troubleshooting Guide: Physical AI & Humanoid Robotics

## Learning Objectives
- Identify common issues in humanoid robotics development
- Apply systematic troubleshooting methodologies
- Resolve problems with ROS 2, simulation, and hardware
- Implement effective debugging strategies
- Maintain system reliability and safety

## Troubleshooting Methodology

### 1. Systematic Problem-Solving Approach

When encountering issues with humanoid robots, follow this systematic approach:

1. **Identify the Problem**: Clearly define what is happening vs. what should happen
2. **Gather Information**: Collect logs, error messages, and system state
3. **Form Hypotheses**: Develop theories about potential causes
4. **Test Hypotheses**: Conduct experiments to validate or invalidate theories
5. **Implement Solution**: Apply the fix that addresses the root cause
6. **Verify Resolution**: Confirm the problem is solved and no new issues emerged
7. **Document**: Record the issue and solution for future reference

### 2. Information Gathering

#### Essential Diagnostic Commands

```bash
# Check ROS 2 system status
ros2 node list
ros2 topic list
ros2 service list
ros2 action list

# Monitor system resources
htop
df -h
free -h

# Check network connectivity
ifconfig
ping [robot_ip]

# Check ROS 2 environment
printenv | grep ROS
source /opt/ros/humble/setup.bash
```

#### Log Collection

```bash
# ROS 2 logging
ros2 run rcl_logging_spdlog list_loggers
ros2 run rcl_logging_spdlog set_logger_level [logger_name] [level]

# System logs
journalctl -u [service_name]
dmesg | grep [keyword]

# ROS 2 bag recording for debugging
ros2 bag record -a -o debug_session_$(date +%Y%m%d_%H%M%S)
```

## ROS 2 Troubleshooting

### 1. Communication Issues

#### Topic Connection Problems
**Symptoms**: Publishers/subscribers not connecting, data not flowing

**Diagnosis**:
```bash
# Check topic status
ros2 topic info /topic_name

# List all topics with details
ros2 topic list -v

# Echo topic to verify data flow
ros2 topic echo /topic_name --field [field_name]
```

**Solutions**:
1. **Domain ID Conflicts**: Ensure all nodes use the same ROS_DOMAIN_ID
```bash
export ROS_DOMAIN_ID=0
```

2. **QoS Mismatch**: Verify publisher and subscriber QoS profiles match
```python
# Example of matching QoS profiles
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE
)

publisher = node.create_publisher(MsgType, 'topic', qos_profile)
subscriber = node.create_subscription(MsgType, 'topic', callback, qos_profile)
```

3. **Network Configuration**: Check firewall and network settings
```bash
# Check for ROS 2 ports (typically 11811-11911)
sudo netstat -tulpn | grep 11811

# Test multicast (required for DDS discovery)
ping 224.0.0.1  # Basic multicast test
```

#### Service Call Failures
**Symptoms**: Service calls timeout or return errors

**Solutions**:
1. Verify service server is running
```bash
ros2 service list
ros2 service info /service_name
```

2. Check service interface compatibility
```bash
ros2 interface show [package_name]/srv/[ServiceName]
```

### 2. Performance Issues

#### High Latency
**Symptoms**: Delayed responses, missed control deadlines

**Troubleshooting**:
```bash
# Monitor topic latency
ros2 topic hz /topic_name

# Check system performance
vmstat 1 10
iostat -x 1 10

# Profile ROS 2 nodes
ros2 run tracetools_analysis convert [trace_directory]
```

**Solutions**:
1. **Increase QoS history depth** for critical topics
2. **Use transient_local durability** for configuration topics
3. **Optimize control loop timing**
4. **Reduce message size** by using compressed formats

#### Memory Leaks
**Symptoms**: Gradually decreasing performance, system slowdown

**Detection**:
```bash
# Monitor process memory usage
watch 'ps aux --sort=-%mem | head -20'

# Check for ROS 2 node memory usage
ros2 run ament_cppcheck cppcheck --show-ids [package_path]
```

**Prevention**:
```python
# Proper resource management in ROS 2 nodes
class ProperResourceManagement(Node):
    def __init__(self):
        super().__init__('resource_managed_node')
        self.timers = []
        self.subscribers = []
        self.publishers = []

    def destroy_node(self):
        """Clean up all resources before destroying node"""
        # Cancel timers
        for timer in self.timers:
            timer.cancel()

        # Destroy subscriptions and publishers
        for sub in self.subscribers:
            sub.destroy()
        for pub in self.publishers:
            pub.destroy()

        super().destroy_node()
```

### 3. Lifecycle Management Issues

#### Node Startup Problems
**Symptoms**: Nodes fail to initialize or crash during startup

**Diagnosis**:
```bash
# Check node launch details
ros2 launch [package] [launch_file] --show-args

# Monitor launch process
ros2 launch [package] [launch_file] --noninteractive
```

**Solutions**:
1. **Parameter Validation**: Ensure all required parameters are provided
```python
# Proper parameter declaration and validation
class ParameterizedNode(Node):
    def __init__(self):
        super().__init__('parameterized_node')

        # Declare parameters with defaults
        self.declare_parameter('control_frequency', 100)
        self.declare_parameter('safety_timeout', 1.0)

        # Validate parameters
        freq = self.get_parameter('control_frequency').value
        if freq <= 0 or freq > 1000:
            self.get_logger().fatal('Invalid control frequency')
            raise ValueError('Control frequency must be between 0 and 1000 Hz')
```

2. **Resource Dependencies**: Check for missing dependencies before initialization
```python
def wait_for_services(self, service_names, timeout=10.0):
    """Wait for critical services before proceeding"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        all_available = True
        for service_name in service_names:
            if not self.client_available(service_name):
                all_available = False
                break

        if all_available:
            return True
        time.sleep(0.1)

    return False
```

## Simulation Troubleshooting

### 1. Gazebo Simulation Issues

#### Physics Instability
**Symptoms**: Robot shaking, exploding, or unrealistic behavior

**Causes and Solutions**:
1. **Time Step Issues**: Adjust physics time step
```xml
<!-- physics.config -->
<physics type="ode">
  <max_step_size>0.001</max_step_size>  <!-- Smaller for stability -->
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000.0</real_time_update_rate>
</physics>
```

2. **Solver Parameters**: Tune solver for stability
```xml
<physics type="ode">
  <ode>
    <solver>
      <type>quick</type>
      <iters>1000</iters>  <!-- More iterations for stability -->
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.000001</cfm>  <!-- Constraint Force Mixing -->
      <erp>0.2</erp>      <!-- Error Reduction Parameter -->
    </constraints>
  </ode>
</physics>
```

3. **Mass and Inertia**: Verify physical properties
```xml
<link name="link_name">
  <inertial>
    <mass value="1.0"/>  <!-- Realistic mass -->
    <inertia
      ixx="0.01" ixy="0.0" ixz="0.0"
      iyy="0.01" iyz="0.0"
      izz="0.01"/>  <!-- Positive definite inertia matrix -->
  </inertial>
</link>
```

#### Model Loading Problems
**Symptoms**: Models fail to load or appear incorrectly

**Solutions**:
1. **Check Model Paths**: Verify all mesh and texture paths exist
```bash
# Verify model structure
ls -la ~/.gazebo/models/[model_name]/
find ~/.gazebo/models/[model_name]/ -name "*.dae" -o -name "*.stl" -o -name "*.jpg" -o -name "*.png"
```

2. **Validate URDF/XACRO**: Check for syntax errors
```bash
# Validate URDF
check_urdf [path_to_urdf_file]
```

3. **Material Issues**: Ensure materials are properly defined
```xml
<material name="blue">
  <color rgba="0 0 1 1"/>
</material>

<gazebo reference="link_name">
  <material>Gazebo/Blue</material>
</gazebo>
```

### 2. Isaac Sim Troubleshooting

#### GPU/CUDA Issues
**Symptoms**: Rendering errors, poor performance, crashes

**Diagnosis**:
```bash
# Check GPU status
nvidia-smi

# Verify CUDA installation
nvcc --version
nvidia-ml-py3 --version

# Check Isaac Sim logs
cat ~/.nvidia-isaac/logs/isaac_sim.log
```

**Solutions**:
1. **Update Drivers**: Ensure latest NVIDIA drivers
```bash
sudo apt update
sudo apt install nvidia-driver-535
```

2. **Adjust Graphics Settings**: Reduce quality for performance
```python
# Isaac Sim configuration
CONFIG = {
    "renderer": {
        "resolution": [1280, 720],  # Lower resolution
        "render_products": {
            "max_update_count": 30,  # Limit updates per second
        }
    }
}
```

#### Scene Loading Failures
**Symptoms**: Scenes fail to load or crash on startup

**Solutions**:
1. **Check Asset Paths**: Verify all asset paths are correct
2. **Reduce Scene Complexity**: Simplify scene for debugging
3. **Memory Management**: Monitor GPU memory usage

## Hardware Troubleshooting

### 1. Actuator Issues

#### Joint Position Errors
**Symptoms**: Joints not reaching commanded positions, drifting

**Diagnosis**:
```bash
# Monitor joint states
ros2 topic echo /joint_states

# Check for specific joint issues
ros2 run rqt_plot rqt_plot /joint_states/position[0]
```

**Solutions**:
1. **Calibration**: Recalibrate joint offsets
```bash
# Example joint calibration
class JointCalibrator:
    def calibrate_joint(self, joint_name, home_position):
        """Calibrate joint to known position"""
        # Move to home position
        self.move_to_position(joint_name, home_position)

        # Record current encoder value as reference
        current_encoder = self.get_encoder_value(joint_name)

        # Calculate offset
        offset = home_position - current_encoder
        self.set_offset(joint_name, offset)
```

2. **Control Tuning**: Adjust PID gains for better tracking
```python
# Adaptive PID control
class AdaptivePIDController:
    def __init__(self, kp=10.0, ki=1.0, kd=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def compute(self, error, dt):
        """Compute control output with adaptive gains"""
        # Update integral
        self.integral += error * dt

        # Calculate derivative
        derivative = (error - self.previous_error) / dt if dt > 0 else 0

        # Compute output
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        # Adaptive gain adjustment based on error magnitude
        if abs(error) > 1.0:  # Large error - increase response
            output *= 1.5
        elif abs(error) < 0.01:  # Small error - reduce noise
            output *= 0.5

        self.previous_error = error
        return output
```

#### Overheating Motors
**Symptoms**: Motors getting hot, reduced performance, thermal shutdowns

**Prevention**:
```python
# Thermal monitoring and protection
class ThermalProtection:
    def __init__(self, max_temp=70.0, shutdown_temp=80.0):
        self.max_temp = max_temp
        self.shutdown_temp = shutdown_temp
        self.motor_temps = {}

    def check_thermal_limits(self):
        """Check motor temperatures and apply limits"""
        for motor_id, temp in self.motor_temps.items():
            if temp > self.shutdown_temp:
                self.emergency_stop(motor_id)
            elif temp > self.max_temp:
                self.reduce_power(motor_id, 0.5)  # Reduce to 50% power
            elif temp > self.max_temp - 5:  # Within 5°C of limit
                self.reduce_power(motor_id, 0.8)  # Reduce to 80% power

    def emergency_stop(self, motor_id):
        """Apply emergency stop to overheating motor"""
        self.set_motor_command(motor_id, 0.0)  # Zero command
        self.log_thermal_event(motor_id, "EMERGENCY_STOP")
```

### 2. Sensor Troubleshooting

#### IMU Drift
**Symptoms**: Orientation estimates drifting over time

**Solutions**:
```python
# IMU drift compensation
import numpy as np
from scipy.spatial.transform import Rotation as R

class IMUDriftCompensator:
    def __init__(self, correction_rate=0.01):
        self.correction_rate = correction_rate
        self.reference_orientation = np.array([0, 0, 0, 1])  # Identity quaternion
        self.bias_estimate = np.zeros(3)  # Gyro bias estimate

    def compensate_drift(self, raw_gyro, raw_accel, dt):
        """Compensate for IMU drift using accelerometer reference"""
        # Estimate bias from accelerometer when stationary
        accel_magnitude = np.linalg.norm(raw_accel)
        if abs(accel_magnitude - 9.81) < 0.5:  # Close to gravity
            # Assume robot is stationary, estimate gyro bias
            self.bias_estimate += self.correction_rate * (raw_gyro - self.bias_estimate)

        # Correct gyro reading
        corrected_gyro = raw_gyro - self.bias_estimate

        # Integrate corrected gyro for orientation
        delta_q = self.gyro_to_quaternion(corrected_gyro, dt)
        current_orientation = self.integrate_quaternion(self.current_orientation, delta_q)

        return current_orientation

    def gyro_to_quaternion(self, gyro, dt):
        """Convert gyro rates to quaternion delta"""
        norm = np.linalg.norm(gyro)
        if norm < 1e-6:
            return np.array([0, 0, 0, 1])

        half_angle = norm * dt * 0.5
        s = np.sin(half_angle) / norm
        return np.array([s * gyro[0], s * gyro[1], s * gyro[2], np.cos(half_angle)])
```

#### Camera Calibration Issues
**Symptoms**: Poor computer vision performance, incorrect depth estimates

**Solutions**:
```bash
# Camera calibration using ROS 2
ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.025 image:=/camera/image_raw camera:=/camera
```

```python
# Camera intrinsics validation
def validate_camera_calibration(camera_info):
    """Validate camera calibration parameters"""
    # Check that all required parameters are present
    required_fields = ['k', 'd', 'r', 'p']
    for field in required_fields:
        if not hasattr(camera_info, field):
            raise ValueError(f"Missing calibration parameter: {field}")

    # Check that distortion coefficients are reasonable
    if len(camera_info.d) < 4:
        raise ValueError("Insufficient distortion coefficients")

    # Check that intrinsic matrix is reasonable
    k_matrix = np.array(camera_info.k).reshape(3, 3)
    if k_matrix[0, 0] < 100 or k_matrix[1, 1] < 100:  # Too small focal length
        raise ValueError("Suspiciously small focal lengths")

    if k_matrix[0, 2] < 0 or k_matrix[1, 2] < 0:  # Principal point outside image
        raise ValueError("Principal point outside image bounds")
```

## Control System Troubleshooting

### 1. Stability Issues

#### Oscillation Problems
**Symptoms**: Robot oscillating around desired position, unstable behavior

**Analysis**:
```python
# Stability analysis tools
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class StabilityAnalyzer:
    def __init__(self):
        self.data_buffer = {'time': [], 'position': [], 'command': [], 'error': []}

    def collect_data(self, time_stamp, position, command):
        """Collect data for stability analysis"""
        error = command - position
        self.data_buffer['time'].append(time_stamp)
        self.data_buffer['position'].append(position)
        self.data_buffer['command'].append(command)
        self.data_buffer['error'].append(error)

    def analyze_stability(self):
        """Analyze collected data for stability"""
        if len(self.data_buffer['error']) < 100:
            return "Insufficient data for analysis"

        errors = np.array(self.data_buffer['error'])

        # Calculate oscillation metrics
        amplitude = np.std(errors)
        frequency = self.estimate_oscillation_frequency(errors)
        damping_ratio = self.estimate_damping_ratio(errors)

        analysis = {
            'amplitude': amplitude,
            'frequency': frequency,
            'damping_ratio': damping_ratio,
            'oscillating': amplitude > 0.1  # Threshold for oscillation detection
        }

        return analysis

    def estimate_oscillation_frequency(self, errors):
        """Estimate dominant frequency of oscillation"""
        fft = np.fft.fft(errors)
        frequencies = np.fft.fftfreq(len(errors))
        power_spectrum = np.abs(fft)**2

        # Find peak frequency
        peak_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2])
        return frequencies[peak_idx + 1]

    def estimate_damping_ratio(self, errors):
        """Estimate damping ratio from decay envelope"""
        # Simple estimation - in practice, use more sophisticated methods
        envelope = np.abs(signal.hilbert(errors))
        log_envelope = np.log(envelope + 1e-6)  # Add small value to avoid log(0)

        # Estimate decay rate
        time_axis = np.array(self.data_buffer['time'])
        slope, _ = np.polyfit(time_axis, log_envelope, 1)

        # Damping ratio approximation
        return -slope if slope < 0 else 0.1  # Positive damping ratio
```

**Solutions**:
1. **Gain Adjustment**: Reduce proportional gain to reduce oscillation
2. **Derivative Action**: Add derivative term for damping
3. **Filtering**: Apply low-pass filtering to reduce noise-induced oscillation

```python
# Anti-oscillation controller
class AntiOscillationController:
    def __init__(self, kp=10.0, ki=1.0, kd=2.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0
        self.error_derivative = 0

        # Anti-oscillation parameters
        self.oscillation_threshold = 0.05
        self.oscillation_counter = 0
        self.oscillation_detected = False

    def compute_with_anti_oscillation(self, error, dt):
        """Compute control with oscillation detection and mitigation"""
        # Update derivative
        if dt > 0:
            self.error_derivative = (error - self.previous_error) / dt

        # Check for oscillation pattern
        if abs(error) > self.oscillation_threshold and np.sign(error) != np.sign(self.previous_error):
            self.oscillation_counter += 1
            if self.oscillation_counter > 3:  # Three consecutive sign changes
                self.oscillation_detected = True
        else:
            self.oscillation_counter = 0
            self.oscillation_detected = False

        # Adjust gains based on oscillation detection
        kp_adj = self.kp * (0.5 if self.oscillation_detected else 1.0)
        kd_adj = self.kd * (2.0 if self.oscillation_detected else 1.0)

        # Compute control output
        self.integral += error * dt
        output = (kp_adj * error) + (self.ki * self.integral) + (kd_adj * self.error_derivative)

        self.previous_error = error
        return output
```

### 2. Safety System Troubleshooting

#### Emergency Stop Issues
**Symptoms**: Emergency stop activating unexpectedly, not responding to genuine emergencies

**Diagnosis**:
```python
# Emergency stop monitoring
class EmergencyStopMonitor:
    def __init__(self):
        self.safety_violations = []
        self.emergency_stop_reasons = []
        self.safety_thresholds = {
            'position': 3.14,      # Joint position limits
            'velocity': 10.0,      # Joint velocity limits
            'effort': 200.0,       # Joint effort limits
            'distance': 0.3,       # Human proximity limits
            'temperature': 70.0    # Temperature limits
        }

    def check_safety_conditions(self, robot_state):
        """Check all safety conditions"""
        violations = []

        # Check joint limits
        for i, (pos, vel, effort) in enumerate(zip(
            robot_state['position'],
            robot_state['velocity'],
            robot_state['effort']
        )):
            if abs(pos) > self.safety_thresholds['position']:
                violations.append(f'Joint {i} position limit: {pos}')

            if abs(vel) > self.safety_thresholds['velocity']:
                violations.append(f'Joint {i} velocity limit: {vel}')

            if abs(effort) > self.safety_thresholds['effort']:
                violations.append(f'Joint {i} effort limit: {effort}')

        # Check proximity to humans
        if robot_state.get('human_distance', float('inf')) < self.safety_thresholds['distance']:
            violations.append(f'Human proximity: {robot_state["human_distance"]}m')

        # Check temperature
        if robot_state.get('temperature', 25) > self.safety_thresholds['temperature']:
            violations.append(f'Temperature limit: {robot_state["temperature"]}°C')

        return violations

    def handle_emergency_stop(self, violations):
        """Handle emergency stop with proper logging"""
        if violations:
            self.emergency_stop_reasons.extend(violations)

            # Log the incident
            timestamp = time.time()
            incident_report = {
                'timestamp': timestamp,
                'violations': violations,
                'robot_state': self.current_robot_state,
                'actions_taken': ['motion_stop', 'power_cut', 'alert_sent']
            }

            self.log_safety_incident(incident_report)
            return True

        return False
```

#### False Positives in Safety Systems
**Symptoms**: Safety systems triggering unnecessarily

**Solutions**:
1. **Hysteresis**: Add hysteresis to prevent chattering
2. **Filtering**: Apply filtering to reduce noise-induced triggers
3. **Debouncing**: Add time delays to prevent rapid state changes

```python
# Debounced safety checker
class DebouncedSafetyChecker:
    def __init__(self, debounce_time=0.1, hysteresis=0.05):
        self.debounce_time = debounce_time
        self.hysteresis = hysteresis
        self.trigger_time = 0
        self.current_state = False
        self.debounced_state = False

    def check_with_debounce(self, raw_condition, current_time):
        """Check condition with debouncing"""
        if raw_condition and not self.current_state:
            # Condition became true - start timer
            if current_time - self.trigger_time > self.debounce_time:
                self.current_state = True
                self.trigger_time = current_time
        elif not raw_condition and self.current_state:
            # Condition became false - start timer with hysteresis
            if current_time - self.trigger_time > self.debounce_time:
                self.current_state = False
                self.trigger_time = current_time

        # Update debounced state based on timing
        if self.current_state and current_time - self.trigger_time >= self.debounce_time:
            self.debounced_state = True
        elif not self.current_state and current_time - self.trigger_time >= self.debounce_time:
            self.debounced_state = False

        return self.debounced_state
```

## Debugging Strategies

### 1. Logging and Monitoring

#### Comprehensive Logging
```python
import logging
import json
from datetime import datetime

class RobotLogger:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger('robot_logger')
        self.logger.setLevel(log_level)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # File handler for persistent logs
        file_handler = logging.FileHandler(f'robot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler for real-time monitoring
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log_robot_state(self, state_dict):
        """Log comprehensive robot state"""
        state_json = json.dumps(state_dict, indent=2, default=str)
        self.logger.info(f"Robot State:\n{state_json}")

    def log_control_event(self, control_type, command, feedback, error):
        """Log control system events"""
        event_data = {
            'type': 'control_event',
            'control_type': control_type,
            'command': command,
            'feedback': feedback,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.debug(f"Control Event: {event_data}")

    def log_safety_violation(self, violation_type, details, severity='WARNING'):
        """Log safety violations"""
        violation_data = {
            'type': 'safety_violation',
            'violation_type': violation_type,
            'details': details,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        if severity == 'ERROR':
            self.logger.error(f"Safety Violation: {violation_data}")
        elif severity == 'CRITICAL':
            self.logger.critical(f"Safety Violation: {violation_data}")
        else:
            self.logger.warning(f"Safety Violation: {violation_data}")
```

#### Real-time Monitoring
```python
# Real-time monitoring dashboard
import time
import threading
from collections import deque

class RealtimeMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.metrics = {
            'cpu_usage': deque(maxlen=window_size),
            'memory_usage': deque(maxlen=window_size),
            'control_latency': deque(maxlen=window_size),
            'safety_events': deque(maxlen=window_size)
        }
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """Start real-time monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.monitor_thread.start()

    def monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            # Collect metrics
            self.collect_metrics()

            # Check for anomalies
            self.check_anomalies()

            time.sleep(0.1)  # 10Hz monitoring

    def collect_metrics(self):
        """Collect system metrics"""
        import psutil

        # CPU usage
        self.metrics['cpu_usage'].append(psutil.cpu_percent())

        # Memory usage
        memory_info = psutil.virtual_memory()
        self.metrics['memory_usage'].append(memory_info.percent)

        # Control latency (if available)
        # This would come from control system timestamps
        self.metrics['control_latency'].append(0.01)  # Placeholder

    def check_anomalies(self):
        """Check for anomalous conditions"""
        if len(self.metrics['cpu_usage']) >= 10:
            recent_cpu = list(self.metrics['cpu_usage'])[-10:]
            if max(recent_cpu) > 90:
                print(f"⚠️  HIGH CPU USAGE: {max(recent_cpu)}%")

            recent_mem = list(self.metrics['memory_usage'])[-10:]
            if max(recent_mem) > 90:
                print(f"⚠️  HIGH MEMORY USAGE: {max(recent_mem)}%")

    def get_current_metrics(self):
        """Get current system metrics"""
        return {
            'cpu_avg': sum(self.metrics['cpu_usage']) / len(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0,
            'memory_avg': sum(self.metrics['memory_usage']) / len(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0,
            'latency_avg': sum(self.metrics['control_latency']) / len(self.metrics['control_latency']) if self.metrics['control_latency'] else 0,
            'recent_safety_events': len([e for e in list(self.metrics['safety_events'])[-10:] if e > 0])
        }
```

### 2. Remote Debugging

#### SSH and Remote Access
```bash
# Secure remote debugging setup
ssh -X [username]@[robot_ip]  # X11 forwarding for GUI
scp -r [local_path] [username]@[robot_ip]:[remote_path]  # File transfer

# Screen/tmux for persistent sessions
screen -S robot_debug
tmux new-session -s robot_debug

# Port forwarding for ROS 2
ssh -L 11311:localhost:11311 [username]@[robot_ip]
```

#### Distributed Debugging
```python
# Remote debugging server
import socket
import threading
import pickle

class RemoteDebugger:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.clients = []
        self.debugging = False

    def start_server(self):
        """Start remote debugging server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)

        print(f"Remote debugger listening on {self.host}:{self.port}")

        while True:
            client_socket, address = self.server_socket.accept()
            print(f"Client connected: {address}")
            self.clients.append(client_socket)

            client_thread = threading.Thread(
                target=self.handle_client,
                args=(client_socket,)
            )
            client_thread.daemon = True
            client_thread.start()

    def handle_client(self, client_socket):
        """Handle communication with remote client"""
        try:
            while True:
                # Receive command from client
                data = client_socket.recv(4096)
                if not data:
                    break

                command = pickle.loads(data)

                # Execute command and return result
                result = self.execute_remote_command(command)
                response = pickle.dumps(result)
                client_socket.send(response)

        except Exception as e:
            print(f"Client error: {e}")
        finally:
            client_socket.close()
            if client_socket in self.clients:
                self.clients.remove(client_socket)

    def execute_remote_command(self, command):
        """Execute command on robot and return result"""
        if command['type'] == 'get_state':
            return self.get_robot_state()
        elif command['type'] == 'set_param':
            return self.set_parameter(command['param'], command['value'])
        elif command['type'] == 'execute_action':
            return self.execute_action(command['action'])
        else:
            return {'error': 'Unknown command type'}
```

## Recovery Procedures

### 1. System Recovery

#### Graceful Degradation
```python
class GracefulDegradationManager:
    def __init__(self):
        self.system_modes = {
            'nominal': 100,      # Full functionality
            'reduced': 70,       # Limited functionality
            'safe': 30,          # Minimal safe operation
            'emergency': 0       # Complete stop
        }
        self.current_mode = 'nominal'

    def assess_system_health(self):
        """Assess overall system health"""
        health_metrics = {
            'sensors': self.check_sensor_health(),
            'actuators': self.check_actuator_health(),
            'computing': self.check_computing_resources(),
            'communications': self.check_communication_health()
        }

        # Calculate overall health score
        weights = {'sensors': 0.25, 'actuators': 0.3, 'computing': 0.2, 'communications': 0.25}
        health_score = sum(health_metrics[k] * weights[k] for k in weights.keys())

        return health_score, health_metrics

    def degrade_gracefully(self, health_score):
        """Degrade system functionality based on health"""
        if health_score >= 90:
            self.set_system_mode('nominal')
        elif health_score >= 70:
            self.set_system_mode('reduced')
            # Disable non-critical functions
            self.disable_non_critical_functions()
        elif health_score >= 40:
            self.set_system_mode('safe')
            # Only essential safety functions
            self.enable_safety_only()
        else:
            self.set_system_mode('emergency')
            # Complete stop for safety

    def set_system_mode(self, mode):
        """Set system operational mode"""
        if mode != self.current_mode:
            self.log_mode_change(self.current_mode, mode)
            self.current_mode = mode
            self.apply_mode_restrictions(mode)

    def disable_non_critical_functions(self):
        """Disable non-critical functions in reduced mode"""
        # Disable advanced perception
        # Reduce motion speed
        # Simplify control algorithms
        pass

    def enable_safety_only(self):
        """Enable only safety-critical functions"""
        # Stop all non-essential motion
        # Enable safety monitoring only
        # Maintain basic communications
        pass
```

### 2. Failure Recovery

#### Component Recovery
```python
class ComponentRecoveryManager:
    def __init__(self):
        self.components = {}
        self.recovery_attempts = {}
        self.max_recovery_attempts = 3

    def register_component(self, name, health_check_func, restart_func):
        """Register a recoverable component"""
        self.components[name] = {
            'health_check': health_check_func,
            'restart': restart_func,
            'status': 'healthy',
            'last_recovery': None
        }

    def monitor_components(self):
        """Monitor all registered components"""
        for name, comp in self.components.items():
            if not comp['health_check']():
                self.handle_component_failure(name)

    def handle_component_failure(self, component_name):
        """Handle component failure with recovery attempts"""
        comp = self.components[component_name]

        # Check if we've exceeded recovery attempts
        if component_name in self.recovery_attempts:
            if self.recovery_attempts[component_name] >= self.max_recovery_attempts:
                comp['status'] = 'failed_perm'
                self.log_permanent_failure(component_name)
                return

        # Attempt recovery
        try:
            success = comp['restart']()
            if success:
                comp['status'] = 'healthy'
                comp['last_recovery'] = time.time()
                self.recovery_attempts[component_name] = 0
                self.log_recovery_success(component_name)
            else:
                comp['status'] = 'failed_temp'
                self.recovery_attempts[component_name] = \
                    self.recovery_attempts.get(component_name, 0) + 1
                self.log_recovery_failure(component_name)

        except Exception as e:
            comp['status'] = 'failed_temp'
            self.recovery_attempts[component_name] = \
                self.recovery_attempts.get(component_name, 0) + 1
            self.log_recovery_exception(component_name, e)

    def get_recovery_status(self):
        """Get overall recovery status"""
        status = {
            'components': {},
            'recovery_needed': 0,
            'permanent_failures': 0
        }

        for name, comp in self.components.items():
            status['components'][name] = comp['status']
            if comp['status'] == 'failed_temp':
                status['recovery_needed'] += 1
            elif comp['status'] == 'failed_perm':
                status['permanent_failures'] += 1

        return status
```

## Best Practices for Troubleshooting

### 1. Preventive Measures
- Regular system health checks
- Automated monitoring and alerts
- Scheduled maintenance windows
- Comprehensive logging from the start

### 2. Documentation
- Maintain detailed troubleshooting guides
- Document common issues and solutions
- Keep system configuration records
- Record incident reports with resolutions

### 3. Testing
- Regular system tests under various conditions
- Stress testing for edge cases
- Integration testing between components
- Safety system testing procedures

This troubleshooting guide provides a comprehensive approach to identifying, diagnosing, and resolving common issues in humanoid robotics systems. The systematic methodology ensures that problems are addressed efficiently while maintaining safety and reliability.