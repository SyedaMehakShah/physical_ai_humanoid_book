---
sidebar_position: 5
---

# Sensor Simulation for Humanoid Robots

## Learning Objectives
- Understand the types of sensors used in humanoid robots
- Learn to simulate various sensors in Gazebo and Unity
- Implement sensor fusion techniques for humanoid perception
- Validate sensor models against real-world performance

## Intuition

Sensor simulation is like giving your virtual robot all the senses it would have in the real world. Just as humans use eyes to see, ears to hear, and skin to feel, robots use cameras, LiDAR, IMUs, and other sensors to perceive their environment. In simulation, we create virtual versions of these sensors that behave similarly to their real counterparts, allowing us to test perception algorithms safely before deploying them on actual robots.

## Concept

Sensor simulation in robotics involves modeling the behavior of physical sensors in virtual environments. For humanoid robots, this includes:
- **Proprioceptive sensors**: Joint encoders, IMUs, force/torque sensors
- **Exteroceptive sensors**: Cameras, LiDAR, ultrasonic sensors
- **Tactile sensors**: Contact sensors, pressure sensors

Each sensor type has specific characteristics that must be accurately modeled:
- Noise and uncertainty
- Range and resolution limitations
- Update rates and latency
- Environmental dependencies

## Common Humanoid Robot Sensors

### 1. Inertial Measurement Units (IMUs)
IMUs provide orientation, angular velocity, and linear acceleration:

```xml title="imu_sensor.urdf"
<gazebo reference="torso_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.02</stddev>  <!-- 0.02 rad/s noise -->
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
            <stddev>0.17</stddev>  <!-- 0.17 m/s² noise -->
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
```

### 2. Force/Torque Sensors
For detecting contact and measuring forces at joints:

```xml title="force_torque_sensor.urdf"
<gazebo reference="ankle_joint">
  <sensor name="ft_sensor" type="force_torque">
    <always_on>true</always_on>
    <update_rate>500</update_rate>
    <force_torque>
      <frame>child</frame>
      <measure_direction>child_to_parent</measure_direction>
    </force_torque>
  </sensor>
</gazebo>
```

### 3. Camera Sensors
For vision-based perception:

```xml title="camera_sensor.urdf">
<gazebo reference="head_camera">
  <sensor name="camera" type="camera">
    <camera>
      <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <always_on>true</always_on>
    <update_rate>30</update_rate>
    <visualize>true</visualize>
  </sensor>
</gazebo>
```

## Sensor Processing Node

Here's how to process multiple sensor streams in ROS 2:

```python title="sensor_fusion_node.py"
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState, Image, PointCloud2
from geometry_msgs.msg import Vector3Stamped, PointStamped
from std_msgs.msg import Float64MultiArray
import numpy as np
import cv2
from cv_bridge import CvBridge

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # Sensor data storage
        self.imu_data = None
        self.joint_states = None
        self.camera_image = None
        self.foot_forces = Float64MultiArray()

        # Publishers
        self.state_estimate_pub = self.create_publisher(
            Vector3Stamped, 'robot_state_estimate', 10)
        self.contact_pub = self.create_publisher(
            PointStamped, 'contact_estimate', 10)

        # Subscribers
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)
        self.camera_sub = self.create_subscription(
            Image, 'camera/image_raw', self.camera_callback, 10)
        self.force_sub = self.create_subscription(
            Float64MultiArray, 'foot_forces', self.force_callback, 10)

        # Timer for sensor fusion
        self.fusion_timer = self.create_timer(0.01, self.sensor_fusion_callback)  # 100Hz

        self.cv_bridge = CvBridge()
        self.get_logger().info('Sensor Fusion Node initialized')

    def imu_callback(self, msg):
        """Process IMU data"""
        self.imu_data = {
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        }

    def joint_callback(self, msg):
        """Process joint state data"""
        self.joint_states = {
            'names': msg.name,
            'positions': msg.position,
            'velocities': msg.velocity,
            'efforts': msg.effort
        }

    def camera_callback(self, msg):
        """Process camera image"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.camera_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error processing camera image: {e}')

    def force_callback(self, msg):
        """Process force/torque sensor data"""
        self.foot_forces = msg.data

    def sensor_fusion_callback(self):
        """Fuse sensor data to estimate robot state"""
        if not all([self.imu_data, self.joint_states]):
            return

        # Estimate robot orientation from IMU
        orientation = self.estimate_orientation()

        # Estimate robot position from joint integration
        position = self.estimate_position()

        # Detect contact from force sensors
        contact_detected = self.detect_contact()

        # Publish fused state estimate
        state_msg = Vector3Stamped()
        state_msg.header.stamp = self.get_clock().now().to_msg()
        state_msg.header.frame_id = 'world'
        state_msg.vector.x = position[0]
        state_msg.vector.y = position[1]
        state_msg.vector.z = orientation[2]  # Z-axis orientation

        self.state_estimate_pub.publish(state_msg)

        if contact_detected:
            contact_msg = PointStamped()
            contact_msg.header.stamp = self.get_clock().now().to_msg()
            contact_msg.header.frame_id = 'world'
            contact_msg.point.x = position[0]
            contact_msg.point.y = position[1]
            contact_msg.point.z = 0.0  # Ground contact

            self.contact_pub.publish(contact_msg)

    def estimate_orientation(self):
        """Estimate robot orientation from IMU data"""
        # Simple orientation estimation (in practice, use sensor fusion filters)
        if self.imu_data:
            return self.imu_data['orientation']
        return [0, 0, 0, 1]  # Identity quaternion

    def estimate_position(self):
        """Estimate robot position from joint integration"""
        # Simplified position estimation (in practice, use forward kinematics)
        if self.joint_states:
            # This would involve complex forward kinematics in practice
            return [0.0, 0.0, 0.0]  # Placeholder
        return [0.0, 0.0, 0.0]

    def detect_contact(self):
        """Detect ground contact from force sensors"""
        # Check if forces exceed threshold
        if len(self.foot_forces) >= 2:  # Assuming left and right foot
            left_force = abs(self.foot_forces[0])
            right_force = abs(self.foot_forces[1])
            contact_threshold = 50.0  # Newtons
            return left_force > contact_threshold or right_force > contact_threshold
        return False

def main(args=None):
    rclpy.init(args=args)
    sensor_fusion = SensorFusionNode()

    try:
        rclpy.spin(sensor_fusion)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_fusion.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sensor Fusion Techniques

### 1. Kalman Filter for State Estimation
```python title="kalman_filter.py"
import numpy as np

class KalmanFilter:
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x  # State dimension
        self.dim_z = dim_z  # Measurement dimension

        # State vector [x, y, z, vx, vy, vz]
        self.x = np.zeros(dim_x)

        # Covariance matrix
        self.P = np.eye(dim_x) * 1000.0

        # Process noise
        self.Q = np.eye(dim_x) * 0.1

        # Measurement noise
        self.R = np.eye(dim_z) * 1.0

        # State transition matrix (constant velocity model)
        self.F = np.eye(dim_x)
        for i in range(3):
            self.F[i, i+3] = 1.0  # Position integrates velocity

        # Measurement matrix
        self.H = np.zeros((dim_z, dim_x))
        for i in range(min(dim_z, dim_x)):
            self.H[i, i] = 1.0

    def predict(self):
        """Predict next state"""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        """Update state with measurement"""
        y = z - np.dot(self.H, self.x)  # Residual
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R  # Residual covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman gain

        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
```

### 2. Particle Filter for Non-linear Systems
```python title="particle_filter.py"
class ParticleFilter:
    def __init__(self, num_particles=1000):
        self.num_particles = num_particles
        self.particles = np.random.uniform(-1, 1, (num_particles, 6))  # [x, y, z, roll, pitch, yaw]
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, control_input, noise_std):
        """Predict particle motion based on control input"""
        dt = 0.01  # Time step
        for i in range(self.num_particles):
            # Apply motion model with noise
            self.particles[i] += control_input * dt + np.random.normal(0, noise_std, 6)

    def update(self, measurement, measurement_std):
        """Update particle weights based on measurement"""
        for i in range(self.num_particles):
            # Calculate likelihood of measurement given particle state
            predicted_measurement = self.forward_model(self.particles[i])
            likelihood = self.gaussian_likelihood(measurement, predicted_measurement, measurement_std)
            self.weights[i] *= likelihood

        # Normalize weights
        self.weights += 1.e-300  # Avoid zero weights
        self.weights /= np.sum(self.weights)

    def resample(self):
        """Resample particles based on weights"""
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def forward_model(self, state):
        """Forward model to predict measurement from state"""
        # Simplified model - in practice, this would be complex
        return state[:3]  # Return position as measurement
```

## Unity Sensor Simulation

In Unity, sensors can be simulated using colliders and raycasting:

```csharp title="UnitySensorSimulation.cs"
using UnityEngine;

public class UnitySensorSimulation : MonoBehaviour
{
    public float cameraFov = 60f;
    public int cameraWidth = 640;
    public int cameraHeight = 480;
    public float lidarRange = 10f;
    public int lidarRays = 360;

    private Camera sensorCamera;
    private GameObject lidarSensor;

    void Start()
    {
        SetupCameraSensor();
        SetupLidarSensor();
    }

    void SetupCameraSensor()
    {
        // Create camera component for vision simulation
        sensorCamera = gameObject.AddComponent<Camera>();
        sensorCamera.fieldOfView = cameraFov;
        sensorCamera.aspect = (float)cameraWidth / cameraHeight;
        sensorCamera.orthographic = false;
    }

    void SetupLidarSensor()
    {
        // Create lidar sensor using raycasting
        lidarSensor = new GameObject("LidarSensor");
        lidarSensor.transform.SetParent(transform);
        lidarSensor.transform.localPosition = Vector3.zero;
    }

    public float[] SimulateLidar()
    {
        float[] ranges = new float[lidarRays];
        float angleStep = 360f / lidarRays;

        for (int i = 0; i < lidarRays; i++)
        {
            float angle = i * angleStep * Mathf.Deg2Rad;
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
            direction = transform.TransformDirection(direction);

            if (Physics.Raycast(transform.position, direction, out RaycastHit hit, lidarRange))
            {
                ranges[i] = hit.distance;
            }
            else
            {
                ranges[i] = lidarRange;  // No obstacle detected
            }
        }

        return ranges;
    }

    void Update()
    {
        // Simulate sensor data publishing
        if (Time.frameCount % 30 == 0)  // Every 30 frames (assuming 60 FPS)
        {
            float[] lidarData = SimulateLidar();
            // Publish lidar data to ROS bridge
            PublishLidarData(lidarData);
        }
    }

    void PublishLidarData(float[] ranges)
    {
        // Send data through ROS bridge
        // Implementation depends on specific ROS-Unity integration
    }
}
```

## Sensor Validation and Calibration

### 1. Noise Modeling
```python title="sensor_noise_model.py"
import numpy as np

class SensorNoiseModel:
    def __init__(self):
        self.bias = 0.0
        self.random_walk = 0.0
        self.noise_density = 0.0

    def add_noise(self, true_value, dt, sensor_type='gyro'):
        """
        Add realistic sensor noise to true value
        """
        if sensor_type == 'gyro':
            # Typical IMU gyro parameters
            self.noise_density = 16e-3  # rad/s/sqrt(Hz)
            bias_instability = 13.2e-3  # rad/s
        elif sensor_type == 'accel':
            # Typical IMU accelerometer parameters
            self.noise_density = 80e-6  # g/sqrt(Hz)
            bias_instability = 50e-6  # g

        # Add white noise
        white_noise = np.random.normal(0, self.noise_density / np.sqrt(dt))

        # Add bias drift (random walk)
        self.random_walk += np.random.normal(0, bias_instability * np.sqrt(dt))

        # Add bias
        self.bias += np.random.normal(0, bias_instability * dt)

        return true_value + white_noise + self.random_walk + self.bias
```

### 2. Sensor Calibration
```python title="sensor_calibration.py"
class SensorCalibration:
    def __init__(self):
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.mag_bias = np.zeros(3)
        self.accel_scale = np.ones(3)
        self.gyro_scale = np.ones(3)

    def calibrate_imu(self, static_readings):
        """
        Calibrate IMU sensors assuming robot is in static position
        """
        # Accelerometer calibration (should read [0, 0, 9.81] when level)
        avg_acc = np.mean(static_readings['accel'], axis=0)
        self.accel_bias = [0, 0, 9.81] - avg_acc
        self.accel_scale = 9.81 / np.linalg.norm(avg_acc)

        # Gyroscope calibration (should read [0, 0, 0] when not rotating)
        avg_gyro = np.mean(static_readings['gyro'], axis=0)
        self.gyro_bias = -avg_gyro

    def apply_calibration(self, raw_data):
        """
        Apply calibration to raw sensor data
        """
        calibrated_data = {}

        # Apply accelerometer calibration
        if 'accel' in raw_data:
            calibrated_data['accel'] = (raw_data['accel'] + self.accel_bias) * self.accel_scale

        # Apply gyroscope calibration
        if 'gyro' in raw_data:
            calibrated_data['gyro'] = raw_data['gyro'] + self.gyro_bias

        return calibrated_data
```

## Exercises

1. Create a sensor fusion node that combines IMU and joint encoder data to estimate robot pose
2. Implement a simple particle filter for humanoid localization in a known map
3. Design a sensor validation framework that compares simulated and real sensor data

## Summary

Sensor simulation is crucial for humanoid robotics development, enabling safe testing of perception and control algorithms. By accurately modeling sensor characteristics including noise, range limitations, and update rates, we can create realistic simulation environments that closely match real-world performance. Effective sensor fusion techniques allow robots to combine multiple sensor modalities for robust perception and navigation capabilities.