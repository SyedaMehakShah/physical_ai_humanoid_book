---
sidebar_position: 2
---

# Real Robot Hardware Interfaces

## Learning Objectives
- Understand the hardware architecture of humanoid robots
- Learn to interface with different actuator types and control systems
- Master communication protocols for robot control
- Implement hardware abstraction layers for different platforms

## Intuition

Real robot hardware interfaces are like the nervous system that connects your brain (the AI controller) to your muscles (the actuators). Just as your brain sends signals through your spinal cord and peripheral nerves to control your muscles, your control software must communicate through specific hardware interfaces to control the robot's actuators. Understanding these interfaces is crucial for effective robot control and troubleshooting.

## Concept

Humanoid robot hardware typically consists of:
- **Actuators**: Motors that provide motion at each joint
- **Sensors**: Encoders, IMUs, force/torque sensors for state estimation
- **Controllers**: Local control electronics for each joint
- **Communication Bus**: Network connecting all components
- **Power System**: Batteries and power distribution

## Common Actuator Types

### 1. Servo Motors
```python title="servo_interface.py"
import serial
import time
import struct

class ServoController:
    def __init__(self, port='/dev/ttyUSB0', baudrate=1000000):
        self.port = port
        self.baudrate = baudrate
        self.connection = None
        self.connect()

    def connect(self):
        """Connect to servo bus"""
        try:
            self.connection = serial.Serial(self.port, self.baudrate, timeout=1)
            self.get_logger().info('Connected to servo bus')
        except Exception as e:
            self.get_logger().error(f'Failed to connect to servo bus: {e}')

    def set_servo_position(self, servo_id, position):
        """Set servo position (in encoder ticks or degrees)"""
        # Protocol-specific command (example for Dynamixel)
        # Instruction packet: [0xFF, 0xFF, ID, LENGTH, INSTRUCTION, PARAMETER, ..., CHECKSUM]
        instruction = 0x03  # Write instruction
        address = 0x1E      # Goal position address
        length = 4          # Instruction + address + 2 bytes data + checksum

        # Calculate checksum
        checksum = (~(servo_id + length + instruction + address + (position & 0xFF) + ((position >> 8) & 0xFF))) & 0xFF

        packet = [0xFF, 0xFF, servo_id, length, instruction, address, position & 0xFF, (position >> 8) & 0xFF, checksum]

        if self.connection:
            self.connection.write(bytes(packet))

    def get_servo_position(self, servo_id):
        """Get current servo position"""
        # Read position command
        instruction = 0x02  # Read instruction
        address = 0x24      # Present position address
        length = 4          # Instruction + address + data length + checksum
        data_length = 2     # 2 bytes for position

        checksum = (~(servo_id + length + instruction + address + data_length)) & 0xFF
        packet = [0xFF, 0xFF, servo_id, length, instruction, address, data_length, checksum]

        if self.connection:
            self.connection.write(bytes(packet))
            # Read response (simplified)
            response = self.connection.read(11)  # Expected response length
            if len(response) >= 8:
                pos_low = response[5]
                pos_high = response[6]
                position = (pos_high << 8) | pos_low
                return position
        return None

    def set_servo_torque(self, servo_id, enabled):
        """Enable/disable torque for servo"""
        instruction = 0x03  # Write instruction
        address = 0x18      # Torque enable address
        value = 1 if enabled else 0
        length = 4

        checksum = (~(servo_id + length + instruction + address + value)) & 0xFF
        packet = [0xFF, 0xFF, servo_id, length, instruction, address, value, checksum]

        if self.connection:
            self.connection.write(bytes(packet))
```

### 2. Brushless DC Motors with Encoders
```python title="bldc_interface.py"
import can
import struct

class BLDCController:
    def __init__(self, channel='can0', bitrate=1000000):
        self.bus = can.Bus(channel=channel, bustype='socketcan', bitrate=bitrate)
        self.joint_ids = [0x141, 0x142, 0x143, 0x144, 0x145, 0x146]  # CAN IDs for joints

    def enable_motor(self, joint_id):
        """Enable motor for specific joint"""
        # Send enable command (protocol-specific)
        enable_cmd = 0x01
        msg = can.Message(arbitration_id=joint_id, data=[enable_cmd], is_extended_id=True)
        self.bus.send(msg)

    def set_motor_torque(self, joint_id, torque):
        """Set torque command for motor"""
        # Convert torque to appropriate format
        torque_int = int(torque * 1000)  # Scale to integer (example)

        # Send torque command
        cmd = 0x02  # Torque control command
        data = struct.pack('<h', torque_int)  # Little-endian signed short
        msg = can.Message(arbitration_id=joint_id, data=[cmd] + list(data), is_extended_id=True)
        self.bus.send(msg)

    def set_motor_position(self, joint_id, position, velocity=0, kp=0, kd=0):
        """Set position with feedforward control"""
        # ODrive protocol example
        pos_int = int(position * 1000)  # Convert to counts
        vel_int = int(velocity * 1000)
        kp_int = int(kp * 1000)
        kd_int = int(kd * 1000)

        # Pack data for position control message
        data = struct.pack('<iiiii', pos_int, vel_int, 0, kp_int, kd_int)  # pos, vel, torque_ff, kp, kd
        msg = can.Message(arbitration_id=joint_id, data=data, is_extended_id=True)
        self.bus.send(msg)

    def get_motor_state(self, joint_id):
        """Get current motor state (position, velocity, effort)"""
        # Request state command
        req_cmd = 0x03
        msg = can.Message(arbitration_id=joint_id, data=[req_cmd], is_extended_id=True)
        self.bus.send(msg)

        # Listen for response
        try:
            response = self.bus.recv(timeout=0.1)
            if response and len(response.data) >= 12:  # Assuming 3x 4-byte values
                pos, vel, effort = struct.unpack('<fff', response.data[:12])
                return {'position': pos, 'velocity': vel, 'effort': effort}
        except can.CanError:
            pass
        return None
```

### 3. Hydraulic Actuators
```python title="hydraulic_interface.py"
import socket
import struct

class HydraulicController:
    def __init__(self, ip_address='192.168.1.10', port=502):
        self.ip_address = ip_address
        self.port = port
        self.socket = None
        self.connect()

    def connect(self):
        """Connect to hydraulic control system"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.ip_address, self.port))
        except Exception as e:
            print(f"Failed to connect to hydraulic controller: {e}")

    def set_hydraulic_pressure(self, actuator_id, pressure):
        """Set hydraulic pressure for specific actuator"""
        if not self.socket:
            return

        # Modbus TCP example
        transaction_id = 0x0001
        protocol_id = 0x0000
        length = 0x0006
        unit_id = 0x01
        function_code = 0x06  # Write single register
        register_address = 0x0000 + actuator_id * 2  # Each actuator uses 2 registers
        value = int(pressure * 100)  # Scale pressure value

        message = struct.pack('>HHHBBBH',
                             transaction_id, protocol_id, length,
                             unit_id, function_code, register_address >> 8, register_address & 0xFF)
        message += struct.pack('>H', value)

        self.socket.send(message)

    def get_hydraulic_state(self, actuator_id):
        """Get current hydraulic state"""
        if not self.socket:
            return None

        # Read holding register
        transaction_id = 0x0002
        protocol_id = 0x0000
        length = 0x0006
        unit_id = 0x01
        function_code = 0x03  # Read holding registers
        start_address = 0x0100 + actuator_id * 2  # State registers start at 0x0100
        register_count = 0x0003  # Read 3 registers (pos, vel, pressure)

        message = struct.pack('>HHHBBBH',
                             transaction_id, protocol_id, length,
                             unit_id, function_code, start_address >> 8, start_address & 0xFF)
        message += struct.pack('>H', register_count)

        self.socket.send(message)

        # Read response
        response = self.socket.recv(256)
        if len(response) >= 9:
            # Parse response
            data = struct.unpack('>HHHBBBHHH', response[:15])
            position = data[6] / 100.0
            velocity = data[7] / 100.0
            pressure = data[8] / 100.0
            return {'position': position, 'velocity': velocity, 'pressure': pressure}

        return None
```

## Communication Protocols

### 1. EtherCAT Interface
```python title="ethercat_interface.py"
import pysoem

class EtherCATController:
    def __init__(self, ifname='eth0'):
        self.ifname = ifname
        self.sdo_interfaces = []
        self.setup_ethercat()

    def setup_ethercat(self):
        """Setup EtherCAT communication"""
        try:
            # Initialize EtherCAT master
            self.master = pysoem.Master()
            self.master.open(self.ifname)

            # Configure slaves (motors)
            self.configure_slaves()

        except Exception as e:
            print(f"Failed to setup EtherCAT: {e}")

    def configure_slaves(self):
        """Configure EtherCAT slaves"""
        # Find and configure all slaves on the network
        self.master.config_init()

        # Configure each slave for cyclic sync (example)
        for i, slave in enumerate(self.master.slaves):
            # Set PDO mappings, sync managers, etc.
            slave.config_func = self.configure_slave_pdo
            slave.is_operational()

    def configure_slave_pdo(self, slave_pos):
        """Configure Process Data Objects for slave"""
        # Example configuration for a servo drive
        # TxPDO: actual position, velocity, status
        # RxPDO: target position, velocity, control word

        # This is highly device-specific
        pass

    def send_cyclic_data(self, joint_commands):
        """Send cyclic control data to all joints"""
        # Pack command data
        tx_data = []
        for cmd in joint_commands:
            # Pack position, velocity, torque commands
            pos_bytes = struct.pack('<i', int(cmd['position'] * 1000))
            vel_bytes = struct.pack('<i', int(cmd['velocity'] * 1000))
            tor_bytes = struct.pack('<h', int(cmd['torque'] * 1000))
            tx_data.extend(pos_bytes + vel_bytes + tor_bytes)

        # Send via EtherCAT
        self.master.send_processdata()
        self.master.receive_processdata(1000)  # 1ms timeout
```

### 2. CAN Bus Interface
```python title="can_interface.py"
import can
import struct
import threading

class CANRobotInterface:
    def __init__(self, channel='can0', bitrate=1000000):
        self.bus = can.Bus(channel=channel, bustype='socketcan', bitrate=bitrate)
        self.joint_controllers = {}  # Map of joint_id to controller config
        self.listeners = {}  # Message listeners
        self.command_buffer = []

        # Start message listener thread
        self.listener_thread = threading.Thread(target=self._listen_messages, daemon=True)
        self.listener_thread.start()

    def register_joint(self, joint_id, control_type='position'):
        """Register a joint controller"""
        self.joint_controllers[joint_id] = {
            'type': control_type,
            'last_position': 0.0,
            'last_velocity': 0.0,
            'last_effort': 0.0
        }

    def send_position_command(self, joint_id, position, velocity=0.0, effort=0.0):
        """Send position command to joint"""
        # Pack command data
        pos_int = int(position * 1000)  # Scale to integer
        vel_int = int(velocity * 1000)
        eff_int = int(effort * 1000)

        # Create CAN message
        data = struct.pack('<iii', pos_int, vel_int, eff_int)
        msg = can.Message(
            arbitration_id=0x180 + joint_id,  # Standard format: 0x180 + joint_id
            data=data,
            is_extended_id=True
        )

        self.bus.send(msg)

    def _listen_messages(self):
        """Listen for incoming CAN messages"""
        for msg in self.bus:
            # Parse message based on ID
            joint_id = msg.arbitration_id - 0x200  # State messages typically use 0x200 + joint_id

            if joint_id in self.joint_controllers and len(msg.data) >= 12:
                # Parse position, velocity, effort
                pos, vel, eff = struct.unpack('<fff', msg.data[:12])

                # Update joint state
                self.joint_controllers[joint_id]['last_position'] = pos
                self.joint_controllers[joint_id]['last_velocity'] = vel
                self.joint_controllers[joint_id]['last_effort'] = eff

    def get_joint_state(self, joint_id):
        """Get current state of a joint"""
        if joint_id in self.joint_controllers:
            return {
                'position': self.joint_controllers[joint_id]['last_position'],
                'velocity': self.joint_controllers[joint_id]['last_velocity'],
                'effort': self.joint_controllers[joint_id]['last_effort']
            }
        return None
```

## Hardware Abstraction Layer

Creating a unified interface for different hardware platforms:

```python title="hardware_abstraction.py"
from abc import ABC, abstractmethod
import numpy as np

class RobotHardwareInterface(ABC):
    """Abstract base class for robot hardware interfaces"""

    @abstractmethod
    def connect(self):
        """Connect to robot hardware"""
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from robot hardware"""
        pass

    @abstractmethod
    def get_joint_positions(self):
        """Get current joint positions"""
        pass

    @abstractmethod
    def get_joint_velocities(self):
        """Get current joint velocities"""
        pass

    @abstractmethod
    def get_joint_efforts(self):
        """Get current joint efforts"""
        pass

    @abstractmethod
    def set_joint_commands(self, commands):
        """Set joint commands (positions, velocities, or torques)"""
        pass

    @abstractmethod
    def enable_torque(self, enable=True):
        """Enable or disable joint torque"""
        pass

class DynamixelHardwareInterface(RobotHardwareInterface):
    """Hardware interface for Dynamixel-based robots"""

    def __init__(self, device_name='/dev/ttyUSB0', baudrate=1000000):
        self.device_name = device_name
        self.baudrate = baudrate
        self.servo_controller = ServoController(device_name, baudrate)
        self.joint_ids = [1, 2, 3, 4, 5, 6]  # Example joint IDs
        self.num_joints = len(self.joint_ids)

    def connect(self):
        """Connect to Dynamixel servos"""
        return self.servo_controller.connect()

    def disconnect(self):
        """Disconnect from Dynamixel servos"""
        if self.servo_controller.connection:
            self.servo_controller.connection.close()

    def get_joint_positions(self):
        """Get joint positions in radians"""
        positions = []
        for joint_id in self.joint_ids:
            raw_pos = self.servo_controller.get_servo_position(joint_id)
            if raw_pos is not None:
                # Convert from encoder ticks to radians (example conversion)
                pos_rad = raw_pos * (2 * np.pi / 4096)  # Assuming 12-bit encoder
                positions.append(pos_rad)
            else:
                positions.append(0.0)
        return np.array(positions)

    def get_joint_velocities(self):
        """Get joint velocities (approximated from position changes)"""
        # In practice, this would use velocity control mode or differentiate positions
        return np.zeros(self.num_joints)

    def get_joint_efforts(self):
        """Get joint efforts (torques)"""
        # Get current efforts from servos (if supported)
        return np.zeros(self.num_joints)

    def set_joint_commands(self, commands):
        """Set joint position commands"""
        for i, (joint_id, cmd) in enumerate(zip(self.joint_ids, commands)):
            # Convert radians to encoder ticks
            cmd_ticks = int(cmd * 4096 / (2 * np.pi))  # Assuming 12-bit encoder
            self.servo_controller.set_servo_position(joint_id, cmd_ticks)

    def enable_torque(self, enable=True):
        """Enable or disable torque for all joints"""
        for joint_id in self.joint_ids:
            self.servo_controller.set_servo_torque(joint_id, enable)

class RealRobotController:
    """Controller that works with any hardware interface"""

    def __init__(self, hardware_interface):
        self.hw_interface = hardware_interface
        self.current_state = {
            'position': np.zeros(6),
            'velocity': np.zeros(6),
            'effort': np.zeros(6)
        }
        self.desired_state = {
            'position': np.zeros(6),
            'velocity': np.zeros(6),
            'effort': np.zeros(6)
        }

    def connect(self):
        """Connect to hardware"""
        return self.hw_interface.connect()

    def update_state(self):
        """Update current robot state"""
        self.current_state['position'] = self.hw_interface.get_joint_positions()
        self.current_state['velocity'] = self.hw_interface.get_joint_velocities()
        self.current_state['effort'] = self.hw_interface.get_joint_efforts()

    def send_commands(self):
        """Send desired commands to hardware"""
        commands = self.compute_commands()
        self.hw_interface.set_joint_commands(commands)

    def compute_commands(self):
        """Compute control commands (placeholder for actual control logic)"""
        # This would contain the actual control algorithm
        # For now, return desired positions
        return self.desired_state['position']

    def set_desired_position(self, joint_positions):
        """Set desired joint positions"""
        self.desired_state['position'] = np.array(joint_positions)

# Example usage
def main():
    # Choose appropriate hardware interface based on robot type
    robot_type = "dynamixel"  # Could be "bldc", "hydraulic", etc.

    if robot_type == "dynamixel":
        hw_interface = DynamixelHardwareInterface()
    # elif robot_type == "bldc":
    #     hw_interface = BLDCInterface()
    # elif robot_type == "hydraulic":
    #     hw_interface = HydraulicInterface()

    controller = RealRobotController(hw_interface)

    # Connect to robot
    if controller.connect():
        print("Connected to robot hardware")

        # Main control loop
        import time
        for i in range(1000):  # 1000 control cycles
            controller.update_state()
            controller.send_commands()
            time.sleep(0.01)  # 10ms control cycle

        # Disable torque and disconnect
        hw_interface.enable_torque(False)
    else:
        print("Failed to connect to robot hardware")

if __name__ == "__main__":
    main()
```

## Power Management and Monitoring

```python title="power_management.py"
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Float32MultiArray
import time

class PowerManagementNode(Node):
    def __init__(self):
        super().__init__('power_management')

        # Power system parameters
        self.battery_voltage = 0.0
        self.battery_current = 0.0
        self.motor_currents = []
        self.power_threshold = 11.0  # Minimum voltage before shutdown
        self.current_limit = 10.0    # Maximum current per motor

        # Publishers and subscribers
        self.battery_sub = self.create_subscription(
            BatteryState, '/battery_state', self.battery_callback, 10)
        self.motor_current_sub = self.create_subscription(
            Float32MultiArray, '/motor_currents', self.motor_current_callback, 10)

        # Timer for power monitoring
        self.power_timer = self.create_timer(1.0, self.power_monitor)  # 1Hz

        self.get_logger().info('Power Management System initialized')

    def battery_callback(self, msg):
        """Process battery state"""
        self.battery_voltage = msg.voltage
        self.battery_current = msg.current

        # Check for low voltage
        if self.battery_voltage < self.power_threshold:
            self.get_logger().warn(f'Low battery voltage: {self.battery_voltage}V')
            self.initiate_safe_shutdown()

    def motor_current_callback(self, msg):
        """Process motor current readings"""
        self.motor_currents = list(msg.data)

        # Check for overcurrent
        for i, current in enumerate(self.motor_currents):
            if abs(current) > self.current_limit:
                self.get_logger().warn(f'Motor {i} overcurrent: {current}A')

    def power_monitor(self):
        """Monitor overall power consumption"""
        total_current = sum(abs(c) for c in self.motor_currents) + abs(self.battery_current)

        # Calculate remaining battery life (simplified)
        if hasattr(self, 'initial_charge') and hasattr(self, 'discharge_rate'):
            remaining_time = (self.initial_charge - self.get_battery_usage()) / self.discharge_rate
            if remaining_time < 300:  # Less than 5 minutes
                self.get_logger().warn(f'Estimated battery life: {remaining_time/60:.1f} minutes')

    def initiate_safe_shutdown(self):
        """Initiate safe shutdown procedures"""
        self.get_logger().error('Initiating safe shutdown due to low power')

        # Send zero commands to all joints
        zero_commands = Float32MultiArray()
        zero_commands.data = [0.0] * 6  # Assuming 6 joints
        # Publish to joint command topic to safely stop all motors

        # Log shutdown event
        self.get_logger().info('Motors commanded to zero position for safe shutdown')

    def get_battery_usage(self):
        """Calculate battery usage (simplified)"""
        # In practice, this would integrate current over time
        return 0.0
```

## Exercises

1. Implement a hardware abstraction layer for a specific humanoid robot platform
2. Create a communication interface for a different actuator type
3. Design a power management system for a humanoid robot

## Summary

Real robot hardware interfaces are the bridge between software control algorithms and physical actuation. Understanding different actuator types, communication protocols, and implementing proper hardware abstraction layers are essential for effective robot control. The diversity of hardware platforms requires flexible, well-designed interfaces that can adapt to different robot configurations while maintaining reliable communication and safety.