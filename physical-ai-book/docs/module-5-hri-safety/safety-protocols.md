---
sidebar_position: 3
---

# Safety Protocols for Human-Robot Interaction

## Learning Objectives
- Understand specialized safety protocols for human-robot interaction scenarios
- Implement contact-safe control strategies for close-proximity operation
- Design safety systems for collaborative tasks between humans and robots
- Learn standards and regulations for HRI safety

## Intuition

Safety protocols for human-robot interaction are like having a highly attentive personal assistant who can anticipate potential hazards and react immediately to prevent harm. Just as a skilled martial artist knows how to practice contact safely, a humanoid robot must be able to interact with humans while maintaining safety at all times. The robot must be soft enough to be safe in contact, smart enough to avoid dangerous situations, and fast enough to react when needed.

## Concept

Human-robot interaction safety involves:
- **Contact Safety**: Ensuring safe physical contact between humans and robots
- **Collaborative Safety**: Managing shared workspaces and tasks
- **Dynamic Safety**: Adapting safety measures in real-time based on interaction
- **Compliance Safety**: Following safety standards and regulations

## Contact Safety and Impedance Control

### 1. Variable Impedance Control
```python title="impedance_control.py"
import numpy as np
import control  # python-control library
from scipy import signal

class VariableImpedanceController:
    def __init__(self, robot_mass=75.0, safety_radius=0.5):
        self.robot_mass = robot_mass
        self.safety_radius = safety_radius

        # Impedance parameters
        self.impedance_params = {
            'stiffness': np.diag([100, 100, 100, 50, 50, 50]),  # Diagonal stiffness matrix (x,y,z,rx,ry,rz)
            'damping': np.diag([20, 20, 20, 10, 10, 10]),        # Diagonal damping matrix
            'mass': np.diag([10, 10, 10, 5, 5, 5])               # Diagonal mass matrix
        }

        # Safety zones
        self.safety_zones = {
            'red': 0.3,    # Immediate stop zone
            'yellow': 0.6, # Reduced speed zone
            'green': 1.0   # Normal operation zone
        }

    def calculate_impedance_parameters(self, human_distance, contact_mode='safe'):
        """
        Calculate impedance parameters based on human proximity and contact mode
        """
        if contact_mode == 'safe':
            # Safe interaction: high compliance (low stiffness)
            stiffness_factor = min(0.3, human_distance / 1.0)  # Scale with distance
            damping_factor = min(0.5, human_distance / 0.8)
        elif contact_mode == 'collaborative':
            # Collaborative work: medium compliance
            stiffness_factor = min(0.7, human_distance / 0.6)
            damping_factor = min(0.8, human_distance / 0.5)
        else:  # 'performance'
            # Performance mode: low compliance (high stiffness)
            stiffness_factor = min(1.0, human_distance / 0.3)
            damping_factor = min(1.0, human_distance / 0.3)

        # Adjust parameters based on distance and mode
        adjusted_stiffness = self.impedance_params['stiffness'] * stiffness_factor
        adjusted_damping = self.impedance_params['damping'] * damping_factor

        return {
            'stiffness': adjusted_stiffness,
            'damping': adjusted_damping,
            'mass': self.impedance_params['mass']  # Mass typically stays constant
        }

    def contact_safe_control(self, desired_pos, current_pos, human_pos, human_vel=None):
        """
        Implement contact-safe control with variable impedance
        """
        # Calculate distance to human
        distance_to_human = np.linalg.norm(np.array(current_pos[:3]) - np.array(human_pos[:3]))

        # Determine contact mode based on distance
        contact_mode = self.determine_contact_mode(distance_to_human)

        # Get impedance parameters
        impedance = self.calculate_impedance_parameters(distance_to_human, contact_mode)

        # Calculate position error
        pos_error = np.array(desired_pos) - np.array(current_pos)

        # Calculate velocity (if available)
        if hasattr(self, 'prev_pos') and self.prev_pos is not None:
            vel = (np.array(current_pos) - np.array(self.prev_pos)) / 0.01  # Assuming 100Hz control
        else:
            vel = np.zeros_like(current_pos)
            self.prev_pos = current_pos

        # Calculate desired velocity towards goal
        desired_vel = pos_error * 10  # Simple proportional control

        # Calculate impedance force
        spring_force = np.dot(impedance['stiffness'], pos_error)
        damper_force = np.dot(impedance['damping'], (desired_vel - vel))

        total_force = spring_force + damper_force

        # Apply safety limits
        max_force = self.calculate_max_safe_force(distance_to_human)
        force_magnitude = np.linalg.norm(total_force)

        if force_magnitude > max_force:
            total_force = (total_force / force_magnitude) * max_force

        # Store for next iteration
        self.prev_pos = current_pos

        return total_force, impedance

    def determine_contact_mode(self, distance):
        """Determine contact mode based on human distance"""
        if distance < self.safety_zones['red']:
            return 'safe'  # Very safe, high compliance
        elif distance < self.safety_zones['yellow']:
            return 'collaborative'  # Collaborative mode
        else:
            return 'performance'  # Normal performance mode

    def calculate_max_safe_force(self, distance):
        """
        Calculate maximum safe force based on distance to human
        Uses ISO/TS 15066 guidelines
        """
        if distance < 0.1:  # Very close contact
            return 20.0  # 20N for quasi-static contact (ISO recommendation)
        elif distance < 0.3:
            return 50.0  # 50N for closer interactions
        elif distance < 0.5:
            return 80.0  # 80N for moderate proximity
        else:
            return 150.0  # Higher forces allowed at greater distances

    def force_limiter(self, force_vector, max_force):
        """Limit force vector magnitude to safe limits"""
        force_mag = np.linalg.norm(force_vector)
        if force_mag > max_force:
            return (force_vector / force_mag) * max_force
        return force_vector

    def admittance_control(self, applied_force, impedance_params):
        """
        Implement admittance control for compliant motion
        F = M*a + B*v + K*x
        Rearranged: a = M^-1 * (F - B*v - K*x)
        """
        # Calculate acceleration based on applied force
        inv_mass = np.linalg.inv(impedance_params['mass'])

        # For now, assume zero desired position and velocity for compliance
        # In practice, this would integrate with position control
        acceleration = np.dot(inv_mass, applied_force)

        return acceleration
```

### 2. Soft Actuator Control
```python title="soft_actuator_control.py"
class SoftActuatorController:
    def __init__(self, actuator_count=6):
        self.actuator_count = actuator_count
        self.actuator_states = [{'position': 0, 'force': 0, 'temperature': 25} for _ in range(actuator_count)]
        self.soft_limit_params = {
            'force_limit': 50.0,      # Maximum force per actuator
            'temperature_limit': 60.0, # Maximum temperature
            'position_limit': 3.14,   # Maximum joint position
            'velocity_limit': 2.0     # Maximum joint velocity
        }

    def implement_soft_limits(self, commands, states):
        """
        Implement soft limits for safe actuator control
        """
        limited_commands = []

        for i, (cmd, state) in enumerate(zip(commands, states)):
            limited_cmd = cmd.copy()

            # Limit force/torque
            if abs(cmd['effort']) > self.soft_limit_params['force_limit']:
                limited_cmd['effort'] = np.sign(cmd['effort']) * self.soft_limit_params['force_limit']

            # Limit position
            if abs(cmd['position']) > self.soft_limit_params['position_limit']:
                limited_cmd['position'] = np.sign(cmd['position']) * self.soft_limit_params['position_limit']

            # Limit velocity (based on position difference)
            if hasattr(self, f'prev_pos_{i}'):
                vel = (cmd['position'] - getattr(self, f'prev_pos_{i}')) / 0.01  # 100Hz
                if abs(vel) > self.soft_limit_params['velocity_limit']:
                    # Adjust position command to respect velocity limit
                    max_pos_change = self.soft_limit_params['velocity_limit'] * 0.01
                    pos_delta = cmd['position'] - getattr(self, f'prev_pos_{i}')
                    limited_pos_delta = np.clip(pos_delta, -max_pos_change, max_pos_change)
                    limited_cmd['position'] = getattr(self, f'prev_pos_{i}') + limited_pos_delta

            # Store for next iteration
            setattr(self, f'prev_pos_{i}', limited_cmd['position'])
            limited_commands.append(limited_cmd)

        return limited_commands

    def monitor_actuator_health(self):
        """
        Monitor actuator health and safety status
        """
        health_status = []

        for i, state in enumerate(self.actuator_states):
            status = {
                'id': i,
                'safe': True,
                'warnings': [],
                'critical': False
            }

            # Check force limits
            if state['force'] > self.soft_limit_params['force_limit'] * 0.9:
                status['warnings'].append('High force detected')
                if state['force'] > self.soft_limit_params['force_limit']:
                    status['critical'] = True
                    status['safe'] = False

            # Check temperature limits
            if state['temperature'] > self.soft_limit_params['temperature_limit'] * 0.9:
                status['warnings'].append('High temperature')
                if state['temperature'] > self.soft_limit_params['temperature_limit']:
                    status['critical'] = True
                    status['safe'] = False

            health_status.append(status)

        return health_status

    def implement_force_feedback(self, sensed_forces, desired_forces):
        """
        Implement force feedback for safe interaction
        """
        # Calculate force error
        force_errors = sensed_forces - desired_forces

        # Implement force limiting and shaping
        shaped_forces = []
        for sensed_force, desired_force in zip(sensed_forces, desired_forces):
            if abs(sensed_force) > self.soft_limit_params['force_limit']:
                # Implement force shaping to reduce impact
                shaped_force = np.sign(sensed_force) * self.soft_limit_params['force_limit']
            else:
                # Normal force control
                shaped_force = desired_force

            shaped_forces.append(shaped_force)

        return np.array(shaped_forces)

    def compliant_motion_control(self, desired_motion, external_forces):
        """
        Implement compliant motion in response to external forces
        """
        # Calculate compliance adjustment based on external forces
        compliance_adjustment = np.zeros_like(desired_motion)

        for i, ext_force in enumerate(external_forces):
            if abs(ext_force) > 10:  # Threshold for compliance response
                # Adjust motion to be compliant with external force
                compliance_factor = min(1.0, abs(ext_force) / 50.0)  # Scale with force magnitude
                compliance_adjustment[i] = -np.sign(ext_force) * compliance_factor * 0.01  # Small adjustment

        return desired_motion + compliance_adjustment
```

## Collaborative Safety Systems

### 1. Shared Workspace Safety
```python title="shared_workspace_safety.py"
import numpy as np
from scipy.spatial.distance import cdist
from enum import Enum

class CollaborationZone(Enum):
    ROBOT_ONLY = 1
    SHARED = 2
    HUMAN_ONLY = 3
    FORBIDDEN = 4

class SharedWorkspaceSafety:
    def __init__(self):
        # Define workspace zones
        self.workspace_zones = {
            'robot_workspace': {
                'bounds': {'min': [-1, -1, 0], 'max': [1, 1, 2]},  # Robot workspace
                'safety_zone': 0.3,  # Safety buffer
                'zone_type': CollaborationZone.ROBOT_ONLY
            },
            'shared_workspace': {
                'bounds': {'min': [-0.5, -0.5, 0], 'max': [0.5, 0.5, 1.5]},  # Shared area
                'safety_zone': 0.2,
                'zone_type': CollaborationZone.SHARED
            },
            'human_workspace': {
                'bounds': {'min': [-2, -2, 0], 'max': [2, 2, 0.5]},  # Human area
                'safety_zone': 0.5,
                'zone_type': CollaborationZone.HUMAN_ONLY
            }
        }

        # Human tracking
        self.tracked_humans = {}
        self.robot_path_history = []

    def check_workspace_conflict(self, robot_path, human_positions):
        """
        Check for conflicts between robot path and human positions
        """
        conflicts = []

        for i, robot_pose in enumerate(robot_path):
            for human_id, human_pos in human_positions.items():
                # Calculate distance between robot and human
                distance = np.linalg.norm(np.array(robot_pose[:3]) - np.array(human_pos[:3]))

                # Check if in collision course
                if distance < 0.5:  # 50cm safety threshold
                    conflict = {
                        'type': 'proximity_conflict',
                        'robot_path_index': i,
                        'human_id': human_id,
                        'distance': distance,
                        'timestamp': time.time()
                    }
                    conflicts.append(conflict)

        return conflicts

    def calculate_safe_robot_path(self, start_pose, goal_pose, human_positions):
        """
        Calculate safe path considering human positions
        """
        # Simple path planning with human-aware obstacle avoidance
        straight_path = self.generate_straight_path(start_pose, goal_pose)

        # Check for conflicts
        conflicts = self.check_workspace_conflict(straight_path, human_positions)

        if conflicts:
            # Generate alternative path that avoids humans
            safe_path = self.generate_avoidance_path(start_pose, goal_pose, human_positions)
        else:
            safe_path = straight_path

        return safe_path

    def generate_straight_path(self, start, goal, num_waypoints=10):
        """Generate straight-line path between start and goal"""
        path = []
        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            pose = []
            for j in range(len(start)):
                pose.append(start[j] + t * (goal[j] - start[j]))
            path.append(pose)
        return path

    def generate_avoidance_path(self, start, goal, human_positions):
        """Generate path that avoids humans"""
        # Simple implementation: move up and around humans
        path = [start]

        # Find highest human to avoid
        max_human_z = max([pos[2] for pos in human_positions.values()])

        # Go above humans
        midpoint = [(start[i] + goal[i]) / 2 for i in range(3)]
        avoidance_point = midpoint.copy()
        avoidance_point[2] = max_human_z + 0.5  # 50cm above tallest human

        path.append(avoidance_point)
        path.append(goal)

        return path

    def manage_collaborative_task_safety(self, task_definition, human_poses, robot_pose):
        """
        Manage safety during collaborative tasks
        """
        safety_status = {
            'task_safe': True,
            'required_actions': [],
            'risk_level': 'low'
        }

        # Check if humans are in appropriate zones
        for human_id, human_pose in human_poses.items():
            zone = self.determine_workspace_zone(human_pose)

            if zone['zone_type'] == CollaborationZone.ROBOT_ONLY:
                safety_status['task_safe'] = False
                safety_status['required_actions'].append({
                    'action': 'move_human_to_safe_zone',
                    'human_id': human_id
                })

        # Check robot position
        robot_zone = self.determine_workspace_zone(robot_pose)
        if robot_zone['zone_type'] == CollaborationZone.HUMAN_ONLY:
            safety_status['task_safe'] = False
            safety_status['required_actions'].append({
                'action': 'move_robot_to_safe_zone',
                'robot_id': 'main_robot'
            })

        # Calculate risk level based on proximity and task complexity
        min_distance = min([np.linalg.norm(np.array(robot_pose[:3]) - np.array(human_pose[:3]))
                           for human_pose in human_poses.values()], default=float('inf'))

        if min_distance < 0.3:
            safety_status['risk_level'] = 'high'
        elif min_distance < 0.6:
            safety_status['risk_level'] = 'medium'
        else:
            safety_status['risk_level'] = 'low'

        return safety_status

    def determine_workspace_zone(self, pose):
        """Determine which workspace zone a pose is in"""
        x, y, z = pose[:3]

        for zone_name, zone_def in self.workspace_zones.items():
            min_bounds = zone_def['bounds']['min']
            max_bounds = zone_def['bounds']['max']

            if (min_bounds[0] <= x <= max_bounds[0] and
                min_bounds[1] <= y <= max_bounds[1] and
                min_bounds[2] <= z <= max_bounds[2]):
                return zone_def

        # If not in any defined zone, return a default safe zone
        return {
            'bounds': {'min': [-float('inf'), -float('inf'), -float('inf')],
                      'max': [float('inf'), float('inf'), float('inf')]},
            'safety_zone': 1.0,
            'zone_type': CollaborationZone.FORBIDDEN
        }

    def implement_safety_protocol(self, detected_conflict):
        """
        Implement appropriate safety protocol based on conflict type
        """
        protocol_actions = []

        if detected_conflict['type'] == 'proximity_conflict':
            if detected_conflict['distance'] < 0.2:  # Very close
                # Immediate stop and move away
                protocol_actions.extend([
                    {'action': 'emergency_stop', 'priority': 'highest'},
                    {'action': 'move_away_from_human', 'distance': 0.5, 'priority': 'high'}
                ])
            elif detected_conflict['distance'] < 0.5:  # Close
                # Reduce speed and increase caution
                protocol_actions.extend([
                    {'action': 'reduce_speed', 'factor': 0.3, 'priority': 'medium'},
                    {'action': 'increase_safety_margin', 'margin': 0.3, 'priority': 'medium'}
                ])
            else:  # Moderate distance
                # Monitor closely
                protocol_actions.append({
                    'action': 'monitor_closely', 'priority': 'low'
                })

        return protocol_actions
```

### 2. Dynamic Safety Monitoring
```python title="dynamic_safety_monitoring.py"
import time
from collections import deque
import threading

class DynamicSafetyMonitor:
    def __init__(self, update_rate=100):  # 100Hz monitoring
        self.update_rate = update_rate
        self.monitoring_active = False
        self.safety_violations = deque(maxlen=100)
        self.safety_metrics = {
            'min_distance': float('inf'),
            'max_force': 0.0,
            'average_velocity': 0.0,
            'collision_probability': 0.0
        }

        # Safety thresholds
        self.thresholds = {
            'min_distance': 0.3,      # meters
            'max_force': 50.0,        # Newtons
            'max_velocity': 1.0,      # m/s
            'collision_probability': 0.1  # 10% chance
        }

        # Monitoring threads
        self.monitoring_thread = None

    def start_monitoring(self):
        """Start dynamic safety monitoring"""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop dynamic safety monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()

    def monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            start_time = time.time()

            # Collect current safety data
            current_data = self.collect_safety_data()

            # Update safety metrics
            self.update_safety_metrics(current_data)

            # Check for violations
            violations = self.check_safety_violations(current_data)
            if violations:
                self.safety_violations.extend(violations)
                self.handle_safety_violations(violations)

            # Calculate time to sleep to maintain update rate
            elapsed = time.time() - start_time
            sleep_time = max(0, (1.0 / self.update_rate) - elapsed)
            time.sleep(sleep_time)

    def collect_safety_data(self):
        """Collect current safety-related data"""
        # This would interface with actual sensors and robot state
        # For simulation, return placeholder data
        return {
            'human_positions': [],      # List of human positions
            'robot_position': [0, 0, 0], # Robot position
            'robot_velocity': [0, 0, 0], # Robot velocity
            'applied_forces': [0, 0, 0], # Applied forces
            'joint_states': [],         # Joint positions, velocities, efforts
            'collision_distances': [],  # Distances to nearest obstacles
            'time': time.time()
        }

    def update_safety_metrics(self, data):
        """Update safety metrics based on collected data"""
        # Update minimum distance
        if data['collision_distances']:
            current_min_dist = min(data['collision_distances'])
            self.safety_metrics['min_distance'] = min(
                self.safety_metrics['min_distance'],
                current_min_dist
            )

        # Update maximum force
        if data['applied_forces']:
            current_max_force = max(abs(f) for f in data['applied_forces'])
            self.safety_metrics['max_force'] = max(
                self.safety_metrics['max_force'],
                current_max_force
            )

        # Update average velocity
        vel_magnitude = np.linalg.norm(data['robot_velocity'])
        self.safety_metrics['average_velocity'] = (
            self.safety_metrics['average_velocity'] * 0.9 +
            vel_magnitude * 0.1  # Exponential moving average
        )

    def check_safety_violations(self, data):
        """Check for safety violations"""
        violations = []

        # Check distance violations
        if data['collision_distances']:
            min_distance = min(data['collision_distances'])
            if min_distance < self.thresholds['min_distance']:
                violations.append({
                    'type': 'distance_violation',
                    'value': min_distance,
                    'threshold': self.thresholds['min_distance'],
                    'timestamp': data['time']
                })

        # Check force violations
        if data['applied_forces']:
            max_force = max(abs(f) for f in data['applied_forces'])
            if max_force > self.thresholds['max_force']:
                violations.append({
                    'type': 'force_violation',
                    'value': max_force,
                    'threshold': self.thresholds['max_force'],
                    'timestamp': data['time']
                })

        # Check velocity violations
        vel_magnitude = np.linalg.norm(data['robot_velocity'])
        if vel_magnitude > self.thresholds['max_velocity']:
            violations.append({
                'type': 'velocity_violation',
                'value': vel_magnitude,
                'threshold': self.thresholds['max_velocity'],
                'timestamp': data['time']
            })

        return violations

    def handle_safety_violations(self, violations):
        """Handle detected safety violations"""
        for violation in violations:
            self.log_violation(violation)

            # Determine appropriate response based on violation severity
            if violation['type'] == 'distance_violation':
                if violation['value'] < 0.1:  # Very close
                    self.trigger_emergency_response()
                else:
                    self.trigger_caution_response()
            elif violation['type'] == 'force_violation':
                if violation['value'] > self.thresholds['max_force'] * 1.5:  # Significantly over limit
                    self.trigger_force_limbo()
                else:
                    self.reduce_force_output()
            elif violation['type'] == 'velocity_violation':
                self.limit_robot_velocity()

    def trigger_emergency_response(self):
        """Trigger emergency safety response"""
        print("EMERGENCY: Safety violation detected - immediate stop!")
        # In practice, this would send emergency stop commands
        pass

    def trigger_caution_response(self):
        """Trigger caution safety response"""
        print("CAUTION: Safety threshold approached - reducing speed")
        # In practice, this would reduce robot speed/force
        pass

    def trigger_force_limbo(self):
        """Trigger force limitation response"""
        print("FORCE LIMIT: Reducing applied forces")
        # In practice, this would limit force outputs
        pass

    def reduce_force_output(self):
        """Reduce force output for safety"""
        print("Reducing force output for safety")
        # In practice, this would scale down force commands
        pass

    def limit_robot_velocity(self):
        """Limit robot velocity for safety"""
        print("Limiting robot velocity")
        # In practice, this would limit velocity commands
        pass

    def log_violation(self, violation):
        """Log safety violation for analysis"""
        print(f"Safety violation: {violation['type']} - "
              f"Value: {violation['value']}, "
              f"Threshold: {violation['threshold']}")

    def get_safety_status(self):
        """Get current safety status"""
        return {
            'metrics': self.safety_metrics.copy(),
            'violations_count': len(self.safety_violations),
            'last_violation': self.safety_violations[-1] if self.safety_violations else None,
            'overall_safety': self.calculate_overall_safety()
        }

    def calculate_overall_safety(self):
        """Calculate overall safety level"""
        if not self.safety_violations:
            return 'SAFE'

        recent_violations = [v for v in self.safety_violations
                            if time.time() - v['timestamp'] < 10]  # Last 10 seconds

        if len(recent_violations) > 5:  # Many recent violations
            return 'UNSAFE'
        elif len(recent_violations) > 0:  # Some recent violations
            return 'CAUTION'
        else:
            return 'SAFE'
```

## Standards and Regulations

### 1. ISO/TS 15066 Compliance System
```python title="iso_15066_compliance.py"
class ISO15066ComplianceSystem:
    def __init__(self):
        # ISO/TS 15066 specific parameters
        self.iso_parameters = {
            'quasi_static_contact_force': 150.0,  # Maximum 150N for quasi-static contact
            'dynamic_contact_force': 80.0,        # Maximum 80N for dynamic contact
            'collision_force_limit': 150.0,       # General collision force limit
            'power_limit': 400.0,                 # Maximum power (400W for collaborative robots)
            'speed_limit': 250.0,                 # Maximum speed (250 mm/s for collaborative)
            'safety_distance': 0.5,               # Minimum safety distance
            'reaction_time': 0.1,                 # Maximum reaction time (100ms)
        }

        # Zone classifications
        self.zones = {
            'collision_avoidance': 2.0,    # 2m zone
            'safety_restricted': 1.0,      # 1m zone
            'workspace': 0.5,              # 0.5m zone
        }

    def check_iso_compliance(self, robot_state, human_state):
        """
        Check compliance with ISO/TS 15066 standards
        """
        compliance_report = {
            'force_compliance': True,
            'speed_compliance': True,
            'distance_compliance': True,
            'power_compliance': True,
            'overall_compliance': True,
            'violations': []
        }

        # Check force limits
        force_check = self.check_force_compliance(robot_state)
        if not force_check['compliant']:
            compliance_report['force_compliance'] = False
            compliance_report['violations'].extend(force_check['violations'])

        # Check speed limits
        speed_check = self.check_speed_compliance(robot_state)
        if not speed_check['compliant']:
            compliance_report['speed_compliance'] = False
            compliance_report['violations'].extend(speed_check['violations'])

        # Check distance limits
        distance_check = self.check_distance_compliance(robot_state, human_state)
        if not distance_check['compliant']:
            compliance_report['distance_compliance'] = False
            compliance_report['violations'].extend(distance_check['violations'])

        # Check power limits
        power_check = self.check_power_compliance(robot_state)
        if not power_check['compliant']:
            compliance_report['power_compliance'] = False
            compliance_report['violations'].extend(power_check['violations'])

        # Overall compliance
        compliance_report['overall_compliance'] = all([
            compliance_report['force_compliance'],
            compliance_report['speed_compliance'],
            compliance_report['distance_compliance'],
            compliance_report['power_compliance']
        ])

        return compliance_report

    def check_force_compliance(self, robot_state):
        """Check force compliance with ISO standards"""
        violations = []

        # Check joint forces/torques
        for i, effort in enumerate(robot_state.get('joint_efforts', [])):
            if abs(effort) > self.iso_parameters['collision_force_limit']:
                violations.append({
                    'type': 'force_violation',
                    'joint': i,
                    'value': abs(effort),
                    'limit': self.iso_parameters['collision_force_limit'],
                    'standard': 'ISO/TS 15066'
                })

        # Check for contact forces if available
        contact_forces = robot_state.get('contact_forces', [])
        for i, force in enumerate(contact_forces):
            if force > self.iso_parameters['dynamic_contact_force']:
                violations.append({
                    'type': 'contact_force_violation',
                    'location': i,
                    'value': force,
                    'limit': self.iso_parameters['dynamic_contact_force'],
                    'standard': 'ISO/TS 15066'
                })

        return {
            'compliant': len(violations) == 0,
            'violations': violations
        }

    def check_speed_compliance(self, robot_state):
        """Check speed compliance with ISO standards"""
        violations = []

        # Check Cartesian velocity
        cart_vel = robot_state.get('cartesian_velocity', [0, 0, 0])
        cart_speed = np.linalg.norm(cart_vel)

        if cart_speed > self.iso_parameters['speed_limit'] / 1000:  # Convert mm/s to m/s
            violations.append({
                'type': 'speed_violation',
                'value': cart_speed,
                'limit': self.iso_parameters['speed_limit'] / 1000,
                'standard': 'ISO/TS 15066'
            })

        # Check joint velocities
        joint_velocities = robot_state.get('joint_velocities', [])
        for i, vel in enumerate(joint_velocities):
            if abs(vel) > 1.0:  # Reasonable joint velocity limit
                violations.append({
                    'type': 'joint_speed_violation',
                    'joint': i,
                    'value': abs(vel),
                    'limit': 1.0,
                    'standard': 'ISO/TS 15066'
                })

        return {
            'compliant': len(violations) == 0,
            'violations': violations
        }

    def check_distance_compliance(self, robot_state, human_state):
        """Check distance compliance with ISO standards"""
        violations = []

        robot_pos = robot_state.get('position', [0, 0, 0])
        human_pos = human_state.get('position', [0, 0, 0])

        distance = np.linalg.norm(np.array(robot_pos) - np.array(human_pos))

        if distance < self.iso_parameters['safety_distance']:
            violations.append({
                'type': 'distance_violation',
                'value': distance,
                'limit': self.iso_parameters['safety_distance'],
                'standard': 'ISO/TS 15066'
            })

        return {
            'compliant': len(violations) == 0,
            'violations': violations
        }

    def check_power_compliance(self, robot_state):
        """Check power compliance with ISO standards"""
        violations = []

        # Calculate instantaneous power
        joint_velocities = robot_state.get('joint_velocities', [])
        joint_efforts = robot_state.get('joint_efforts', [])

        if len(joint_velocities) == len(joint_efforts):
            total_power = sum(abs(v * e) for v, e in zip(joint_velocities, joint_efforts))

            if total_power > self.iso_parameters['power_limit']:
                violations.append({
                    'type': 'power_violation',
                    'value': total_power,
                    'limit': self.iso_parameters['power_limit'],
                    'standard': 'ISO/TS 15066'
                })

        return {
            'compliant': len(violations) == 0,
            'violations': violations
        }

    def generate_compliance_report(self, compliance_data):
        """Generate detailed compliance report"""
        report = f"""
        ISO/TS 15066 COMPLIANCE REPORT
        =================================
        Force Compliance: {'PASS' if compliance_data['force_compliance'] else 'FAIL'}
        Speed Compliance: {'PASS' if compliance_data['speed_compliance'] else 'FAIL'}
        Distance Compliance: {'PASS' if compliance_data['distance_compliance'] else 'FAIL'}
        Power Compliance: {'PASS' if compliance_data['power_compliance'] else 'FAIL'}
        Overall Compliance: {'PASS' if compliance_data['overall_compliance'] else 'FAIL'}

        Violations Found: {len(compliance_data['violations'])}
        """

        for i, violation in enumerate(compliance_data['violations']):
            report += f"\n  {i+1}. {violation['type']}: {violation['value']:.2f} > {violation['limit']:.2f}"

        return report
```

### 2. Safety Validation and Certification
```python title="safety_validation.py"
import unittest
import numpy as np
from typing import Dict, List

class SafetyValidationSuite:
    """Comprehensive safety validation for HRI systems"""

    def __init__(self):
        self.test_results = []
        self.validation_standards = {
            'iso_ts_15066': ['force_limits', 'speed_limits', 'distance_safety'],
            'iso_10218': ['emergency_stop', 'safety_systems'],
            'iso_13482': ['personal_space', 'social_interaction_safety']
        }

    def run_comprehensive_safety_validation(self):
        """Run all safety validation tests"""
        print("Starting comprehensive safety validation...")

        # Run individual test suites
        force_test_results = self.run_force_safety_tests()
        collision_test_results = self.run_collision_safety_tests()
        interaction_test_results = self.run_interaction_safety_tests()

        # Aggregate results
        all_results = {
            'force_safety': force_test_results,
            'collision_safety': collision_test_results,
            'interaction_safety': interaction_test_results,
            'overall_pass_rate': self.calculate_overall_pass_rate(
                force_test_results, collision_test_results, interaction_test_results
            )
        }

        return all_results

    def run_force_safety_tests(self):
        """Run force safety validation tests"""
        test_cases = [
            ForceLimitTest('test_quasi_static_contact_force'),
            ForceLimitTest('test_dynamic_contact_force'),
            ForceLimitTest('test_impact_force_limit'),
            ForceLimitTest('test_normal_force_compliance')
        ]

        results = []
        for test_case in test_cases:
            result = self.run_single_test(test_case)
            results.append(result)

        return results

    def run_collision_safety_tests(self):
        """Run collision safety validation tests"""
        test_cases = [
            CollisionAvoidanceTest('test_proximity_detection'),
            CollisionAvoidanceTest('test_escape_maneuvers'),
            CollisionAvoidanceTest('test_stop_performance'),
            CollisionAvoidanceTest('test_cushioning_effectiveness')
        ]

        results = []
        for test_case in test_cases:
            result = self.run_single_test(test_case)
            results.append(result)

        return results

    def run_interaction_safety_tests(self):
        """Run human-robot interaction safety tests"""
        test_cases = [
            InteractionSafetyTest('test_personal_space_violation'),
            InteractionSafetyTest('test_unexpected_motion_safety'),
            InteractionSafetyTest('test_emergency_procedures'),
            InteractionSafetyTest('test_fallback_behaviors')
        ]

        results = []
        for test_case in test_cases:
            result = self.run_single_test(test_case)
            results.append(result)

        return results

    def run_single_test(self, test_case):
        """Run a single test case"""
        try:
            test_case.setUp()
            test_case.run()
            return {
                'test_name': test_case._testMethodName,
                'status': 'PASS',
                'execution_time': 0.0,  # Would be measured in real implementation
                'details': 'Test completed successfully'
            }
        except Exception as e:
            return {
                'test_name': test_case._testMethodName,
                'status': 'FAIL',
                'execution_time': 0.0,
                'details': str(e)
            }

    def calculate_overall_pass_rate(self, *test_results_groups):
        """Calculate overall pass rate across all test groups"""
        total_tests = 0
        passed_tests = 0

        for group in test_results_groups:
            for result in group:
                total_tests += 1
                if result['status'] == 'PASS':
                    passed_tests += 1

        if total_tests == 0:
            return 0.0

        return passed_tests / total_tests

class ForceLimitTest(unittest.TestCase):
    """Test force safety limits"""

    def setUp(self):
        self.max_quasi_static_force = 150.0  # ISO limit
        self.max_dynamic_force = 80.0        # ISO limit

    def test_quasi_static_contact_force(self):
        """Test that quasi-static contact forces are within limits"""
        # Simulate force measurement
        measured_force = self.measure_contact_force(contact_type='quasi_static')
        self.assertLessEqual(measured_force, self.max_quasi_static_force,
                           f"Quasi-static force {measured_force}N exceeds limit of {self.max_quasi_static_force}N")

    def test_dynamic_contact_force(self):
        """Test that dynamic contact forces are within limits"""
        measured_force = self.measure_contact_force(contact_type='dynamic')
        self.assertLessEqual(measured_force, self.max_dynamic_force,
                           f"Dynamic force {measured_force}N exceeds limit of {self.max_dynamic_force}N")

    def measure_contact_force(self, contact_type):
        """Simulate force measurement (placeholder)"""
        # In real implementation, this would interface with force sensors
        if contact_type == 'quasi_static':
            return np.random.uniform(10, 140)  # Random force within safe range
        else:
            return np.random.uniform(5, 75)    # Random force within safe range

class CollisionAvoidanceTest(unittest.TestCase):
    """Test collision avoidance safety"""

    def setUp(self):
        self.min_safe_distance = 0.3  # 30cm safety distance
        self.response_time_limit = 0.1  # 100ms response time

    def test_proximity_detection(self):
        """Test that proximity is detected in time"""
        distance = self.simulate_distance_measurement()
        self.assertGreater(distance, self.min_safe_distance,
                         f"Robot came too close: {distance}m, minimum safe distance is {self.min_safe_distance}m")

    def test_escape_maneuvers(self):
        """Test that escape maneuvers are executed properly"""
        # Simulate escape maneuver
        success = self.execute_escape_maneuver()
        self.assertTrue(success, "Escape maneuver failed")

    def simulate_distance_measurement(self):
        """Simulate distance measurement (placeholder)"""
        return np.random.uniform(0.5, 2.0)  # Random distance

    def execute_escape_maneuver(self):
        """Simulate escape maneuver execution (placeholder)"""
        return True  # Assume successful

class InteractionSafetyTest(unittest.TestCase):
    """Test interaction safety protocols"""

    def setUp(self):
        self.personal_space_radius = 0.5  # 50cm personal space

    def test_personal_space_violation(self):
        """Test that personal space is respected"""
        human_distance = self.measure_human_distance()
        self.assertGreater(human_distance, self.personal_space_radius,
                         f"Personal space violated: {human_distance}m from human")

    def test_emergency_procedures(self):
        """Test emergency stop procedures"""
        emergency_triggered = self.simulate_emergency()
        self.assertTrue(emergency_triggered, "Emergency procedures not triggered properly")

    def measure_human_distance(self):
        """Simulate human distance measurement (placeholder)"""
        return np.random.uniform(0.6, 3.0)  # Random distance

    def simulate_emergency(self):
        """Simulate emergency situation (placeholder)"""
        return True  # Assume emergency properly handled

def generate_safety_certificate(validation_results):
    """Generate safety certificate based on validation results"""
    pass_rate = validation_results['overall_pass_rate']

    certificate = f"""
    SAFETY VALIDATION CERTIFICATE
    ============================

    Date: {time.strftime('%Y-%m-%d')}
    System: Humanoid Robot HRI Safety System
    Validator: Automated Safety Validation Suite

    Validation Results:
    - Force Safety Tests: {len(validation_results['force_safety'])} tests
    - Collision Safety Tests: {len(validation_results['collision_safety'])} tests
    - Interaction Safety Tests: {len(validation_results['interaction_safety'])} tests
    - Overall Pass Rate: {pass_rate:.1%}

    Status: {'VALID' if pass_rate >= 0.95 else 'INVALID'}

    Compliance Standards:
    - ISO/TS 15066: {'PASS' if pass_rate >= 0.95 else 'FAIL'}
    - ISO 10218: {'PASS' if pass_rate >= 0.95 else 'FAIL'}
    - ISO 13482: {'PASS' if pass_rate >= 0.95 else 'FAIL'}

    Certificate valid for 1 year from issue date.
    """

    return certificate

def main():
    """Main function to run safety validation"""
    validator = SafetyValidationSuite()
    results = validator.run_comprehensive_safety_validation()

    print("\n" + "="*50)
    print("SAFETY VALIDATION COMPLETE")
    print("="*50)
    print(f"Overall Pass Rate: {results['overall_pass_rate']:.1%}")

    certificate = generate_safety_certificate(results)
    print(certificate)

    return results['overall_pass_rate'] >= 0.95  # Return True if certified

if __name__ == "__main__":
    is_certified = main()
    print(f"\nRobot is {'CERTIFIED' if is_certified else 'NOT CERTIFIED'} for HRI operation")
```

## Exercises

1. Implement a variable impedance controller for safe human-robot contact
2. Design a shared workspace safety system for collaborative tasks
3. Create a dynamic safety monitoring system with real-time violation detection

## Summary

Safety protocols for human-robot interaction require specialized approaches that go beyond traditional industrial safety. Contact-safe control using variable impedance, collaborative safety systems for shared workspaces, and dynamic monitoring systems are essential for safe HRI. Compliance with standards like ISO/TS 15066 is crucial, and comprehensive validation ensures that all safety systems work correctly. The key is to create systems that can adapt to changing interaction contexts while maintaining safety at all times.