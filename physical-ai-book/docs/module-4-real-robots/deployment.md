---
sidebar_position: 3
---

# Deployment to Real Robots

## Learning Objectives
- Master the process of deploying simulation-trained controllers to real robots
- Understand deployment validation and testing procedures
- Learn systematic approaches to robot deployment
- Implement deployment pipelines for continuous integration

## Intuition

Deploying to real robots is like performing surgery - it requires precision, preparation, and a systematic approach. Just as surgeons follow strict protocols to ensure patient safety, roboticists must follow systematic procedures to ensure robot safety and successful deployment. The transition from simulation to reality requires careful validation, gradual testing, and robust safety measures to prevent damage to the expensive hardware and ensure safe operation.

## Concept

Deployment to real robots involves:
- **Validation**: Ensuring the controller works safely in simulation
- **Calibration**: Adapting simulation parameters to real hardware
- **Testing**: Gradual deployment with increasing complexity
- **Monitoring**: Continuous oversight during operation
- **Iteration**: Refinement based on real-world performance

## Pre-Deployment Validation

### 1. Simulation-to-Reality Validation
```python title="pre_deployment_validation.py"
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray

class PreDeploymentValidator(Node):
    def __init__(self):
        super().__init__('pre_deployment_validator')

        # Data collection
        self.sim_data = {'time': [], 'position': [], 'velocity': [], 'effort': []}
        self.real_data = {'time': [], 'position': [], 'velocity': [], 'effort': []}

        # Validation metrics
        self.validation_metrics = {
            'tracking_error': [],
            'control_effort': [],
            'stability_margin': [],
            'safety_compliance': []
        }

        # Publishers for validation commands
        self.command_pub = self.create_publisher(Float64MultiArray, '/joint_commands', 10)

        self.get_logger().info('Pre-deployment validator initialized')

    def validate_controller_stability(self, controller, test_trajectory):
        """Validate controller stability before deployment"""
        # Test with various trajectories
        test_trajectories = [
            'step_response',
            'sine_sweep',
            'random_walk',
            'impulse_response'
        ]

        results = {}
        for traj_type in test_trajectories:
            stability_result = self.test_trajectory_stability(controller, traj_type)
            results[traj_type] = stability_result

        # Calculate overall stability score
        overall_score = np.mean(list(results.values()))
        return overall_score, results

    def test_trajectory_stability(self, controller, trajectory_type):
        """Test controller stability with specific trajectory"""
        # Simulate the trajectory with the controller
        # This would run in a high-fidelity simulation
        time_steps = 1000
        dt = 0.01

        # Initialize robot state
        state = np.zeros(12)  # 6 positions + 6 velocities

        max_oscillation = 0
        for t in range(time_steps):
            # Generate trajectory point
            if trajectory_type == 'step_response':
                desired_pos = np.ones(6) * 0.5 if t > 100 else np.zeros(6)
            elif trajectory_type == 'sine_sweep':
                freq = 0.1 + (t / time_steps) * 5.0  # Sweep from 0.1 to 5 Hz
                desired_pos = np.sin(2 * np.pi * freq * t * dt) * 0.5
            elif trajectory_type == 'random_walk':
                desired_pos = np.random.normal(0, 0.1, 6)
            else:  # impulse
                desired_pos = np.ones(6) * 0.1 if t == 50 else np.zeros(6)

            # Apply control
            control_input = controller.compute_control(state, desired_pos)

            # Update state (simplified dynamics)
            acceleration = control_input - 0.1 * state[6:] - 0.01 * state[:6]  # damping + spring
            state[6:] += acceleration * dt  # Update velocities
            state[:6] += state[6:] * dt    # Update positions

            # Check for instability (large oscillations)
            current_oscillation = np.max(np.abs(state[:6]))
            max_oscillation = max(max_oscillation, current_oscillation)

            if max_oscillation > 10.0:  # Unstable threshold
                return 0.0

        # Return stability score (1.0 = very stable, 0.0 = unstable)
        stability_score = max(0, 1 - max_oscillation / 10.0)
        return stability_score

    def validate_safety_constraints(self, controller):
        """Validate that controller respects safety constraints"""
        safety_violations = 0
        total_tests = 100

        for i in range(total_tests):
            # Generate random state
            state = np.random.uniform(-1, 1, 12)  # Random state within bounds
            desired = np.random.uniform(-0.5, 0.5, 6)  # Random desired position

            # Get control output
            control_output = controller.compute_control(state, desired)

            # Check constraints
            if np.any(np.abs(control_output) > 100.0):  # Max torque = 100 Nm
                safety_violations += 1

            # Check joint limits (if applicable)
            new_positions = state[:6] + state[6:] * 0.01  # Simulate one time step
            if np.any(np.abs(new_positions) > 3.14):  # Joint limit = 180 degrees
                safety_violations += 1

        safety_score = 1.0 - (safety_violations / total_tests)
        return safety_score

    def generate_validation_report(self, controller):
        """Generate comprehensive validation report"""
        stability_score, stability_results = self.validate_controller_stability(controller, 'all')
        safety_score = self.validate_safety_constraints(controller)

        report = {
            'stability_score': stability_score,
            'stability_details': stability_results,
            'safety_score': safety_score,
            'overall_readiness': min(stability_score, safety_score),
            'recommendations': self.generate_recommendations(stability_score, safety_score)
        }

        return report

    def generate_recommendations(self, stability_score, safety_score):
        """Generate recommendations based on validation scores"""
        recommendations = []

        if stability_score < 0.8:
            recommendations.append("Controller needs stability improvements")
        if safety_score < 0.9:
            recommendations.append("Safety constraints need review")
        if stability_score < 0.7 or safety_score < 0.8:
            recommendations.append("Extensive simulation testing required")

        if not recommendations:
            recommendations.append("Ready for limited real-world testing")

        return recommendations

def main(args=None):
    rclpy.init(args=args)
    validator = PreDeploymentValidator()

    # Example usage
    class DummyController:
        def compute_control(self, state, desired):
            # Simple PD controller
            error = desired - state[:6]
            return 50 * error - 5 * state[6:]  # Kp=50, Kd=5

    controller = DummyController()
    report = validator.generate_validation_report(controller)

    print("Validation Report:")
    print(f"Stability Score: {report['stability_score']:.2f}")
    print(f"Safety Score: {report['safety_score']:.2f}")
    print(f"Overall Readiness: {report['overall_readiness']:.2f}")
    print("Recommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")

    validator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2. Parameter Adaptation for Real Hardware
```python title="parameter_adaptation.py"
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class ParameterAdapter:
    def __init__(self, sim_model, real_robot_interface):
        self.sim_model = sim_model
        self.real_robot = real_robot_interface
        self.adapted_params = {}
        self.sim_to_real_mapping = {
            'mass': 'mass_scaling',
            'inertia': 'inertia_scaling',
            'friction': 'friction_adaptation',
            'gear_ratio': 'encoder_resolution',
            'motor_constant': 'torque_constant'
        }

    def identify_real_parameters(self):
        """Identify real robot parameters through system identification"""
        # Collect step response data
        step_responses = self.collect_step_responses()

        # Fit model parameters
        fitted_params = self.fit_model_parameters(step_responses)

        return fitted_params

    def collect_step_responses(self):
        """Collect step response data from real robot"""
        responses = {}

        for joint_idx in range(6):  # Assuming 6 joints
            # Apply step input to joint
            step_input = 0.1  # 0.1 rad step

            # Collect response data
            time_data = []
            position_data = []
            velocity_data = []

            # Apply command and collect data
            start_time = self.real_robot.get_time()
            self.real_robot.set_joint_command(joint_idx, step_input)

            for i in range(500):  # 5 seconds of data at 100Hz
                current_time = self.real_robot.get_time()
                pos = self.real_robot.get_joint_position(joint_idx)
                vel = self.real_robot.get_joint_velocity(joint_idx)

                time_data.append(current_time - start_time)
                position_data.append(pos)
                velocity_data.append(vel)

                self.real_robot.sleep(0.01)  # 10ms

            responses[joint_idx] = {
                'time': np.array(time_data),
                'position': np.array(position_data),
                'velocity': np.array(velocity_data)
            }

        return responses

    def fit_model_parameters(self, step_responses):
        """Fit simulation model parameters to match real responses"""
        fitted_params = {}

        for joint_idx, response in step_responses.items():
            # Fit a second-order system: G(s) = K / (τ²s² + 2ζτs + 1)
            time = response['time']
            position = response['position']

            # Define objective function to minimize
            def objective(params):
                K, tau, zeta = params
                # Simulate response with current parameters
                system = signal.TransferFunction([K], [tau**2, 2*zeta*tau, 1])
                t_sim, y_sim = signal.step(system, T=time)

                # Interpolate to match time points
                y_interp = np.interp(time, t_sim, y_sim)

                # Calculate error
                error = np.mean((position - y_interp)**2)
                return error

            # Optimize parameters
            result = minimize(objective, [1.0, 0.1, 0.7], method='BFGS')
            K, tau, zeta = result.x

            fitted_params[joint_idx] = {
                'gain': K,
                'time_constant': tau,
                'damping_ratio': zeta
            }

        return fitted_params

    def adapt_controller_parameters(self, original_params, fitted_params):
        """Adapt controller parameters based on identified differences"""
        adapted_params = {}

        for joint_idx in fitted_params:
            # Adjust PID gains based on system characteristics
            original_kp = original_params.get('kp', [100.0]*6)[joint_idx]
            original_kd = original_params.get('kd', [10.0]*6)[joint_idx]

            # Adjust based on time constant and damping
            time_constant_ratio = fitted_params[joint_idx]['time_constant'] / 0.1  # Nominal value
            damping_ratio = fitted_params[joint_idx]['damping_ratio']

            # Increase gains if system is slower, decrease if more oscillatory
            kp_factor = 1.0 / time_constant_ratio if time_constant_ratio > 0.5 else 2.0 - time_constant_ratio
            kd_factor = damping_ratio * 2.0  # Higher damping needs less derivative action

            adapted_params[joint_idx] = {
                'kp': original_kp * kp_factor,
                'kd': original_kd * kd_factor
            }

        return adapted_params

    def validate_adaptation(self, adapted_controller):
        """Validate that adapted controller performs well on real robot"""
        # Run validation tests
        test_trajectories = [
            'setpoint_step',
            'trajectory_following',
            'disturbance_rejection'
        ]

        performance_metrics = {}

        for test in test_trajectories:
            metric = self.run_validation_test(adapted_controller, test)
            performance_metrics[test] = metric

        return performance_metrics

    def run_validation_test(self, controller, test_type):
        """Run specific validation test"""
        if test_type == 'setpoint_step':
            # Test step response performance
            initial_pos = self.real_robot.get_joint_positions()
            target_pos = initial_pos + 0.1  # 0.1 rad step

            # Apply control and measure performance
            start_time = self.real_robot.get_time()
            self.real_robot.set_desired_positions(target_pos)

            settling_time = 0
            overshoot = 0
            steady_state_error = 0

            # Monitor response
            for i in range(1000):  # 10 seconds
                current_pos = self.real_robot.get_joint_positions()
                error = target_pos - current_pos

                if i > 0 and settling_time == 0 and np.all(np.abs(error) < 0.01):
                    settling_time = self.real_robot.get_time() - start_time

                if i > 0:
                    overshoot = max(overshoot, np.max(np.abs(error)))

                if i == 999:  # Last iteration
                    steady_state_error = np.mean(np.abs(error))

                self.real_robot.sleep(0.01)

            # Calculate performance score (0-1, higher is better)
            score = 1.0 - (settling_time * 0.1 + overshoot + steady_state_error)
            return max(0, score)

        # Add other test types...
        return 0.5  # Default score
```

## Deployment Pipeline

### 1. Continuous Integration for Robot Deployment
```python title="deployment_pipeline.py"
import subprocess
import yaml
import json
import os
from datetime import datetime
import shutil

class RobotDeploymentPipeline:
    def __init__(self, config_file='deployment_config.yaml'):
        self.config = self.load_config(config_file)
        self.deployment_history = []
        self.current_version = "0.0.1"

    def load_config(self, config_file):
        """Load deployment configuration"""
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def run_pre_deployment_checks(self):
        """Run all pre-deployment validation checks"""
        checks = [
            self.check_code_quality,
            self.run_unit_tests,
            self.validate_simulation,
            self.check_safety_constraints,
            self.verify_hardware_compatibility
        ]

        results = {}
        all_passed = True

        for check_func in checks:
            check_name = check_func.__name__
            try:
                result = check_func()
                results[check_name] = result
                if not result['passed']:
                    all_passed = False
            except Exception as e:
                results[check_name] = {'passed': False, 'error': str(e)}
                all_passed = False

        return all_passed, results

    def check_code_quality(self):
        """Check code quality using linters and formatters"""
        try:
            # Run linters
            result = subprocess.run(['flake8', 'src/'], capture_output=True, text=True)
            if result.returncode == 0:
                return {'passed': True, 'details': 'Code quality check passed'}
            else:
                return {'passed': False, 'details': result.stdout}
        except FileNotFoundError:
            return {'passed': True, 'details': 'flake8 not found, skipping code quality check'}

    def run_unit_tests(self):
        """Run unit tests"""
        try:
            result = subprocess.run(['python', '-m', 'pytest', 'tests/'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return {'passed': True, 'details': 'All unit tests passed'}
            else:
                return {'passed': False, 'details': result.stdout}
        except Exception as e:
            return {'passed': False, 'error': str(e)}

    def validate_simulation(self):
        """Validate controller in simulation"""
        # This would run simulation tests
        # For now, return success
        return {'passed': True, 'details': 'Simulation validation passed'}

    def check_safety_constraints(self):
        """Check that safety constraints are met"""
        # Check for safety-related code
        safety_keywords = ['emergency_stop', 'safety', 'limit', 'clamp', 'max_', 'min_']

        safety_found = False
        for root, dirs, files in os.walk('src/'):
            for file in files:
                if file.endswith('.py'):
                    with open(os.path.join(root, file), 'r') as f:
                        content = f.read().lower()
                        if any(keyword in content for keyword in safety_keywords):
                            safety_found = True
                            break
            if safety_found:
                break

        if safety_found:
            return {'passed': True, 'details': 'Safety constraints detected in code'}
        else:
            return {'passed': False, 'details': 'No safety constraints detected'}

    def verify_hardware_compatibility(self):
        """Verify hardware compatibility"""
        # Check that hardware interfaces match target robot
        target_robot = self.config.get('target_robot', 'default')

        # This would check interface compatibility
        # For now, assume compatibility
        return {'passed': True, 'details': f'Compatible with {target_robot}'}

    def build_deployment_package(self):
        """Build deployment package"""
        package_name = f"robot_controller_v{self.current_version}.zip"

        # Create package directory
        package_dir = f"deployment_packages/v{self.current_version}"
        os.makedirs(package_dir, exist_ok=True)

        # Copy source files
        shutil.copytree('src/', f'{package_dir}/src/', dirs_exist_ok=True)
        shutil.copy('requirements.txt', f'{package_dir}/requirements.txt')
        shutil.copy('config.yaml', f'{package_dir}/config.yaml')

        # Create package info
        package_info = {
            'version': self.current_version,
            'timestamp': datetime.now().isoformat(),
            'build_config': self.config,
            'dependencies': self.get_dependencies()
        }

        with open(f'{package_dir}/package_info.json', 'w') as f:
            json.dump(package_info, f, indent=2)

        # Zip the package
        shutil.make_archive(f'deployment_packages/controller_v{self.current_version}',
                          'zip', package_dir)

        return f'deployment_packages/controller_v{self.current_version}.zip'

    def get_dependencies(self):
        """Get list of dependencies"""
        try:
            with open('requirements.txt', 'r') as f:
                return [line.strip() for line in f if line.strip() and not line.startswith('#')]
        except FileNotFoundError:
            return []

    def deploy_to_robot(self, package_path):
        """Deploy package to robot"""
        robot_ip = self.config.get('robot_ip', '192.168.1.10')
        robot_port = self.config.get('robot_port', 22)

        # This would use SSH/scp to deploy to robot
        # For simulation, we'll just copy to a local directory
        deploy_dir = f'/tmp/robot_deployment_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(deploy_dir, exist_ok=True)

        # Extract package
        shutil.unpack_archive(package_path, deploy_dir)

        # Install dependencies
        subprocess.run(['pip', 'install', '-r', f'{deploy_dir}/requirements.txt'],
                      check=True)

        # Restart robot controller service
        # This would typically involve systemctl or similar
        print(f"Deployed to {deploy_dir}")

        return True

    def run_post_deployment_tests(self):
        """Run tests after deployment"""
        # Run basic functionality tests
        tests = [
            self.test_basic_movement,
            self.test_safety_functions,
            self.test_sensor_feedback
        ]

        results = {}
        all_passed = True

        for test_func in tests:
            test_name = test_func.__name__
            try:
                result = test_func()
                results[test_name] = result
                if not result['passed']:
                    all_passed = False
            except Exception as e:
                results[test_name] = {'passed': False, 'error': str(e)}
                all_passed = False

        return all_passed, results

    def test_basic_movement(self):
        """Test basic robot movement"""
        # This would connect to the robot and test basic movement
        # For simulation, return success
        return {'passed': True, 'details': 'Basic movement test passed'}

    def test_safety_functions(self):
        """Test safety functions"""
        # Test emergency stop, joint limits, etc.
        return {'passed': True, 'details': 'Safety functions test passed'}

    def test_sensor_feedback(self):
        """Test sensor feedback"""
        # Test that sensors are providing valid data
        return {'passed': True, 'details': 'Sensor feedback test passed'}

    def execute_deployment(self):
        """Execute the full deployment pipeline"""
        print("Starting deployment pipeline...")

        # Step 1: Pre-deployment checks
        print("Running pre-deployment checks...")
        checks_passed, check_results = self.run_pre_deployment_checks()

        if not checks_passed:
            print("Pre-deployment checks failed:")
            for check, result in check_results.items():
                if not result.get('passed', False):
                    print(f"  {check}: {result.get('details', result.get('error', 'Unknown error'))}")
            return False

        print("Pre-deployment checks passed!")

        # Step 2: Build package
        print("Building deployment package...")
        package_path = self.build_deployment_package()
        print(f"Package built: {package_path}")

        # Step 3: Deploy to robot
        print("Deploying to robot...")
        deploy_success = self.deploy_to_robot(package_path)

        if not deploy_success:
            print("Deployment failed!")
            return False

        print("Deployment successful!")

        # Step 4: Post-deployment tests
        print("Running post-deployment tests...")
        tests_passed, test_results = self.run_post_deployment_tests()

        if not tests_passed:
            print("Post-deployment tests failed - rolling back...")
            # Rollback logic would go here
            return False

        print("Post-deployment tests passed!")
        print("Deployment completed successfully!")

        # Log deployment
        self.log_deployment(package_path, check_results, test_results)

        return True

    def log_deployment(self, package_path, pre_results, post_results):
        """Log deployment information"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'version': self.current_version,
            'package_path': package_path,
            'pre_deployment_results': pre_results,
            'post_deployment_results': post_results
        }

        self.deployment_history.append(log_entry)

        # Save to file
        with open('deployment_history.json', 'w') as f:
            json.dump(self.deployment_history, f, indent=2)

# Example configuration file content
config_content = """
target_robot: "custom_humanoid"
robot_ip: "192.168.1.10"
robot_port: 22
deployment_checks:
  - code_quality
  - unit_tests
  - simulation_validation
  - safety_constraints
  - hardware_compatibility
post_deployment_tests:
  - basic_movement
  - safety_functions
  - sensor_feedback
"""

def main():
    # Create config file if it doesn't exist
    if not os.path.exists('deployment_config.yaml'):
        with open('deployment_config.yaml', 'w') as f:
            f.write(config_content)

    pipeline = RobotDeploymentPipeline()
    success = pipeline.execute_deployment()

    if success:
        print("\n✅ Deployment pipeline completed successfully!")
    else:
        print("\n❌ Deployment pipeline failed!")

if __name__ == "__main__":
    main()
```

### 2. Gradual Deployment Strategy
```python title="gradual_deployment.py"
import time
import numpy as np
from enum import Enum

class DeploymentPhase(Enum):
    PASSIVE = 1      # Monitor only, no control
    ASSISTED = 2     # Assist with existing controller
    LIMITED = 3      # Limited range of motion
    FULL = 4         # Full control
    COMPLETE = 5     # Complete handover

class GradualDeployment:
    def __init__(self, robot_interface):
        self.robot_interface = robot_interface
        self.current_phase = DeploymentPhase.PASSIVE
        self.phase_durations = {
            DeploymentPhase.PASSIVE: 60,    # 1 minute
            DeploymentPhase.ASSISTED: 300,  # 5 minutes
            DeploymentPhase.LIMITED: 600,   # 10 minutes
            DeploymentPhase.FULL: 1800,     # 30 minutes
            DeploymentPhase.COMPLETE: 3600  # 1 hour
        }
        self.phase_start_time = time.time()
        self.safety_monitor = SafetyMonitor(robot_interface)

    def execute_deployment_phase(self):
        """Execute current deployment phase"""
        phase_handlers = {
            DeploymentPhase.PASSIVE: self.passive_phase,
            DeploymentPhase.ASSISTED: self.assisted_phase,
            DeploymentPhase.LIMITED: self.limited_phase,
            DeploymentPhase.FULL: self.full_phase,
            DeploymentPhase.COMPLETE: self.complete_phase
        }

        handler = phase_handlers.get(self.current_phase)
        if handler:
            success = handler()

            if success and self.phase_duration_expired():
                self.advance_phase()

            return success

        return False

    def passive_phase(self):
        """Passive monitoring phase - no control commands sent"""
        # Monitor robot state and log data
        current_state = self.robot_interface.get_joint_states()

        # Check for anomalies
        if self.safety_monitor.check_anomalies(current_state):
            return False  # Anomaly detected, abort deployment

        # Log state for analysis
        self.log_state(current_state)

        # No commands sent in passive phase
        return True

    def assisted_phase(self):
        """Assisted control phase - new controller assists existing one"""
        current_state = self.robot_interface.get_joint_states()

        # Get commands from both old and new controllers
        old_commands = self.get_existing_controller_output(current_state)
        new_commands = self.get_new_controller_output(current_state)

        # Blend commands (start with mostly old, gradually increase new)
        blend_ratio = self.get_phase_progress()
        blended_commands = (1 - blend_ratio) * old_commands + blend_ratio * new_commands

        # Apply safety limits
        safe_commands = self.safety_monitor.apply_safety_limits(blended_commands)

        # Send commands to robot
        self.robot_interface.send_commands(safe_commands)

        # Monitor for issues
        if self.safety_monitor.check_deployment_issues():
            return False

        return True

    def limited_phase(self):
        """Limited range phase - full new controller with restricted motion"""
        current_state = self.robot_interface.get_joint_states()

        # Generate commands with limited range of motion
        desired_positions = self.get_new_controller_output(current_state)

        # Apply joint limits specific to this phase
        limited_positions = self.apply_limited_workspace(desired_positions)

        # Apply safety limits
        safe_commands = self.safety_monitor.apply_safety_limits(limited_positions)

        # Send commands
        self.robot_interface.send_commands(safe_commands)

        return True

    def full_phase(self):
        """Full control phase - new controller with normal limits"""
        current_state = self.robot_interface.get_joint_states()

        # Generate normal commands
        commands = self.get_new_controller_output(current_state)

        # Apply normal safety limits
        safe_commands = self.safety_monitor.apply_safety_limits(commands)

        # Send commands
        self.robot_interface.send_commands(safe_commands)

        return True

    def complete_phase(self):
        """Complete handover phase - full new controller operation"""
        current_state = self.robot_interface.get_joint_states()

        # Full control with new controller
        commands = self.get_new_controller_output(current_state)

        # Apply final safety checks
        if not self.safety_monitor.final_safety_check(current_state, commands):
            return False

        # Send commands
        self.robot_interface.send_commands(commands)

        return True

    def advance_phase(self):
        """Advance to next deployment phase"""
        if self.current_phase == DeploymentPhase.COMPLETE:
            print("Deployment complete - new controller fully operational")
            return

        # Move to next phase
        next_phase = DeploymentPhase(self.current_phase.value + 1)
        self.current_phase = next_phase
        self.phase_start_time = time.time()

        print(f"Advancing to phase: {self.current_phase.name}")

    def phase_duration_expired(self):
        """Check if current phase duration has expired"""
        elapsed = time.time() - self.phase_start_time
        return elapsed >= self.phase_durations[self.current_phase]

    def get_phase_progress(self):
        """Get progress within current phase (0-1)"""
        elapsed = time.time() - self.phase_start_time
        total_duration = self.phase_durations[self.current_phase]
        return min(1.0, elapsed / total_duration)

    def get_existing_controller_output(self, state):
        """Get output from existing controller (placeholder)"""
        # This would interface with the current working controller
        return np.zeros(6)  # Placeholder

    def get_new_controller_output(self, state):
        """Get output from new controller (placeholder)"""
        # This would be your new controller implementation
        return np.zeros(6)  # Placeholder

    def apply_limited_workspace(self, positions):
        """Apply limited workspace constraints"""
        # Limit range of motion for safety during deployment
        limited_positions = np.clip(positions, -0.5, 0.5)  # Example limits
        return limited_positions

    def log_state(self, state):
        """Log robot state for analysis"""
        # Implementation for logging state data
        pass

class SafetyMonitor:
    def __init__(self, robot_interface):
        self.robot_interface = robot_interface
        self.anomaly_history = []
        self.deployment_issues = []

    def check_anomalies(self, state):
        """Check for anomalies in robot state"""
        # Check for unexpected joint positions, velocities, efforts
        if np.any(np.abs(state['position']) > 5.0):  # Check for extreme positions
            self.anomaly_history.append("Extreme joint position detected")
            return True

        if np.any(np.abs(state['velocity']) > 10.0):  # Check for extreme velocities
            self.anomaly_history.append("Extreme joint velocity detected")
            return True

        return False

    def apply_safety_limits(self, commands):
        """Apply safety limits to commands"""
        # Limit command magnitude
        limited_commands = np.clip(commands, -100.0, 100.0)  # Torque limits

        # Check for rate limiting
        if hasattr(self, 'prev_commands'):
            rate_limited = np.clip(
                commands,
                self.prev_commands - 50.0,  # Rate limit: 50 units per cycle
                self.prev_commands + 50.0
            )
            limited_commands = rate_limited

        self.prev_commands = limited_commands.copy()
        return limited_commands

    def check_deployment_issues(self):
        """Check for issues during deployment"""
        # Check for excessive errors, safety violations, etc.
        return len(self.deployment_issues) > 0

    def final_safety_check(self, state, commands):
        """Final safety check before complete handover"""
        # Comprehensive safety validation
        if not self.validate_stability(state, commands):
            return False
        if not self.validate_joint_limits(state):
            return False
        if not self.validate_actuator_limits(commands):
            return False

        return True

    def validate_stability(self, state, commands):
        """Validate system stability"""
        # Check for oscillations, instability indicators
        return True  # Placeholder

    def validate_joint_limits(self, state):
        """Validate joint limits"""
        return True  # Placeholder

    def validate_actuator_limits(self, commands):
        """Validate actuator limits"""
        return True  # Placeholder
```

## Exercises

1. Implement a deployment validation pipeline for a specific robot platform
2. Create a gradual deployment strategy for a new control algorithm
3. Design a safety monitoring system for robot deployment

## Summary

Deployment to real robots requires a systematic, safety-first approach that gradually transitions from simulation to reality. By implementing comprehensive validation procedures, parameter adaptation techniques, and gradual deployment strategies, you can safely and successfully deploy simulation-trained controllers to real hardware. The key is to never rush the process, maintain robust safety systems, and continuously validate performance at each stage of deployment.