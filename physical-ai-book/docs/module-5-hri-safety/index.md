---
sidebar_position: 1
---

# Module 5: Human-Robot Interaction & Safety

## Learning Objectives
- Understand principles of safe human-robot interaction for humanoid robots
- Learn safety frameworks and standards for physical human-robot interaction
- Explore communication modalities between humans and humanoid robots
- Understand ethical considerations in humanoid robot deployment
- Design AI agents that respect human safety and social norms

## Intuition

Human-Robot Interaction (HRI) is like teaching a robot to be a polite, safe, and helpful companion. Just as you would teach a child to interact appropriately with others, we must program humanoid robots to understand human social cues, respect personal space, and respond appropriately to human emotions and intentions. The robot must be able to work alongside humans safely while being helpful and not intimidating or dangerous.

## Concept

Human-Robot Interaction encompasses the design, development, and evaluation of robots that interact with humans. For humanoid robots, this includes:
- **Physical Safety**: Ensuring robots don't harm humans during interaction
- **Social Acceptance**: Making robots that humans feel comfortable around
- **Communication**: Enabling effective human-robot communication
- **Trust Building**: Creating interactions that build human confidence

## Safety Frameworks and Standards

### 1. ISO/TS 15066 Guidelines
ISO/TS 15066 provides guidelines for safe human-robot collaboration:

```python title="iso_15066_compliance.py"
import numpy as np
from enum import Enum

class InteractionMode(Enum):
    COEXISTENCE = 1      # Robot and human in same space, separated by safeguarding
    SEQUENTIAL = 2       # Robot and human use space sequentially
    PARALLEL = 3         # Robot and human work simultaneously in same space
    COLLABORATION = 4    # Direct interaction between robot and human

class ISO15066Compliance:
    def __init__(self):
        self.safety_zones = {
            'collision_avoidance': 2.0,  # 2m minimum distance
            'safety_restricted': 1.0,    # 1m safety zone
            'workspace': 0.5             # 0.5m workspace boundary
        }

        self.speed_limits = {
            'collision_avoidance': 0.5,  # m/s when humans nearby
            'safety_restricted': 0.2,    # m/s in safety zone
            'workspace': 0.05            # m/s in workspace
        }

        self.force_limits = {
            'quasi_static': 150.0,       # 150N for quasi-static contact
            'dynamic': 80.0              # 80N for dynamic contact
        }

    def calculate_safe_distance(self, robot_speed, human_reaction_time=0.8):
        """
        Calculate minimum safe distance based on robot speed and human reaction time
        """
        # Distance = speed * reaction_time + stopping_distance
        stopping_distance = (robot_speed ** 2) / (2 * 1.0)  # Assuming 1.0 m/s^2 deceleration
        safe_distance = robot_speed * human_reaction_time + stopping_distance
        return max(safe_distance, 0.5)  # Minimum 0.5m safety distance

    def assess_interaction_risk(self, human_pos, robot_pos, robot_velocity):
        """
        Assess risk level based on human-robot proximity and robot motion
        """
        distance = np.linalg.norm(np.array(human_pos) - np.array(robot_pos))
        speed = np.linalg.norm(np.array(robot_velocity))

        # Calculate risk based on distance and speed
        if distance < self.safety_zones['workspace']:
            risk_level = 'HIGH'
            interaction_mode = InteractionMode.COLLABORATION
        elif distance < self.safety_zones['safety_restricted']:
            risk_level = 'MEDIUM'
            interaction_mode = InteractionMode.PARALLEL
        elif distance < self.safety_zones['collision_avoidance']:
            risk_level = 'LOW'
            interaction_mode = InteractionMode.SEQUENTIAL
        else:
            risk_level = 'NONE'
            interaction_mode = InteractionMode.COXISTENCE

        return {
            'risk_level': risk_level,
            'interaction_mode': interaction_mode,
            'distance': distance,
            'safe_distance': self.calculate_safe_distance(speed)
        }

    def apply_safety_limits(self, desired_velocity, human_distance):
        """
        Apply speed limits based on human proximity
        """
        if human_distance <= self.safety_zones['workspace']:
            max_speed = self.speed_limits['workspace']
        elif human_distance <= self.safety_zones['safety_restricted']:
            max_speed = self.speed_limits['safety_restricted']
        elif human_distance <= self.safety_zones['collision_avoidance']:
            max_speed = self.speed_limits['collision_avoidance']
        else:
            max_speed = 1.0  # Normal operation

        # Limit velocity magnitude
        speed = np.linalg.norm(desired_velocity)
        if speed > max_speed:
            limited_velocity = (desired_velocity / speed) * max_speed
        else:
            limited_velocity = desired_velocity

        return limited_velocity
```

### 2. Risk Assessment and Mitigation
```python title="risk_assessment.py"
class HRIRiskAssessment:
    def __init__(self):
        self.risk_matrix = {
            'probability': {
                'rare': 1,
                'unlikely': 2,
                'possible': 3,
                'likely': 4,
                'almost_certain': 5
            },
            'severity': {
                'negligible': 1,
                'minor': 2,
                'moderate': 3,
                'major': 4,
                'catastrophic': 5
            }
        }

    def calculate_risk_level(self, probability, severity):
        """
        Calculate risk level using risk matrix
        """
        prob_score = self.risk_matrix['probability'][probability]
        sev_score = self.risk_matrix['severity'][severity]
        risk_score = prob_score * sev_score

        if risk_score <= 4:
            return 'LOW'
        elif risk_score <= 9:
            return 'MEDIUM'
        elif risk_score <= 16:
            return 'HIGH'
        else:
            return 'EXTREME'

    def assess_collision_risk(self, robot_state, human_state):
        """
        Assess collision risk between robot and human
        """
        # Calculate time to collision
        relative_velocity = np.array(robot_state['velocity']) - np.array(human_state['velocity'])
        relative_position = np.array(robot_state['position']) - np.array(human_state['position'])

        if np.linalg.norm(relative_velocity) == 0:
            time_to_collision = float('inf')
        else:
            time_to_collision = np.dot(relative_position, relative_velocity) / np.dot(relative_velocity, relative_velocity)

        distance = np.linalg.norm(relative_position)

        # Determine probability based on time to collision
        if time_to_collision < 1.0:  # Less than 1 second
            probability = 'almost_certain'
        elif time_to_collision < 3.0:
            probability = 'likely'
        elif time_to_collision < 10.0:
            probability = 'possible'
        else:
            probability = 'unlikely'

        # Determine severity based on robot size, speed, and mass
        robot_kinetic_energy = 0.5 * robot_state['mass'] * np.dot(relative_velocity, relative_velocity)
        if robot_kinetic_energy > 100:  # High energy impact
            severity = 'major'
        elif robot_kinetic_energy > 20:  # Moderate energy impact
            severity = 'moderate'
        else:
            severity = 'minor'

        risk_level = self.calculate_risk_level(probability, severity)

        return {
            'risk_level': risk_level,
            'probability': probability,
            'severity': severity,
            'time_to_collision': time_to_collision,
            'kinetic_energy': robot_kinetic_energy
        }
```

## Communication Modalities

### 1. Multimodal Communication System
```python title="multimodal_communication.py"
import speech_recognition as sr
import pyttsx3
import cv2
import numpy as np
from enum import Enum

class CommunicationMode(Enum):
    SPEECH = 1
    GESTURE = 2
    FACIAL_EXPRESSION = 3
    PROXEMICS = 4
    HAPTICS = 5

class MultimodalCommunication:
    def __init__(self):
        # Initialize speech recognition
        self.speech_recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Initialize text-to-speech
        self.tts_engine = pyttsx3.init()
        self.setup_tts()

        # Initialize computer vision for gesture recognition
        self.gesture_detector = self.setup_gesture_detector()

        # Initialize facial expression system
        self.facial_expressions = {
            'neutral': [0, 0, 0, 0, 0],
            'happy': [1, 1, 0, 1, 0],
            'sad': [0, 0, 1, 0, 1],
            'surprised': [1, 0, 0, 1, 1],
            'attentive': [0, 1, 0, 1, 0]
        }

    def setup_tts(self):
        """Setup text-to-speech parameters"""
        voices = self.tts_engine.getProperty('voices')
        self.tts_engine.setProperty('rate', 150)  # Words per minute
        self.tts_engine.setProperty('volume', 0.8)

    def setup_gesture_detector(self):
        """Setup gesture detection system"""
        # This would use OpenCV or other computer vision libraries
        # For now, return a placeholder
        return None

    def listen_for_speech(self, timeout=5):
        """
        Listen for speech and convert to text
        """
        try:
            with self.microphone as source:
                self.speech_recognizer.adjust_for_ambient_noise(source)
                audio = self.speech_recognizer.listen(source, timeout=timeout)

            text = self.speech_recognizer.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            return None

    def speak_text(self, text):
        """
        Convert text to speech and speak
        """
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def recognize_gesture(self, image):
        """
        Recognize human gestures from image
        """
        # In a real implementation, this would use computer vision
        # For now, return a placeholder
        return "wave"  # Example gesture

    def set_facial_expression(self, expression_name):
        """
        Set robot's facial expression
        """
        if expression_name in self.facial_expressions:
            expression_data = self.facial_expressions[expression_name]
            # This would control the robot's facial servos/display
            print(f"Setting facial expression to: {expression_name}")
            return True
        return False

    def process_human_intention(self, speech=None, gesture=None, facial=None):
        """
        Process multiple communication modalities to understand human intention
        """
        intentions = []

        if speech:
            # Process speech for commands/intentions
            speech_intention = self.process_speech(speech)
            intentions.append(('speech', speech_intention))

        if gesture:
            # Process gesture for commands/intentions
            gesture_intention = self.process_gesture(gesture)
            intentions.append(('gesture', gesture_intention))

        if facial:
            # Process facial expression for emotional context
            emotional_context = self.process_facial_expression(facial)
            intentions.append(('facial', emotional_context))

        # Fuse multiple modalities to determine most likely intention
        fused_intention = self.fuse_modalities(intentions)
        return fused_intention

    def process_speech(self, speech):
        """Process speech for commands or questions"""
        # Simple keyword-based processing
        if 'hello' in speech.lower() or 'hi' in speech.lower():
            return {'type': 'greeting', 'content': speech}
        elif 'help' in speech.lower():
            return {'type': 'request_help', 'content': speech}
        elif 'stop' in speech.lower():
            return {'type': 'emergency_stop', 'content': speech}
        else:
            return {'type': 'general', 'content': speech}

    def process_gesture(self, gesture):
        """Process gesture for commands or intentions"""
        gesture_map = {
            'wave': 'greeting',
            'point': 'request_attention',
            'clap': 'acknowledgment',
            'beckon': 'request_approach',
            'stop_sign': 'emergency_stop'
        }

        return gesture_map.get(gesture, 'unknown')

    def process_facial_expression(self, facial_expr):
        """Process human facial expression for emotional context"""
        # This would use facial recognition to detect emotions
        return {'emotion': 'neutral', 'confidence': 0.8}

    def fuse_modalities(self, intentions):
        """Fuse multiple modalities to determine overall intention"""
        # Simple fusion - return most confident intention
        if len(intentions) == 1:
            return intentions[0][1]

        # For multiple modalities, look for consistency
        speech_intent = None
        gesture_intent = None

        for modality, intent in intentions:
            if modality == 'speech':
                speech_intent = intent
            elif modality == 'gesture':
                gesture_intent = intent

        # If both speech and gesture agree, confidence is high
        if speech_intent and gesture_intent and speech_intent['type'] == gesture_intent:
            speech_intent['confidence'] = 'high'

        return speech_intent if speech_intent else (intentions[0][1] if intentions else None)
```

### 2. Proxemics and Personal Space
```python title="proxemics_management.py"
class ProxemicsManager:
    def __init__(self):
        # Define proxemic zones according to Hall's model
        self.proxemic_zones = {
            'intimate': (0.0, 0.45),    # 0-18 inches
            'personal': (0.45, 1.2),    # 18 inches - 4 feet
            'social': (1.2, 3.6),       # 4-12 feet
            'public': (3.6, float('inf'))  # 12+ feet
        }

        # Robot behavior in each zone
        self.zone_behaviors = {
            'intimate': {
                'speed_limit': 0.05,
                'approach_angle': 'frontal',
                'eye_contact': 'direct',
                'gesture_level': 'subtle'
            },
            'personal': {
                'speed_limit': 0.1,
                'approach_angle': 'angular',
                'eye_contact': 'occasional',
                'gesture_level': 'moderate'
            },
            'social': {
                'speed_limit': 0.2,
                'approach_angle': 'lateral',
                'eye_contact': 'intermittent',
                'gesture_level': 'normal'
            },
            'public': {
                'speed_limit': 0.5,
                'approach_angle': 'variable',
                'eye_contact': 'minimal',
                'gesture_level': 'expressive'
            }
        }

    def determine_proxemic_zone(self, distance):
        """
        Determine which proxemic zone a given distance falls into
        """
        for zone, (min_dist, max_dist) in self.proxemic_zones.items():
            if min_dist <= distance < max_dist:
                return zone
        return 'public'  # Default to public if outside defined zones

    def get_behavior_advice(self, human_distance):
        """
        Get behavioral advice based on human distance
        """
        zone = self.determine_proxemic_zone(human_distance)
        return self.zone_behaviors[zone]

    def calculate_approach_trajectory(self, robot_pos, human_pos, human_orientation):
        """
        Calculate safe approach trajectory respecting proxemics
        """
        # Calculate vector from robot to human
        approach_vector = np.array(human_pos) - np.array(robot_pos)
        distance = np.linalg.norm(approach_vector)

        zone = self.determine_proxemic_zone(distance)

        if zone == 'intimate':
            # Approach from front, very slowly
            approach_direction = self.calculate_frontal_approach(human_orientation)
        elif zone == 'personal':
            # Approach from angle, moderate speed
            approach_direction = self.calculate_angular_approach(human_orientation)
        else:
            # Normal approach
            approach_direction = approach_vector / distance

        # Limit speed based on zone
        behavior = self.get_behavior_advice(distance)
        max_speed = behavior['speed_limit']

        return {
            'direction': approach_direction,
            'max_speed': max_speed,
            'zone': zone
        }

    def calculate_frontal_approach(self, human_orientation):
        """Calculate frontal approach direction"""
        # Approach from front (same direction as human facing)
        return np.array([np.cos(human_orientation), np.sin(human_orientation)])

    def calculate_angular_approach(self, human_orientation):
        """Calculate angular approach direction"""
        # Approach from 45-degree angle to human facing
        approach_angle = human_orientation + np.pi/4  # 45 degrees
        return np.array([np.cos(approach_angle), np.sin(approach_angle)])
```

## Ethical Considerations and Trust Building

### 1. Ethical Decision Making
```python title="ethical_decision_making.py"
class EthicalDecisionMaker:
    def __init__(self):
        # Define ethical principles
        self.ethical_principles = {
            'beneficence': 1.0,      # Do good
            'non_malfeasance': 1.0,  # Do no harm
            'autonomy': 0.8,         # Respect human autonomy
            'justice': 0.7,          # Fair treatment
            'veracity': 0.9          # Tell the truth
        }

        # Define ethical rules
        self.ethical_rules = [
            "Never cause physical harm to humans",
            "Always respect human privacy",
            "Do not deceive humans unnecessarily",
            "Respect human dignity and autonomy",
            "Be transparent about capabilities and limitations"
        ]

    def evaluate_action_ethics(self, action, context):
        """
        Evaluate an action based on ethical principles
        """
        ethical_score = 0.0
        total_weight = 0.0

        # Evaluate against each principle
        for principle, weight in self.ethical_principles.items():
            score = self.evaluate_principle(action, context, principle)
            ethical_score += score * weight
            total_weight += weight

        final_score = ethical_score / total_weight if total_weight > 0 else 0.0
        return final_score

    def evaluate_principle(self, action, context, principle):
        """
        Evaluate how well an action adheres to a specific ethical principle
        """
        if principle == 'non_malfeasance':
            # Check for potential harm
            if self.would_cause_harm(action, context):
                return 0.0
            else:
                return 1.0
        elif principle == 'autonomy':
            # Check if action respects human autonomy
            if self.respects_autonomy(action, context):
                return 1.0
            else:
                return 0.0
        elif principle == 'veracity':
            # Check if action involves deception
            if self.involves_deception(action, context):
                return 0.0
            else:
                return 1.0
        else:
            # Default evaluation
            return 0.5

    def would_cause_harm(self, action, context):
        """Check if action would cause harm"""
        # Check for physical harm
        if 'move_to' in action:
            target_pos = action['move_to']
            humans_nearby = context.get('humans_nearby', [])

            for human in humans_nearby:
                dist = np.linalg.norm(np.array(target_pos) - np.array(human['position']))
                if dist < 0.5:  # Less than 50cm
                    return True

        # Check for other types of harm (emotional, psychological, etc.)
        return False

    def respects_autonomy(self, action, context):
        """Check if action respects human autonomy"""
        # Check if action overrides human commands
        human_command = context.get('human_command', None)
        if human_command and action.get('override', False):
            return False

        return True

    def involves_deception(self, action, context):
        """Check if action involves deception"""
        # Check if robot is pretending to have capabilities it doesn't have
        if action.get('pretend_capable', False):
            return True

        # Check if robot is hiding its limitations
        if action.get('hide_limitation', False):
            return True

        return False

    def make_ethical_decision(self, possible_actions, context):
        """
        Choose the most ethical action from possibilities
        """
        best_action = None
        best_score = -1.0

        for action in possible_actions:
            score = self.evaluate_action_ethics(action, context)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action, best_score
```

### 2. Trust Building Mechanisms
```python title="trust_building.py"
class TrustBuilder:
    def __init__(self):
        self.trust_scores = {}  # Per-human trust scores
        self.trust_history = {}  # Interaction history
        self.transparency_level = 0.7  # Default transparency

    def initialize_human_trust(self, human_id):
        """Initialize trust for a new human"""
        self.trust_scores[human_id] = {
            'overall': 0.5,  # Start neutral
            'reliability': 0.5,
            'competence': 0.5,
            'benevolence': 0.5,
            'integrity': 0.5
        }
        self.trust_history[human_id] = []

    def update_trust_after_interaction(self, human_id, interaction_outcome):
        """
        Update trust based on interaction outcome
        """
        if human_id not in self.trust_scores:
            self.initialize_human_trust(human_id)

        trust_update = self.calculate_trust_update(interaction_outcome)

        # Update trust components
        self.trust_scores[human_id]['reliability'] = self.update_component(
            self.trust_scores[human_id]['reliability'],
            trust_update.get('reliability', 0)
        )

        self.trust_scores[human_id]['competence'] = self.update_component(
            self.trust_scores[human_id]['competence'],
            trust_update.get('competence', 0)
        )

        # Calculate overall trust as weighted average
        weights = {'reliability': 0.4, 'competence': 0.3, 'benevolence': 0.2, 'integrity': 0.1}
        overall = sum(
            self.trust_scores[human_id][comp] * weights[comp]
            for comp in weights
        )
        self.trust_scores[human_id]['overall'] = overall

        # Log interaction
        self.trust_history[human_id].append({
            'outcome': interaction_outcome,
            'update': trust_update,
            'timestamp': time.time()
        })

    def calculate_trust_update(self, outcome):
        """Calculate trust update based on interaction outcome"""
        updates = {}

        if outcome['success']:
            updates['reliability'] = 0.1
            updates['competence'] = 0.1
        else:
            updates['reliability'] = -0.1
            updates['competence'] = -0.05

        # Safety-related outcomes affect benevolence
        if outcome.get('safety_violation', False):
            updates['benevolence'] = -0.2
        elif outcome.get('safety_positive', False):
            updates['benevolence'] = 0.1

        # Honesty affects integrity
        if outcome.get('honest_interaction', False):
            updates['integrity'] = 0.1

        return updates

    def update_component(self, current_value, update):
        """Update a trust component with bounded values [0, 1]"""
        new_value = current_value + update
        return max(0.0, min(1.0, new_value))

    def get_trust_level(self, human_id):
        """Get overall trust level for a human"""
        if human_id not in self.trust_scores:
            return 0.5  # Neutral trust for unknown humans

        return self.trust_scores[human_id]['overall']

    def adjust_behavior_for_trust(self, human_id, base_behavior):
        """
        Adjust robot behavior based on trust level
        """
        trust_level = self.get_trust_level(human_id)

        adjusted_behavior = base_behavior.copy()

        # Adjust based on trust level
        if trust_level < 0.3:
            # Low trust: Be very careful and transparent
            adjusted_behavior['speed'] *= 0.5  # Move slowly
            adjusted_behavior['transparency'] = 1.0  # Be fully transparent
            adjusted_behavior['initiative'] = 0.1  # Take little initiative
        elif trust_level < 0.6:
            # Medium trust: Cautious but helpful
            adjusted_behavior['speed'] *= 0.7
            adjusted_behavior['transparency'] = 0.8
            adjusted_behavior['initiative'] = 0.3
        elif trust_level < 0.8:
            # High trust: Normal helpful behavior
            adjusted_behavior['transparency'] = 0.7
            adjusted_behavior['initiative'] = 0.6
        else:
            # Very high trust: More proactive
            adjusted_behavior['initiative'] = 0.8
            adjusted_behavior['proactivity'] = 0.7

        return adjusted_behavior

    def build_trust_through_transparency(self, human_id, explanation):
        """
        Build trust by being transparent about actions
        """
        if human_id not in self.trust_scores:
            self.initialize_human_trust(human_id)

        # Provide explanation and update trust
        self.speak_explanation(explanation)
        self.update_trust_after_interaction(human_id, {
            'success': True,
            'honest_interaction': True
        })

    def speak_explanation(self, explanation):
        """Speak an explanation to the human"""
        # This would use the communication system to explain actions
        print(f"Robot explanation: {explanation}")
```

## Exercises

1. Implement a multimodal communication system for human-robot interaction
2. Design a proxemics-aware navigation system for safe human-robot coexistence
3. Create an ethical decision-making framework for humanoid robots

## Summary

Human-Robot Interaction & Safety is a critical aspect of humanoid robotics that requires careful consideration of physical safety, social acceptance, communication modalities, and ethical principles. By implementing comprehensive safety frameworks, multimodal communication systems, and trust-building mechanisms, we can create humanoid robots that interact safely and effectively with humans. The key is to balance functionality with safety while respecting human dignity and autonomy.