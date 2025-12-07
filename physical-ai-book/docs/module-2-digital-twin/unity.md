---
sidebar_position: 3
---

# Unity for High-Fidelity Rendering

## Learning Objectives
- Understand Unity's role in robotics visualization and simulation
- Learn to create high-fidelity humanoid robot models in Unity
- Integrate Unity with ROS 2 for robot control and visualization
- Design human-robot interaction scenarios in Unity

## Intuition

Unity is like a Hollywood movie studio for robots, where you can create photorealistic environments and interactions. While Gazebo focuses on accurate physics simulation, Unity excels at creating visually stunning, immersive environments that can be used for human-robot interaction studies, training scenarios, and high-fidelity visualization. Think of Unity as the place where you make your robot "look real" and interact naturally with humans in realistic settings.

## Concept

Unity is a real-time 3D development platform that provides high-quality rendering, physics simulation, and interactive capabilities. For robotics, Unity can be used for:
- High-fidelity visualization of robot behaviors
- Human-robot interaction studies
- Virtual reality training environments
- Photorealistic simulation for computer vision
- User interface development for robot operators

## Unity-ROS Integration

Unity can be integrated with ROS 2 through several methods:

### 1. Unity Robotics Hub
The Unity Robotics Hub provides tools and packages for connecting Unity with ROS 2:

```csharp title="RobotController.cs"
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;

public class RobotController : MonoBehaviour
{
    ROSConnection ros;
    string robotTopic = "unity_robot_position";

    // Robot joint angles
    public float[] jointAngles = new float[6];

    // Start is called before the first frame update
    void Start()
    {
        // Get ROS connection static instance
        ros = ROSConnection.instance;

        // Subscribe to ROS topic
        ros.Subscribe<JointStateMsg>(robotTopic, OnJointStateReceived);
    }

    // Called when a joint state message is received
    void OnJointStateReceived(JointStateMsg jointState)
    {
        // Update robot visualization based on joint states
        for (int i = 0; i < jointState.position.Count && i < jointAngles.Length; i++)
        {
            jointAngles[i] = (float)jointState.position[i];
        }

        // Apply joint angles to Unity robot model
        UpdateRobotModel();
    }

    void UpdateRobotModel()
    {
        // Update the Unity robot model based on joint angles
        // This would typically involve rotating joint GameObjects
        Debug.Log("Updating robot model with joint angles");
    }

    // Update is called once per frame
    void Update()
    {
        // Send current robot position to ROS
        var robotPosition = new TransformMsg();
        robotPosition.translation = new Vector3Msg(transform.position.x,
                                                   transform.position.y,
                                                   transform.position.z);
        robotPosition.rotation = new QuaternionMsg(transform.rotation.x,
                                                  transform.rotation.y,
                                                  transform.rotation.z,
                                                  transform.rotation.w);

        ros.Send("unity_robot_pose", robotPosition);
    }
}
```

### 2. ROS# Bridge
Another popular option for Unity-ROS integration:

```csharp title="RosBridgeClient.cs"
using RosSharp.RosBridgeClient;
using RosSharp.Messages.Geometry;

public class RosBridgeClient : MonoBehaviour
{
    public string rosBridgeServerUrl = "ws://192.168.1.1:9090";
    private RosSocket rosSocket;

    void Start()
    {
        WebSocketNativeClient webSocket = new WebSocketNativeClient(rosBridgeServerUrl);
        rosSocket = new RosSocket(webSocket);

        // Subscribe to topics
        rosSocket.Subscribe<OdometryMsg>("/odom", ReceiveOdometry);
    }

    void ReceiveOdometry(OdometryMsg odom)
    {
        // Update Unity robot position based on odometry
        transform.position = new Vector3((float)odom.pose.pose.position.x,
                                        (float)odom.pose.pose.position.y,
                                        (float)odom.pose.pose.position.z);
    }
}
```

## Creating Humanoid Robot Models in Unity

### 1. Importing Robot Models
Unity can import robot models in various formats:
- **FBX**: Most common format for 3D models
- **URDF**: Through specialized importers
- **STL/OBJ**: For basic geometry

### 2. Setting up Humanoid Animation
For humanoid robots, Unity's animation system can be used to create realistic movements:

```csharp title="HumanoidAnimation.cs"
using UnityEngine;

[RequireComponent(typeof(Animator))]
public class HumanoidAnimation : MonoBehaviour
{
    private Animator animator;
    private HumanoidController controller;

    void Start()
    {
        animator = GetComponent<Animator>();
        controller = GetComponent<HumanoidController>();
    }

    void Update()
    {
        // Update animation parameters based on robot state
        if (controller != null)
        {
            animator.SetFloat("Speed", controller.GetLinearVelocity());
            animator.SetFloat("Turn", controller.GetAngularVelocity());
            animator.SetBool("IsWalking", controller.IsMoving());
            animator.SetBool("IsBalancing", controller.IsBalancing());
        }
    }
}
```

## Unity Scene Setup for Human-Robot Interaction

Here's an example of setting up a scene for human-robot interaction:

```csharp title="HRIInteraction.cs"
using UnityEngine;
using UnityEngine.XR;

public class HRIInteraction : MonoBehaviour
{
    public GameObject robot;
    public GameObject humanAvatar;
    public float interactionDistance = 2.0f;
    public LayerMask interactionLayer;

    void Update()
    {
        // Check for human-robot proximity
        float distance = Vector3.Distance(robot.transform.position,
                                         humanAvatar.transform.position);

        if (distance <= interactionDistance)
        {
            // Trigger interaction behaviors
            OnHumanNearby();
        }
        else
        {
            // Return to default behaviors
            OnHumanDeparted();
        }
    }

    void OnHumanNearby()
    {
        // Robot looks at human, adjusts posture, etc.
        Vector3 lookDirection = humanAvatar.transform.position - robot.transform.position;
        lookDirection.y = 0; // Keep head level
        robot.transform.rotation = Quaternion.LookRotation(lookDirection);

        // Trigger social behaviors
        Debug.Log("Human detected, initiating interaction");
    }

    void OnHumanDeparted()
    {
        // Return to default behavior
        Debug.Log("Human departed, returning to default state");
    }
}
```

## Best Practices for Unity Robotics

1. **Performance Optimization**:
   - Use Level of Detail (LOD) for complex models
   - Optimize draw calls and batching
   - Use occlusion culling for large environments

2. **Realistic Physics**:
   - Use appropriate physics materials
   - Configure collision detection properly
   - Match Unity physics to real-world parameters

3. **Visual Quality**:
   - Use physically-based materials (PBR)
   - Configure lighting to match real environments
   - Add post-processing effects for realism

4. **Integration Considerations**:
   - Maintain consistent coordinate systems
   - Handle time synchronization between Unity and ROS
   - Manage data types and units properly

## Setting up a Basic Unity Scene

Here's a step-by-step setup for a basic humanoid robot scene:

1. **Create the Scene Structure**:
   - Main Camera with appropriate viewing angle
   - Lighting (directional light for sun simulation)
   - Ground plane with realistic materials
   - Robot model imported and positioned

2. **Configure Physics**:
   - Set up collision meshes for robot
   - Configure physics materials for different surfaces
   - Set up joints if using Unity physics

3. **Add Visualization Elements**:
   - UI elements for displaying robot status
   - Debug visualization for sensor data
   - Path visualization for navigation

## Exercises

1. Create a Unity scene with a humanoid robot model and basic environment
2. Implement ROS communication to control the robot's position in Unity
3. Design a simple human-robot interaction scenario with basic behaviors

## Summary

Unity provides the high-fidelity visualization capabilities essential for advanced robotics applications. When combined with ROS 2 integration, Unity becomes a powerful platform for creating immersive, realistic environments for human-robot interaction studies, training scenarios, and visualization of complex robot behaviors. Understanding Unity's capabilities and integration patterns is crucial for developing comprehensive digital twin systems for humanoid robots.