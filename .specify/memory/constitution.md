# Constitution: Physical AI & Humanoid Robotics Book

## 1. QUALITY & STYLE

### Technical Writing Standards
- All content must use clear, student-friendly language with short paragraphs (2-4 sentences)
- Use descriptive headings that clearly indicate content purpose
- Include practical examples for every concept introduced
- Provide visual diagrams for complex system interactions
- Use consistent terminology throughout all modules

### Robotics Terminology Standards
- Use standard ROS 2 terminology: nodes, topics, services, actions, parameters
- Define all technical terms when first introduced
- Use "digital twin" consistently to refer to simulation environments
- Use "URDF" for Unified Robot Description Format
- Use "rclpy" specifically for Python ROS 2 client library
- Use "Gazebo" for physics simulation, "Unity" for high-fidelity rendering

### Code Example Standards
- All code examples must be runnable without modification
- Keep examples minimal and focused on the concept being taught
- Include comprehensive comments explaining each code section
- Use Python + rclpy for all ROS 2 examples (avoid C++)
- Every code example must include expected output or behavior
- Use consistent code formatting and style across all examples

### Learning Structure Requirements
- Every chapter/module must begin with clear learning outcomes
- Every chapter/module must end with a concise recap
- Include hands-on exercises with solution guides
- Provide troubleshooting sections for common errors

## 2. SPEC-DRIVEN WORKFLOW

### Required Process Flow
This project MUST always follow the Spec-Kit Plus phases in sequence:
- `/sp.specify` → `/sp.clarify` → `/sp.plan` → `/sp.tasks` → `/sp.implement`

### No Vibe Coding Policy
- Absolutely no direct coding or documentation changes without updating specs first
- All features must be specified before implementation
- Any deviation from the spec-driven workflow requires updating this Constitution

### Artifact Requirements
Every feature or module must include:
- A spec at `.specify/specs/<feature>/spec.md` with user stories and requirements
- A plan at `.specify/specs/<feature>/plan.md` with technical architecture
- Tasks at `.specify/specs/<feature>/tasks.md` with testable implementation steps

### Spec-First Change Policy
- Any major change to content or structure must update the spec first
- Minor corrections may be made directly but must be reflected in specs immediately
- Content and code must always align with the latest spec

## 3. BOOK SCOPE & BOUNDARIES

### Target Technology Stack
- **Book Platform**: Docusaurus with MDX support
- **Hosting**: GitHub Pages
- **Robotics Framework**: ROS 2 (Humble Hawksbill or later)
- **Simulation**: Gazebo for physics, Unity for high-fidelity rendering
- **AI Integration**: Python-based AI agents connecting to ROS 2
- **Advanced Simulation**: NVIDIA Isaac for advanced scenarios

### In Scope
- Conceptual understanding of Physical AI and embodied intelligence
- Hands-on humanoid simulation using Gazebo and Unity
- Basic humanoid control flows using ROS 2
- Bridges between AI agents and ROS 2 systems
- Digital twin concepts and implementation
- Safety considerations in human-robot interaction
- Sim-to-real transfer techniques

### Out of Scope
- Complete hardware bring-up procedures for specific robot platforms
- Production-grade safety certification processes
- Comprehensive C++ ROS 2 development (focus on Python)
- Deep 3D modeling or asset creation in Unity
- Advanced real-time control optimization
- Complete robot maintenance and repair procedures

## 4. ROBUSTNESS & TESTING

### Example Project Requirements
Every example project must include:
- Basic validation steps with expected results
- Clear instructions for setup and execution
- Troubleshooting notes for common student errors
- Performance benchmarks where applicable
- Compatibility notes for different ROS 2 distributions

### Testing Standards
- All ROS 2 examples must be tested with actual ROS 2 installation
- Simulation examples must work in both Gazebo and Unity environments
- Code examples must be validated for syntax and functionality
- Include automated validation where possible

### Documentation Requirements
- All automation scripts must be documented in the book
- CI/CD processes must be explained for student understanding
- Troubleshooting guides must be comprehensive and accessible
- Error messages must be explained with resolution steps

## 5. ACCESSIBILITY & PEDAGOGY

### Target Audience Profile
- Students with intermediate Python programming skills
- Students familiar with AI concepts (LLMs, agents, tools)
- Learners new to robotics concepts and physical systems
- Students with basic understanding of system architecture

### Learning Progression Structure
Every module must follow this sequence:
- **Intuition**: Relatable analogy or real-world connection
- **Concept**: Clear definition and explanation
- **Diagram**: Visual representation of the concept
- **Minimal Example**: Simple, runnable code or procedure
- **Exercises**: Hands-on practice with solution guides

### Content Organization
- Break complex topics into small, digestible units
- Avoid large code dumps; instead, build incrementally
- Use consistent formatting for examples and explanations
- Include frequent check-ins and self-assessment questions
- Provide multiple learning pathways for different paces

## 6. GOVERNANCE

### Constitution Authority
- This Constitution is the single source of truth for quality and process standards
- All future specs, plans, tasks, and implementations must conform to these principles
- If any artifact violates these principles, it must be updated to comply

### Amendment Process
- Changes to this Constitution require explicit update via `/sp.specify` process
- All team members must acknowledge Constitution updates
- Constitution changes must be reflected in all existing artifacts

### Iterative Development Policy
- Favor short, iterative features over large all-in-one implementations
- Each feature should deliver value independently
- Maintain working state at all times during development
- Use feature flags for incomplete functionality rather than broken states

### Quality Assurance
- Regular reviews must ensure compliance with all Constitution requirements
- Peer review process must verify adherence to standards
- Automated checks should validate code examples and links
- Student feedback should inform continuous improvement

**Version**: 1.0.0 | **Ratified**: 2025-12-06 | **Last Amended**: 2025-12-06
