# Physical AI & Humanoid Robotics Book

This book is designed for students who have completed earlier AI quarters (LLMs, agents, tools) and are familiar with Python and basic AI concepts. Our goal is to bridge your "digital AI" knowledge into the physical world using humanoid robots, guiding you from conceptual foundations to simulated and real humanoid control using ROS 2, Gazebo, Unity, and NVIDIA Isaac.

## Features

- Comprehensive curriculum covering physical AI and humanoid robotics
- Structured learning path: Intuition → Concept → Diagram → Minimal Example → Exercises
- Industry-standard tools: ROS 2, Gazebo, Unity, NVIDIA Isaac
- Interactive content and practical exercises
- Spec-driven development workflow

## Modules

1. **Module 1: The Robotic Nervous System (ROS 2)** - Learn how ROS 2 acts as the communication backbone for robots
2. **Module 2: The Digital Twin (Gazebo & Unity)** - Explore physics simulation and virtual environments
3. **Module 3: NVIDIA Isaac & Advanced Simulation** - Advanced simulation techniques and AI integration
4. **Module 4: From Simulation to Real Humanoids** - Transitioning from virtual to real-world robots
5. **Module 5: Human-Robot Interaction & Safety** - Safe and effective human-robot collaboration

## Getting Started

### Prerequisites

- Node.js (version 18 or higher)
- npm or yarn package manager

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd physical-ai-book
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

### Build

To build the static site for production:

```bash
npm run build
```

The static files will be generated in the `build/` folder.

### Deployment

The site can be deployed to GitHub Pages using the provided GitHub Actions workflow, or to any static hosting service.

## Contributing

This project follows a spec-driven development approach. All contributions should follow the Spec-Kit Plus workflow:

1. Create a feature specification using `/sp.specify`
2. Plan the implementation with `/sp.plan`
3. Generate tasks with `/sp.tasks`
4. Implement following the `/sp.implement` command

For more information about how this book is structured and maintained, see our [Spec-Kit Plus workflow documentation](/docs/resources/spec-driven-workflow).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions, please open an issue in the repository or contact the Physical AI Book Team.