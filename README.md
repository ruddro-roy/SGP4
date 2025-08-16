# SGP4 Satellite Tracking Experiment

## What I'm Building Here

I'm working on a satellite tracking platform that uses SGP4 orbital mechanics to predict and visualize satellite positions in real-time. This is currently an experimental project in active development, not a finished product.

## Why This Matters

Space is getting crowded. With thousands of satellites and debris pieces orbiting Earth, we need better tools to track them and predict potential collisions. Current tracking systems are often expensive, closed-source, or limited in scope. I want to build something that's:

- Open and accessible to researchers and enthusiasts
- Accurate enough for real collision avoidance analysis
- Easy to understand and modify
- Built with modern web technologies

## Current Status: Building Phase

This project is actively under construction. Here's what's working and what's not:

### Advanced Orbital Analysis Capabilities

The system now includes orbital mechanics analysis with B* parameter sensitivity studies and trajectory propagation:

## Orbital Trajectories and B* Sensitivity Analysis
<img width="1400" height="600" alt="Figure_8" src="https://github.com/user-attachments/assets/6d54e0cc-f643-43c5-b3f4-3e9ebeba85a0" />


*Real orbital propagation results showing trajectory visualization, B* drag coefficient sensitivity analysis, altitude decay patterns, and period sensitivity to atmospheric drag variations.*

### What's Done
- Basic project structure with microservices architecture
- Docker containerization setup
- CI/CD pipeline configuration (security-cleaned)
- Environment configuration templates

### What I'm Working On
- SGP4 orbital propagation implementation
- Real-time satellite position calculations
- 3D visualization with CesiumJS
- API integration with CELESTRAK and Space-Track
- Collision detection algorithms

### What's Not Ready Yet
- The actual tracking functionality (core feature!)
- User interface and visualization
- Real-time data feeds
- Performance optimization
- Comprehensive testing

## The Technical Challenges I'm Facing

### Orbital Mechanics Complexity
SGP4 (Simplified General Perturbations 4) is the standard algorithm for satellite tracking, but it's not simple to implement correctly. The math involves:
- Converting Two-Line Element (TLE) data into orbital parameters
- Accounting for Earth's gravitational field irregularities
- Handling atmospheric drag effects on low Earth orbit satellites
- Dealing with coordinate system transformations (TEME to ITRF)

### Real-Time Performance
Tracking thousands of satellites simultaneously while maintaining accuracy is computationally intensive. I need to balance:
- Update frequency vs computational load
- Prediction accuracy vs processing speed
- Memory usage for caching orbital data
- Network bandwidth for real-time updates

### Data Quality and Availability
- TLE data can be hours or days old by the time it's published
- Different data sources have varying accuracy levels
- Space-Track.org requires authentication and has rate limits
- Some satellite operators don't publish orbital data at all

### Safety and Responsibility Concerns

This isn't just a coding project. Satellite tracking has real-world implications:

**Collision Avoidance**: If someone uses this for actual mission planning, incorrect calculations could lead to satellite collisions. I need to be very clear about accuracy limitations and testing status.

**Space Debris**: Misidentifying debris or predicting wrong trajectories could affect space operations. The system needs proper error handling and uncertainty quantification.

**Security**: Satellite positions can be sensitive information. While most orbital data is public, I need to be careful about how detailed tracking information is presented.

**Dual-Use Concerns**: Satellite tracking technology can have military applications. I'm keeping this project focused on civilian space safety and making it openly available to promote transparency.

## How to Run This (When It's Ready)

Right now, the project structure is set up but the core functionality isn't implemented yet. If you want to explore the codebase:

```bash
# Clone the repository
git clone https://github.com/ruddro-roy/SGP4-experiment.git
cd SGP4-experiment

# Set up environment
cp .env.example .env
# Edit .env with your API keys (when you get them)

# Try to run it (expect errors for now)
docker-compose up --build
```

**Warning**: This won't actually track satellites yet. The services will start but won't do much useful work.

## What I'm Learning

This project is as much about learning as it is about building. I'm diving deep into:

- **Orbital mechanics**: Understanding how satellites actually move
- **Coordinate systems**: Converting between different reference frames
- **Real-time systems**: Handling continuous data streams efficiently  
- **3D visualization**: Making complex orbital data understandable
- **API design**: Building systems that others can actually use

## The Bigger Picture

Space situational awareness is becoming critical as more countries and companies launch satellites. The current tracking infrastructure is mostly government-controlled and not always accessible to researchers, students, or smaller organizations.

I believe open-source tools can help democratize space safety. If we can build accurate, accessible tracking systems, more people can contribute to solving the space debris problem.

## Current Roadmap

1. **Get basic SGP4 working** - Start with single satellite tracking
2. **Add real TLE data feeds** - Connect to CELESTRAK and Space-Track APIs  
3. **Build simple visualization** - 2D map before attempting 3D
4. **Implement collision detection** - Basic closest approach calculations
5. **Add real-time updates** - WebSocket connections for live data
6. **Scale to multiple satellites** - Performance optimization
7. **Add uncertainty quantification** - Show prediction confidence levels

## Want to Help?

If you're interested in orbital mechanics, space safety, or just want to learn alongside me:

- **Check the issues** - I'll document specific problems I'm working on
- **Test calculations** - Help verify SGP4 implementation accuracy
- **Improve documentation** - Make this more accessible to newcomers
- **Share knowledge** - Point out better approaches or resources

This is a learning project, so don't expect production-ready code. But if you're curious about how satellite tracking works, you're welcome to follow along.

## Disclaimer

This is experimental software for educational and research purposes. Do not use it for actual space operations, mission planning, or any safety-critical applications. Always use official, validated tracking systems for real-world space activities.
