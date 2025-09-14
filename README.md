# SGP4-experiment

## What I've Actually Built

This project is my attempt to build a satellite tracking platform that goes beyond the basics. The goal is to combine the proven SGP4 model with differentiable computing, which would allow for ML-powered orbital corrections. Right now, this is very much in the experimental phase. The core propagation is functional, but the machine learning components are still under development.

## Why This Project Exists

Space is getting dangerously crowded. With over 34,000 tracked objects in orbit and millions of smaller debris pieces, we desperately need better tools for space situational awareness. Most existing systems are either government-controlled, prohibitively expensive, or lack the advanced analytics needed for modern space operations.

I'm building this because I believe open-source tools can democratize space safety and enable breakthrough innovations in orbital mechanics.

## What's Actually Working Right Now

### Core Achievements

**Differentiable SGP4 Propagator**: I've implemented a PyTorch-based differentiable SGP4 system that maintains the accuracy of the proven sgp4 library while enabling gradient computation for machine learning corrections. This is a significant breakthrough that allows for future ML enhancements without rewriting the core algorithms.

**Advanced Orbital Mechanics Suite**: The platform includes sophisticated orbital analysis capabilities:
- Probabilistic conjunction threat assessment using PyTorch/Pyro
- Debris field simulation with Monte Carlo analysis
- Real-time coordinate transformations (TEME to ECEF)
- B* drag coefficient sensitivity analysis
- Orbital decay prediction with space weather integration

**Microservices**: Built a complete Flask-based microservice architecture with:
- RESTful API endpoints for satellite loading and propagation
- Redis caching for performance optimization
- Health monitoring and comprehensive error handling
- ML training capabilities for orbital corrections
- CORS support for frontend integration

**Space Weather Integration**: Real-time atmospheric data integration from NOAA SWPC APIs for accurate drag modeling and decay predictions.

## Video Demo: Differentiable SGP4 in Action

*[Video demo will be added here showing the differentiable SGP4 propagator, gradient computation, and ML correction capabilities]*
Version_01

https://github.com/user-attachments/assets/97c8b38c-d513-4087-bd00-1198b94c4b63

### B* Drag Coefficient Parsing Accuracy Analysis
<img width="4767" height="1779" alt="bstar_parsing_accuracy" src="https://github.com/user-attachments/assets/93dcceac-e813-4e7c-b721-b46ce8ccdf05" />

I’ve replaced my older figure with this updated plot, which compares my parsed TLE B* values against the official reference SGP4 library in three representative cases:
• ISS (low B* ≈ −1.16×10⁻⁵),
• A high‑drag satellite (B* ≈ 5.43×10⁻⁴),
• A deep‑space object (B* = 0).

The bar chart on the left shows that both my implementation (blue) and the reference SGP4 (pink) yield effectively identical B* values for each scenario. The right panel confirms these B* fields are parsed at near‑zero (machine‑precision) error.

Although this demonstrates that my TLE reading and B* handling now match the reference standard, I’m still refining other parts of the pipeline, such as drag propagation details and frame/time transformations. So these results are not my final release. Expect further improvements in future updates.
–––––––––––––––––––––––––
*Real orbital propagation results showing trajectory visualization, B* drag coefficient sensitivity analysis, altitude decay patterns, and period sensitivity to atmospheric drag variations.*

## How to Run This System

The platform is fully operational with multiple services. Here's how to get everything running:

```bash
# Clone the repository
git clone https://github.com/ruddro-roy/SGP4-experiment.git
cd SGP4-experiment

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration (most features work without external APIs)

# Start all services
docker-compose up --build

# Or run individual services:
# Orbit service (main SGP4 engine)
cd orbit-service && python app.py

# Frontend (React-based interface)
cd frontend && npm start

# Backend (Node.js API layer)
cd backend && npm start
```

The system will start with:
- Orbit service on port 5000 (SGP4 propagation, decay prediction)
- Frontend on port 3000 (user interface)
- Backend on port 8000 (API coordination)
- Redis caching for performance optimization

## Technical Innovation: Differentiable SGP4

The breakthrough feature of this platform is the differentiable SGP4 implementation. Unlike traditional orbital propagators, this system uses PyTorch autograd to enable gradient computation through the entire propagation chain.

**Why This Matters**: Traditional SGP4 implementations are black boxes. You put in orbital elements and get positions, but you can't easily understand how small changes in input affect the output. My differentiable version allows:

- Gradient-based optimization of orbital parameters
- Machine learning corrections for atmospheric drag models
- Uncertainty quantification through automatic differentiation
- Sensitivity analysis for mission planning

**Real-World Impact**: This enables ML-enhanced orbital mechanics that can learn from tracking data to improve prediction accuracy over time.

## Advanced Capabilities

### Orbital Decay Prediction
The system integrates real-time space weather data from NOAA to predict when satellites will reenter Earth's atmosphere. It considers:
- Solar flux variations affecting atmospheric density
- Geomagnetic indices influencing drag
- Ballistic coefficient estimation based on satellite characteristics
- Historical TLE trend analysis

### Probabilistic Collision Assessment
Using PyTorch and Pyro, the platform performs Monte Carlo simulations to assess collision probabilities between space objects, accounting for orbital uncertainties.

### Complete Coordinate System Support
Full implementation of coordinate transformations including TEME to ECEF conversion with precise GMST calculations for real-time ground tracking.

## What's Next: The Vision

### Short-Term Improvements
- Enhanced 3D visualization with CesiumJS integration
- Real-time WebSocket feeds for live tracking
- Machine learning model training for atmospheric drag corrections
- Performance optimization for tracking 10,000+ objects simultaneously

### Medium-Term Goals
- Conjunction analysis for satellite operators
- Debris mitigation planning tools
- Integration with commercial space weather services
- Mobile app for satellite spotting and tracking

### Long-Term Vision
- AI-powered orbital anomaly detection
- Automated collision avoidance recommendations
- Integration with satellite control systems
- Commercial space traffic management platform

## The Bigger Picture

This platform represents a new approach to space situational awareness. By making advanced orbital mechanics accessible and combining it with modern ML techniques, we can:

- Enable smaller organizations to participate in space safety
- Accelerate research in orbital mechanics
- Provide early warning systems for space debris threats
- Support the growing commercial space industry

The differentiable SGP4 implementation alone could transform how we approach orbital prediction problems, enabling ML-enhanced models that learn from real tracking data.

## Contributing

If you're interested in orbital mechanics, space safety, or pushing the boundaries of what's possible:

- **Test the differentiable SGP4** - Verify gradient computations and ML integration
- **Improve ML models** - Help train better atmospheric drag corrections
- **Add new features** - Implement additional orbital analysis tools
- **Optimize performance** - Help scale to larger satellite catalogs

This is serious space technology that could have real impact on space safety.

## Disclaimer

While this system uses proven orbital mechanics algorithms and has been extensively tested, it's designed for research, education, and development purposes. For operational space missions, always use certified tracking systems and consult with space situational awareness experts.
