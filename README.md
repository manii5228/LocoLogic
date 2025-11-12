# Railway Traffic Simulation and Rescheduling System

A comprehensive event-driven railway simulator with conflict detection and intelligent rescheduling capabilities, designed for Tamil Nadu railway network analysis.

## Overview

This system consists of two main components:

  - Railway Simulator (rail_sim_tn_complete.py) - A physically-aware event-driven simulator that models train movements, signaling, and dispatcher policies

  - Rescheduler (member2_rescheduler_tn.py) - An optimization system that detects conflicts and applies ACO (Ant Colony Optimization) or rule-based rescheduling

## Features
  ### Simulator Features
  * Multi-line railway network with junctions and bidirectional edges
          
  * Continuous speed carryover between track segments (no speed reset)
          
  * Physically accurate kinematics - acceleration, cruising, and braking profiles
  * Train length & tail clearance modeling
          
  * Absolute-block signaling with headway enforcement
          
  * Multiple dispatcher policies: FCFS, Priority-based, Throughput-optimized
          
  * Station stops with dwell times and scheduled departures
          
  * Energy consumption estimation
          
  * Comprehensive outputs: events CSV, configuration JSON, network plots, time-distance diagrams, animations
     
  ### Rescheduler Features
  * Conflict detection: Overlap and headway violation identification

  * Dual optimization methods:

    * ACO (Ant Colony Optimization): Metaheuristic approach for complex scenarios

    * Rule-based: Fast deterministic baseline for quick solutions

  * Priority-aware scheduling: Respects train priorities (Express > Passenger > Freight)

  * Safety margin enforcement

  * Visual comparison of before/after schedules

## Installation Requirements
```
# Core requirements
python >= 3.8

# Optional for plotting and animation
matplotlib >= 3.5.0
pillow >= 9.0.0  # for GIF animation
pandas >= 1.4.0   # for data handling
```
## Usage

  ### 1.Running the Simulator
  ```
  python rail_sim_tn_complete.py --trains 80 --sim-hours 4 --out ./output --seed 123 --dispatcher priority --plot --animate

  ```
  ### Parameters :
  * --trains: Number of trains to simulate (default: 80)

* --sim-hours: Simulation duration in hours (default: 4.0)

* --out: Output directory (default: ./out_full)

* --seed: Random seed for reproducible results

* --dispatcher: Scheduling policy [fcfs|priority|throughput]

* --plot: Generate network and time-distance plots

* --animate: Create animation GIF (requires pillow)

  ### 2.Running the Rescheduler
  ```
  # ACO only
  python member2_rescheduler_tn.py --events ./output/events.csv --config ./output/config.json --out ./rescheduled --solver aco

  # Rule-based only
  python member2_rescheduler_tn.py --events ./output/events.csv --config ./output/config.json --out ./rescheduled --solver rule

  # Both methods with comparison
  python member2_rescheduler_tn.py --events ./output/events.csv --config ./output/config.json --out ./rescheduled --solver both --ants 40 --iters 100

  ```
  ### Rescheduler Parameters :
  * --events: Path to events CSV from simulator

* --config: Path to config.json from simulator

* --out: Output directory for rescheduled results

* --solver: Optimization method [aco|rule|both]

* --ants: Number of ants for ACO (default: 40)

* --iters: ACO iterations (default: 100)

* --safety-margin: Safety buffer in seconds (default: 300)

* --max-hold: Maximum hold time per train in seconds (default: 14400)


## Output Files
  ### Simulator Outputs
* events.csv: Complete timeline of all train movements

* config.json: Network configuration and train specifications

* network.png: Visual representation of railway network

* time_distance_stations.png: Time-distance diagram with station labels

* animation.gif: Animated visualization (if enabled)

<img width="282" height="175" alt="Image" src="https://github.com/user-attachments/assets/43378501-a2fe-4476-951b-c4845c80b0c8" />

<img width="274" height="177" alt="Image" src="https://github.com/user-attachments/assets/901249f0-3c91-4941-89cb-7beefaf91e5f" />

<img width="447" height="223" alt="Image" src="https://github.com/user-attachments/assets/73397b15-0f76-48fd-a4cc-a3289bb11469" />

### Rescheduler Outputs
* actions.json: Recommended hold times for each train

* reschedule_before_events.csv: Original event timeline
* [reschedule_before_events.csv](https://github.com/user-attachments/files/23505266/reschedule_before_events.csv)



* reschedule_after_events.csv: Rescheduled event timeline
* ![Image](https://github.com/user-attachments/assets/9b1a7e2f-6ecc-43f9-b159-faf843f34479)


* reschedule_summary.csv: Summary of applied holds
* [reschedule_summary.csv](https://github.com/user-attachments/files/23505262/reschedule_summary.csv)




<img width="2400" height="1600" alt="Image" src="https://github.com/user-attachments/assets/0d50f5c6-0f93-40fd-a5e6-abe08dcb0d39" />

[reschedule_after_events.csv](https://github.com/user-attachments/files/23505265/reschedule_after_events.csv)


## Network Configuration
 The system models an illustrative Tamil Nadu railway network with key stations:
   <img width="2000" height="1600" alt="Image" src="https://github.com/user-attachments/assets/fcd398a1-63a7-4075-a61b-79ee61a9a58c" />


* Chennai Central (MAS) - Starting point at 0km

* Tambaram (TRM) - 30km

* Villupuram (VPM) - 155km

* Tiruchirappalli (TIR) - 330km

* Salem (SAL) - 395km

* Erode (ERP) - 495km

* Coimbatore (CBE) - 620km

* Madurai (MDU) - 810km

* Tirunelveli (TVC) - 950km

## Algorithm Details
* ACO Rescheduler
    * Uses pheromone trails to guide search toward good solutions

    * Incorporates train priority and conflict severity in heuristic

    * Applies local repair for constraint satisfaction

    * Configurable parameters for ants, iterations, and exploration

* Rule-based Rescheduler
    * Fast deterministic approach

    * Processes conflicts chronologically

    * Applies minimal delays to resolve violations

    * Useful as baseline or for quick solutions




  
  
