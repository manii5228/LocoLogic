# Railway Traffic Simulation and Rescheduling System

A comprehensive event-driven railway simulator with conflict detection and intelligent rescheduling capabilities, designed for Tamil Nadu railway network analysis.

## Overview

This system consists of two main components:

  - Railway Simulator (rail_sim_tn_complete.py) - A physically-aware event-driven simulator that models train movements, signaling, and dispatcher policies

  - Rescheduler (member2_rescheduler_tn.py) - An optimization system that detects conflicts and applies ACO (Ant Colony Optimization) or rule-based rescheduling

  ## Features
    ### Simulator Features
        - Multi-line railway network with junctions and bidirectional edges
          
        - Continuous speed carryover between track segments (no speed reset)
          
        - Physically accurate kinematics - acceleration, cruising, and braking profiles
          
        - Train length & tail clearance modeling
          
        - Absolute-block signaling with headway enforcement
          
        - Multiple dispatcher policies: FCFS, Priority-based, Throughput-optimized
          
        - Station stops with dwell times and scheduled departures
          
        - Energy consumption estimation
          
        - Comprehensive outputs: events CSV, configuration JSON, network plots, time-distance diagrams, animations
     
     ### Rescheduler Features
