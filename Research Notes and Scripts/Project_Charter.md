# Predictive FOD/DOD System - VIT 4th Year Project

## Core Concept
- **FOD:** Foreign Object Damage/Debris
- **DOD:** Damage Object Detection
- **Goal:** Build a **predictive** system ("prevention better than cure") rather than a purely reactive one.

## Team & Stakeholders
- **Professor:** Dr. Parthiban K
  - *Key Directive:* Search for the availability of datasets first. Then decide which domain suits the project best. Identify exactly *where* the FOD happens (which part in which object).
- **Teammate:** Atharva
  - *Key Concern:* How the implementation will work practically. Also agrees that Aviation is too complex to implement and should be avoided.

## Domains & Ideas Evaluated

1. **Aviation** (Primary Focus)
   - *Idea:* Airport Runway FOD Detection.
   - *Datasets Found:* FOD-A Dataset (Kaggle), Roboflow Universe.
   - *Implementation:* YOLOv8 object detection on runway images to identify nuts, bolts, and debris.

2. **Automotive** (Strong Candidate)
   - *Idea:* AI-Based Road Hazard Detection and Tire Damage Risk Assessment.
   - *Scenario:* Detecting debris/people on the road *before* a collision occurs to slow down the vehicle predictively.

3. **PCB Manufacturing** (Strong Candidate)
   - *Idea:* AI-Based Detection of Foreign Metal Particles on PCB Tracks for Short-Circuit Prevention.
   - *Scenario:* Detecting tiny metal anomalies during the assembly process to prevent later failures.

4. **Marine** (Dropped)
   - *Verdict:* Team lead confirmed this is not an option.

5. **Medical** (Alternative Candidate)
   - *Idea:* Detecting abnormalities in samples and imaging results.

## Action Plan (How we move forward)
To answer Atharva's question ("So should we decide the focused part prior and go with the research or decide the focused part based on research?"):
**We must follow Dr. Parthiban's advice:** Decide the focused part *based on dataset research*.

**Step 1:** Use the **FOD-A** dataset (or Roboflow equivalent) which contains annotated airport runway debris.
**Step 2:** Train a YOLOv8 object detection model on the dataset to detect small objects (nuts, bolts, safety wires).
**Step 3:** Implement predictive analysis (e.g., mapping debris density on runways to predict high-risk zones).
