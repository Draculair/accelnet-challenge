## Running the Simulation

1. **Start ROS Core**
   - Open a terminal and run the following command:
     ```sh
     roscore
     ```

2. **Launch the Simulator**
   - Open a second terminal and navigate to the project directory to start the simulator:
     ```sh
     cd <path to accel-challenge>
     source bash/run_simulator.sh
     ```

3. **Initialize the CRTK Interface**
   - Open a third terminal to initialize the `crtk_interface` which handles controllers and ROS topics:
     ```sh
     cd <path to accel-challenge>
     source bash/run_crtk_interface.sh
     ```

4. **Run Challenge 1 Script**
   - Open a fourth terminal and execute the script for challenge 1:
     ```sh
     cd <path to accel-challenge>
     source bash/init.sh
     cd <path to accel-challenge>/accel_challenge/challenge1
     python examples/challenge1_traj.py 
     ```

5. **Evaluate Challenge 1**
   - Open a fifth terminal to run the evaluation for challenge 1:
     ```bash
     cd <path to accel-challenge>
     source bash/init.sh
     cd <path to surgical_robotics_challenge>/scripts/surgical_robotics_challenge/evaluation
     python evaluation.py -t Amagi -e 1
     ```