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

4. **Run needle tracking algorithm**
   - Open fourth terminal to run needle tracking algorithm
     ```sh
     cd <path to accel-challenge>
     source bash/init.sh
     cd <path to accel-challenge>/accel_challenge/challenge2
     python task_completion_report_for_challenge2.py
     ```
        
5. **Run Challenge 2 Script**
   - Open a fifth terminal and execute the script for challenge 2:
     ```sh
     cd <path to accel-challenge>
     source bash/init.sh
     cd <path to accel-challenge>/accel_challenge/challenge1
     python challenge2_traj.py
     ```

6. **Evaluate Challenge 2**
   - Open a sixth terminal to run the evaluation for challenge 1:
     ```bash
     cd <path to accel-challenge>
     source bash/init.sh
     cd <path to surgical_robotics_challenge>/scripts/surgical_robotics_challenge/evaluation
     python evaluation.py -t Amagi -e 2
     ```