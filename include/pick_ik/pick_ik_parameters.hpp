#pragma once

#include <string>

namespace pick_ik {
struct PickIKParameters {
    std::string mode;     // IK solver mode. Set to global to allow the initial guess to be a long
                          // distance from the goal, or local if the initial guess is near the goal.
    double gd_step_size;  // Gradient descent step size for joint perturbation
    int gd_max_iters;     // Maximum iterations for local gradient descent
    double gd_min_cost_delta;  // Minimum change in cost function value for gradient descent

    // Cost functions and thresholds
    double position_threshold;     // Position threshold for solving IK, in meters
    double orientation_threshold;  // Orientation threshold for solving IK, in radians
    double approximate_solution_position_threshold;  // Position threshold for approximate IK
                                                     // solutions, in meters. If displacement is
                                                     // larger than this, the approximate solution
                                                     // will fall back to the initial guess
    double
        approximate_solution_orientation_threshold;  // Orientation threshold for approximate IK
                                                     // solutions, in radians. If displacement is
                                                     // larger than this, the approximate solution
                                                     // will fall back to the initial guess
    double
        cost_threshold;  // Scalar value for comparing to result of cost functions. IK is considered
                         // solved when all position/rotation/twist thresholds are satisfied and all
                         // cost functions return a value lower than this value.
    double rotation_scale;  // The rotation scale for the pose cost function. Set to 0.0 to solve
                            // for only position
    double center_joints_weight;  // Weight for centering cost function, >0.0 enables const function
    double avoid_joint_limits_weight;    // Weight for avoiding joint limits cost function, >0.0
                                         // enables const function
    double minimal_displacement_weight;  // Weight for minimal displacement cost function, >0.0
                                         // enables const function

    // Memetic IK specific parameters
    int memetic_num_threads;              // Number of threads for memetic IK
    bool memetic_stop_on_first_solution;  // If true, stops on first solution and terminates other
                                          // threads
    int memetic_population_size;          // Population size for memetic IK
    int memetic_elite_size;               // Number of elite members of memetic IK population
    double memetic_wipeout_fitness_tol;  // Minimum fitness must improve by this value or population
                                         // will be wiped out
    int memetic_max_generations;         // Maximum iterations of evolutionary algorithm
    int memetic_gd_max_iters;  // Maximum iterations of gradient descent during memetic exploitation
    double
        memetic_gd_max_time;  // Maximum time spent on gradient descent during memetic exploitation
};
}  // namespace pick_ik
