#include <pick_ik/fk_moveit.hpp>
#include <pick_ik/goal.hpp>
#include <pick_ik/ik_gradient.hpp>
#include <pick_ik/ik_memetic.hpp>
#include <pick_ik/pick_ik_parameters.hpp>
#include <pick_ik/robot.hpp>

#include <pluginlib/class_list_macros.hpp>

#include <moveit/kinematics_base/kinematics_base.h>
#include <moveit/robot_model/joint_model_group.h>
#include <moveit/robot_state/robot_state.h>
#include <ros/ros.h>
#include <string>
#include <vector>

namespace pick_ik {
namespace {
char const* LOGNAME = "pick_ik";
}

class PickIKPlugin : public kinematics::KinematicsBase {
    ros::NodeHandle nh_;
    moveit::core::JointModelGroup const* jmg_;

    std::vector<std::string> joint_names_;
    std::vector<std::string> link_names_;
    std::vector<size_t> tip_link_indices_;
    std::string ns_;
    Robot robot_;

   public:
    virtual bool initialize(moveit::core::RobotModel const& robot_model,
                            std::string const& group_name,
                            std::string const& base_frame,
                            std::vector<std::string> const& tip_frames,
                            double search_discretization) {
        // Initialize internal state of base class KinematicsBase
        // Creates these internal state variables:
        // robot_model_ <- shared_ptr to RobotModel
        // robot_description_ <- empty string
        // group_name_ <- group_name string
        // base_frame_ <- base_frame without leading /
        // tip_frames_ <- tip_frames without leading /
        // redundant_joint_discretization_ <- vector initialized with
        // search_discretization
        storeValues(robot_model, group_name, base_frame, tip_frames, search_discretization);

        // Initialize internal state
        jmg_ = robot_model_->getJointModelGroup(group_name);
        if (!jmg_) {
            ROS_ERROR(LOGNAME, "failed to get joint model group %s", group_name.c_str());
            return false;
        }

        // Joint names come from jmg
        for (auto* joint_model : jmg_->getJointModels()) {
            if (joint_model->getName() != base_frame_ &&
                joint_model->getType() != moveit::core::JointModel::UNKNOWN &&
                joint_model->getType() != moveit::core::JointModel::FIXED) {
                joint_names_.push_back(joint_model->getName());
            }
        }

        // If jmg has tip frames, set tip_frames_ to jmg tip frames
        // consider removing these lines as they might be unnecessary
        // as tip_frames_ is set by the call to storeValues above
        auto jmg_tips = std::vector<std::string>{};
        jmg_->getEndEffectorTips(jmg_tips);
        if (!jmg_tips.empty()) tip_frames_ = jmg_tips;

        // link_names are the same as tip frames
        // TODO: why do we need to set this
        link_names_ = tip_frames_;

        // Create our internal Robot object from the robot model
        tip_link_indices_ =
            get_link_indices(robot_model_, tip_frames_)
                .or_else([](auto const& error) { throw std::invalid_argument(error); })
                .value();
        robot_ = Robot::from(robot_model_, jmg_, tip_link_indices_);

        ns_ = "/robot_description_kinematics/" + group_name + "/";
        return true;
    }

    virtual bool searchPositionIK(
        std::vector<geometry_msgs::Pose> const& ik_poses,
        std::vector<double> const& ik_seed_state,
        double timeout,
        std::vector<double> const&,
        std::vector<double>& solution,
        IKCallbackFn const& solution_callback,
        IKCostFn cost_function,
        moveit_msgs::MoveItErrorCodes& error_code,
        kinematics::KinematicsQueryOptions const& options = kinematics::KinematicsQueryOptions(),
        moveit::core::RobotState const* context_state = nullptr) const {
        (void)context_state;  // not used

        // Read current ROS parameters
        PickIKParameters params;
        if (!nh_.getParam(ns_ + "rotation_scale", params.rotation_scale)) {
            params.rotation_scale = 0.5;
            ROS_INFO_STREAM(
                "Param rotation_scale was not set. Using default value: " << params.rotation_scale);
        }
        if (!nh_.getParam(ns_ + "position_threshold", params.position_threshold)) {
            params.position_threshold = 0.001;
            ROS_INFO_STREAM("Param position_threshold was not set. Using default value: "
                            << params.position_threshold);
        }
        if (!nh_.getParam(ns_ + "center_joints_weight", params.center_joints_weight)) {
            params.center_joints_weight = 0.0;
            ROS_INFO_STREAM("Param center_joints_weight was not set. Using default value: "
                            << params.center_joints_weight);
        }
        if (!nh_.getParam(ns_ + "avoid_joint_limits_weight", params.avoid_joint_limits_weight)) {
            params.avoid_joint_limits_weight = 0.0;
            ROS_INFO_STREAM("Param avoid_joint_limits_weight was not set. Using default value: "
                            << params.avoid_joint_limits_weight);
        }
        if (!nh_.getParam(ns_ + "orientation_threshold", params.orientation_threshold)) {
            params.orientation_threshold = 0.001;
            ROS_INFO_STREAM("Param orientation_threshold was not set. Using default value: "
                            << params.orientation_threshold);
        }
        if (!nh_.getParam(ns_ + "minimal_displacement_weight",
                          params.minimal_displacement_weight)) {
            params.minimal_displacement_weight = 0.0;
            ROS_INFO_STREAM("Param minimal_displacement_weight was not set. Using default value: "
                            << params.minimal_displacement_weight);
        }
        if (!nh_.getParam(ns_ + "cost_threshold", params.cost_threshold)) {
            params.cost_threshold = 0.001;
            ROS_INFO_STREAM(
                "Param cost_threshold was not set. Using default value: " << params.cost_threshold);
        }
        if (!nh_.getParam(ns_ + "mode", params.mode)) {
            params.mode = "global";
            ROS_INFO_STREAM("Param mode was not set. Using default value: " << params.mode);
        }

        // Check parameter limits
        if (params.rotation_scale < 0.0) {
            params.rotation_scale = 0.5;
            ROS_INFO_STREAM("Param rotation_scale was less than minimum 0.0. Using default value: "
                            << params.rotation_scale);
        }
        if (params.position_threshold < 0.0) {
            params.position_threshold = 0.001;
            ROS_INFO_STREAM(
                "Param position_threshold was less than minimum 0.0. Using default value: "
                << params.position_threshold);
        }
        if (params.center_joints_weight < 0.0) {
            params.center_joints_weight = 0.0;
            ROS_INFO_STREAM(
                "Param center_joints_weight was less than minimum 0.0. Using default value: "
                << params.center_joints_weight);
        }
        if (params.avoid_joint_limits_weight < 0.0) {
            params.avoid_joint_limits_weight = 0.0;
            ROS_INFO_STREAM(
                "Param avoid_joint_limits_weight was less than minimum 0.0. Using default value: "
                << params.avoid_joint_limits_weight);
        }
        if (params.orientation_threshold < 0.0) {
            params.orientation_threshold = 0.001;
            ROS_INFO_STREAM(
                "Param orientation_threshold was less than minimum 0.0. Using default value: "
                << params.orientation_threshold);
        }
        if (params.minimal_displacement_weight < 0.0) {
            params.minimal_displacement_weight = 0.0;
            ROS_INFO_STREAM(
                "Param minimal_displacement_weight was less than minimum 0.0. Using default value: "
                << params.minimal_displacement_weight);
        }
        if (params.cost_threshold < 0.0) {
            params.cost_threshold = 0.001;
            ROS_INFO_STREAM("Param cost_threshold was less than minimum 0.0. Using default value: "
                            << params.cost_threshold);
        }
        if (params.mode != "local" && params.mode != "global") {
            params.mode = "global";
            ROS_INFO_STREAM(
                "Param mode was not \"local\" or \"global\". Using default value: " << params.mode);
        }

        auto const goal_frames = [&]() {
            auto robot_state = moveit::core::RobotState(robot_model_);
            robot_state.setToDefaultValues();
            robot_state.setJointGroupPositions(jmg_, ik_seed_state);
            robot_state.update();
            return transform_poses_to_frames(robot_state, ik_poses, getBaseFrame());
        }();

        // Test functions to determine if we are at our goal frame
        auto const test_rotation = (params.rotation_scale > 0);
        std::optional<double> orientation_threshold = std::nullopt;
        if (test_rotation) {
            orientation_threshold = params.orientation_threshold;
        }
        auto const frame_tests =
            make_frame_tests(goal_frames, params.position_threshold, orientation_threshold);

        // Cost functions used for optimizing towards goal frames
        auto const pose_cost_functions =
            make_pose_cost_functions(goal_frames, params.rotation_scale);

        // forward kinematics function
        auto const fk_fn = make_fk_fn(robot_model_, jmg_, tip_link_indices_);

        // Create goals (weighted cost functions)
        auto goals = std::vector<Goal>{};
        if (params.center_joints_weight > 0.0) {
            goals.push_back(Goal{make_center_joints_cost_fn(robot_), params.center_joints_weight});
        }
        if (params.avoid_joint_limits_weight > 0.0) {
            goals.push_back(
                Goal{make_avoid_joint_limits_cost_fn(robot_), params.avoid_joint_limits_weight});
        }
        if (params.minimal_displacement_weight > 0.0) {
            goals.push_back(Goal{make_minimal_displacement_cost_fn(robot_, ik_seed_state),
                                 params.minimal_displacement_weight});
        }
        if (cost_function) {
            for (auto const& pose : ik_poses) {
                goals.push_back(
                    Goal{make_ik_cost_fn(pose, cost_function, robot_model_, jmg_, ik_seed_state),
                         1.0});
            }
        }

        // test if this is a valid solution
        auto const solution_fn =
            make_is_solution_test_fn(frame_tests, goals, params.cost_threshold, fk_fn);

        // single function used by gradient descent to calculate cost of solution
        auto const cost_fn = make_cost_fn(pose_cost_functions, goals, fk_fn);

        // Search for a solution using either the local or global solver.
        std::optional<std::vector<double>> maybe_solution;
        if (params.mode == "global") {
            // Read ROS parameters related to global solver

            if (!nh_.getParam(ns_ + "memetic_population_size", params.memetic_population_size)) {
                params.memetic_population_size = 16;
                ROS_INFO_STREAM("Param memetic_population_size was not set. Using default value: "
                                << params.memetic_population_size);
            }
            if (!nh_.getParam(ns_ + "memetic_elite_size", params.memetic_elite_size)) {
                params.memetic_elite_size = 4;
                ROS_INFO_STREAM("Param memetic_elite_size was not set. Using default value: "
                                << params.memetic_elite_size);
            }
            if (!nh_.getParam(ns_ + "memetic_wipeout_fitness_tol",
                              params.memetic_wipeout_fitness_tol)) {
                params.memetic_wipeout_fitness_tol = 0.00001;
                ROS_INFO_STREAM(
                    "Param memetic_wipeout_fitness_tol was not set. Using default value: "
                    << params.memetic_wipeout_fitness_tol);
            }
            if (!nh_.getParam(ns_ + "memetic_num_threads", params.memetic_num_threads)) {
                params.memetic_num_threads = 1;
                ROS_INFO_STREAM("Param memetic_num_threads was not set. Using default value: "
                                << params.memetic_num_threads);
            }
            if (!nh_.getParam(ns_ + "memetic_stop_on_first_solution",
                              params.memetic_stop_on_first_solution)) {
                params.memetic_stop_on_first_solution = true;
                ROS_INFO_STREAM(
                    "Param memetic_stop_on_first_solution was not set. Using default value: "
                    << params.memetic_stop_on_first_solution);
            }
            if (!nh_.getParam(ns_ + "memetic_max_generations", params.memetic_max_generations)) {
                params.memetic_max_generations = 100;
                ROS_INFO_STREAM("Param memetic_max_generations was not set. Using default value: "
                                << params.memetic_max_generations);
            }
            if (!nh_.getParam(ns_ + "gd_step_size", params.gd_step_size)) {
                params.gd_step_size = 0.0001;
                ROS_INFO_STREAM(
                    "Param gd_step_size was not set. Using default value: " << params.gd_step_size);
            }
            if (!nh_.getParam(ns_ + "gd_min_cost_delta", params.gd_min_cost_delta)) {
                params.gd_min_cost_delta = 1.0e-12;
                ROS_INFO_STREAM("Param gd_min_cost_delta was not set. Using default value: "
                                << params.gd_min_cost_delta);
            }
            if (!nh_.getParam(ns_ + "memetic_gd_max_iters", params.memetic_gd_max_iters)) {
                params.memetic_gd_max_iters = 100;
                ROS_INFO_STREAM("Param memetic_gd_max_iters was not set. Using default value: "
                                << params.memetic_gd_max_iters);
            }
            if (!nh_.getParam(ns_ + "memetic_gd_max_time", params.memetic_gd_max_time)) {
                params.memetic_gd_max_time = 0.005;
                ROS_INFO_STREAM("Param memetic_gd_max_time was not set. Using default value: "
                                << params.memetic_gd_max_time);
            }

            // Check paramter limits
            if (params.memetic_population_size < 1) {
                params.memetic_population_size = 16;
                ROS_INFO_STREAM(
                    "Param memetic_population_size was less than minimum 1. Using default "
                    "value: "
                    << params.memetic_population_size);
            }
            if (params.memetic_elite_size < 1) {
                params.memetic_elite_size = 4;
                ROS_INFO_STREAM(
                    "Param memetic_elite_size was less than minimum 1. Using default value: "
                    << params.memetic_elite_size);
            }
            if (params.memetic_wipeout_fitness_tol < 0) {
                params.memetic_wipeout_fitness_tol = 0.00001;
                ROS_INFO_STREAM(
                    "Param memetic_wipeout_fitness_tol was less than minimum 0. Using default "
                    "value: "
                    << params.memetic_wipeout_fitness_tol);
            }
            if (params.memetic_num_threads < 0) {
                params.memetic_num_threads = 1;
                ROS_INFO_STREAM(
                    "Param memetic_num_threads was less than minimum 0. Using default value: "
                    << params.memetic_num_threads);
            }
            if (params.memetic_max_generations < 1) {
                params.memetic_max_generations = 100;
                ROS_INFO_STREAM(
                    "Param memetic_max_generations was less than minimum 1. Using default value: "
                    << params.memetic_max_generations);
            }
            if (params.gd_step_size < 1.0e-12) {
                params.gd_step_size = 0.0001;
                ROS_INFO_STREAM(
                    "Param gd_step_size was less than minimum 1.0e-12. Using default value: "
                    << params.gd_step_size);
            }
            if (params.gd_min_cost_delta < 1.0e-64) {
                params.gd_min_cost_delta = 1.0e-12;
                ROS_INFO_STREAM(
                    "Param gd_min_cost_delta was less than minimum 1.0e-64. Using default value: "
                    << params.gd_min_cost_delta);
            }
            if (params.memetic_gd_max_iters < 1) {
                params.memetic_gd_max_iters = 100;
                ROS_INFO_STREAM("Param memetic_gd_max_iters was not set. Using default value: "
                                << params.memetic_gd_max_iters);
            }
            if (params.memetic_gd_max_time < 0) {
                params.memetic_gd_max_time = 0.005;
                ROS_INFO_STREAM(
                    "Param memetic_gd_max_time was less than minimum 0. Using default value: "
                    << params.memetic_gd_max_time);
            }

            MemeticIkParams ik_params;
            ik_params.population_size = static_cast<size_t>(params.memetic_population_size);
            ik_params.elite_size = static_cast<size_t>(params.memetic_elite_size);
            ik_params.wipeout_fitness_tol = params.memetic_wipeout_fitness_tol;
            ik_params.num_threads = static_cast<size_t>(params.memetic_num_threads);
            ik_params.stop_on_first_soln = params.memetic_stop_on_first_solution;
            ik_params.max_generations = static_cast<int>(params.memetic_max_generations);
            ik_params.max_time = timeout;

            ik_params.gd_params.step_size = params.gd_step_size;
            ik_params.gd_params.min_cost_delta = params.gd_min_cost_delta;
            ik_params.gd_params.max_iterations = static_cast<int>(params.memetic_gd_max_iters);
            ik_params.gd_params.max_time = params.memetic_gd_max_time;

            maybe_solution = ik_memetic(ik_seed_state,
                                        robot_,
                                        cost_fn,
                                        solution_fn,
                                        ik_params,
                                        options.return_approximate_solution,
                                        false /* No debug print */);
        } else if (params.mode == "local") {
            // Read ROS parameters related to local solver
            if (!nh_.getParam(ns_ + "gd_step_size", params.gd_step_size)) {
                params.memetic_gd_max_time = 0.005;
                ROS_INFO_STREAM("Param memetic_gd_max_time was not set. Using default value: "
                                << params.memetic_gd_max_time);
            }
            if (!nh_.getParam(ns_ + "gd_min_cost_delta", params.gd_min_cost_delta)) {
                params.gd_min_cost_delta = 1.0e-12;
                ROS_INFO_STREAM("Param gd_min_cost_delta was not set. Using default value: "
                                << params.gd_min_cost_delta);
            }
            if (!nh_.getParam(ns_ + "gd_max_iters", params.gd_max_iters)) {
                params.gd_max_iters = 100;
                ROS_INFO_STREAM(
                    "Param gd_max_iters was not set. Using default value: " << params.gd_max_iters);
            }

            // Check paramter limits
            if (params.gd_step_size < 1.0e-12) {
                params.gd_step_size = 0.0001;
                ROS_INFO_STREAM(
                    "Param gd_step_size was less than minimum 1.0e-12. Using default value: "
                    << params.gd_step_size);
            }
            if (params.gd_min_cost_delta < 1.0e-64) {
                params.gd_min_cost_delta = 1.0e-12;
                ROS_INFO_STREAM(
                    "Param gd_min_cost_delta was less than minimum 1.0e-64. Using default value: "
                    << params.gd_min_cost_delta);
            }
            if (params.gd_max_iters < 0) {
                params.gd_max_iters = 100;
                ROS_INFO_STREAM("Param gd_max_iters was less than minimum 1. Using default value: "
                                << params.gd_max_iters);
            }
            GradientIkParams gd_params;
            gd_params.step_size = params.gd_step_size;
            gd_params.min_cost_delta = params.gd_min_cost_delta;
            gd_params.max_time = timeout;
            gd_params.max_iterations = static_cast<int>(params.gd_max_iters);

            maybe_solution = ik_gradient(ik_seed_state,
                                         robot_,
                                         cost_fn,
                                         solution_fn,
                                         gd_params,
                                         options.return_approximate_solution);
        } else {
            ROS_ERROR(LOGNAME, "Invalid solver mode: %s", params.mode.c_str());
            return false;
        }

        if (maybe_solution.has_value()) {
            // Set the output parameter solution.
            // Assumes that the angles were already wrapped by the solver.
            error_code.val = error_code.SUCCESS;
            solution = maybe_solution.value();
            if (jmg_->enforcePositionBounds(solution.data())) {
                error_code.val = error_code.NO_IK_SOLUTION;
                solution = ik_seed_state;
            }
        } else {
            error_code.val = error_code.NO_IK_SOLUTION;
            solution = ik_seed_state;
        }

        // If using an approximate solution, check against the maximum allowable pose threshold.
        // If the approximate solution is too far from the goal frame, fall back to the initial
        // state.
        if (options.return_approximate_solution) {
            // Read ROS parameters related to approximate solutions
            if (!nh_.getParam(ns_ + "approximate_solution_position_threshold",
                              params.approximate_solution_position_threshold)) {
                params.approximate_solution_position_threshold = 0.05;
                ROS_INFO_STREAM(
                    "Param approximate_solution_position_threshold was not set. Using default "
                    "value: "
                    << params.approximate_solution_position_threshold);
            }
            if (!nh_.getParam(ns_ + "approximate_solution_orientation_threshold",
                              params.approximate_solution_orientation_threshold)) {
                params.approximate_solution_orientation_threshold = 0.05;
                ROS_INFO_STREAM(
                    "Param approximate_solution_orientation_threshold was not set. Using default "
                    "value: "
                    << params.approximate_solution_orientation_threshold);
            }
            std::optional<double> approximate_solution_orientation_threshold = std::nullopt;
            if (test_rotation) {
                approximate_solution_orientation_threshold =
                    params.approximate_solution_orientation_threshold;
            }
            auto const approx_frame_tests =
                make_frame_tests(goal_frames,
                                 params.approximate_solution_position_threshold,
                                 approximate_solution_orientation_threshold);
            auto const tip_frames = fk_fn(solution);
            bool approx_solution_valid = true;
            for (size_t i = 0; i < approx_frame_tests.size(); ++i) {
                if (!approx_frame_tests[i](tip_frames[i])) {
                    approx_solution_valid = false;
                    break;
                }
            }
            if (!approx_solution_valid) {
                error_code.val = error_code.NO_IK_SOLUTION;
                solution = ik_seed_state;
            }
        }

        // Execute solution callback only on successful solution.
        auto const found_solution = error_code.val == error_code.SUCCESS;
        if (solution_callback && found_solution) {
            solution_callback(ik_poses.front(), solution, error_code);
        }

        return found_solution;
    }

    virtual std::vector<std::string> const& getJointNames() const { return joint_names_; }

    virtual std::vector<std::string> const& getLinkNames() const { return link_names_; }

    virtual bool getPositionFK(std::vector<std::string> const&,
                               std::vector<double> const&,
                               std::vector<geometry_msgs::Pose>&) const {
        return false;
    }

    virtual bool getPositionIK(geometry_msgs::Pose const&,
                               std::vector<double> const&,
                               std::vector<double>&,
                               moveit_msgs::MoveItErrorCodes&,
                               kinematics::KinematicsQueryOptions const&) const {
        return false;
    }

    virtual bool searchPositionIK(geometry_msgs::Pose const& ik_pose,
                                  std::vector<double> const& ik_seed_state,
                                  double timeout,
                                  std::vector<double>& solution,
                                  moveit_msgs::MoveItErrorCodes& error_code,
                                  kinematics::KinematicsQueryOptions const& options =
                                      kinematics::KinematicsQueryOptions()) const {
        return searchPositionIK(std::vector<geometry_msgs::Pose>{ik_pose},
                                ik_seed_state,
                                timeout,
                                std::vector<double>(),
                                solution,
                                IKCallbackFn(),
                                error_code,
                                options);
    }

    virtual bool searchPositionIK(geometry_msgs::Pose const& ik_pose,
                                  std::vector<double> const& ik_seed_state,
                                  double timeout,
                                  std::vector<double> const& consistency_limits,
                                  std::vector<double>& solution,
                                  moveit_msgs::MoveItErrorCodes& error_code,
                                  kinematics::KinematicsQueryOptions const& options =
                                      kinematics::KinematicsQueryOptions()) const {
        return searchPositionIK(std::vector<geometry_msgs::Pose>{ik_pose},
                                ik_seed_state,
                                timeout,
                                consistency_limits,
                                solution,
                                IKCallbackFn(),
                                error_code,
                                options);
    }

    virtual bool searchPositionIK(geometry_msgs::Pose const& ik_pose,
                                  std::vector<double> const& ik_seed_state,
                                  double timeout,
                                  std::vector<double>& solution,
                                  IKCallbackFn const& solution_callback,
                                  moveit_msgs::MoveItErrorCodes& error_code,
                                  kinematics::KinematicsQueryOptions const& options =
                                      kinematics::KinematicsQueryOptions()) const {
        return searchPositionIK(std::vector<geometry_msgs::Pose>{ik_pose},
                                ik_seed_state,
                                timeout,
                                std::vector<double>(),
                                solution,
                                solution_callback,
                                error_code,
                                options);
    }

    virtual bool searchPositionIK(geometry_msgs::Pose const& ik_pose,
                                  std::vector<double> const& ik_seed_state,
                                  double timeout,
                                  std::vector<double> const& consistency_limits,
                                  std::vector<double>& solution,
                                  IKCallbackFn const& solution_callback,
                                  moveit_msgs::MoveItErrorCodes& error_code,
                                  kinematics::KinematicsQueryOptions const& options =
                                      kinematics::KinematicsQueryOptions()) const {
        return searchPositionIK(std::vector<geometry_msgs::Pose>{ik_pose},
                                ik_seed_state,
                                timeout,
                                consistency_limits,
                                solution,
                                solution_callback,
                                error_code,
                                options);
    }

    virtual bool searchPositionIK(
        std::vector<geometry_msgs::Pose> const& ik_poses,
        std::vector<double> const& ik_seed_state,
        double timeout,
        std::vector<double> const& consistency_limits,
        std::vector<double>& solution,
        IKCallbackFn const& solution_callback,
        moveit_msgs::MoveItErrorCodes& error_code,
        kinematics::KinematicsQueryOptions const& options = kinematics::KinematicsQueryOptions(),
        moveit::core::RobotState const* context_state = NULL) const {
        return searchPositionIK(ik_poses,
                                ik_seed_state,
                                timeout,
                                consistency_limits,
                                solution,
                                solution_callback,
                                IKCostFn(),
                                error_code,
                                options,
                                context_state);
    }
};

}  // namespace pick_ik

PLUGINLIB_EXPORT_CLASS(pick_ik::PickIKPlugin, kinematics::KinematicsBase);
