import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    description_share = get_package_share_directory('manipulator_description')
    urdf_path = os.path.join(description_share, 'urdf', 'manipulator_gazebo.urdf')

    robot_description = ParameterValue(
        Command(['cat ', urdf_path]),
        value_type=str
    )


    share_parent = os.path.dirname(description_share)
    existing_resource_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    new_resource_path = share_parent if not existing_resource_path else f"{existing_resource_path}:{share_parent}"

    set_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=new_resource_path
    )

    # 1. robot_state_publisher — publishes TF tree from joint states
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}]
    )

    # 2. Launch Gazebo Sim (ros_gz_sim) with an empty world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'),
                         'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-r empty.sdf'}.items()
    )

    # 3. Spawn the robot into the running Gazebo world, reading from /robot_description
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description', '-name', 'manipulator', '-z', '0.05'],
        output='screen'
    )

    # 4. Bridge /clock so ROS 2 nodes use simulation time
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen'
    )


    load_joint_state_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'joint_state_broadcaster'],
        output='screen'
    )

    load_velocity_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'velocity_controller'],
        output='screen'
    )

    delayed_controllers = TimerAction(
        period=5.0,
        actions=[load_joint_state_broadcaster, load_velocity_controller]
    )

    return LaunchDescription([
        set_resource_path,
        robot_state_publisher_node,
        gazebo,
        clock_bridge,
        spawn_entity,
        delayed_controllers,
    ])