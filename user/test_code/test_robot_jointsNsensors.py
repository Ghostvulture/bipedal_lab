"""测试机器人关节运动 - 使用扭矩控制

这个脚本演示如何：
1. 加载USD机器人文件作为Articulation（关节机器人）
2. 配置执行器（Actuators）来控制关节
3. 应用扭矩来测试关节是否能运动


Usage:
    ./isaaclab.sh -p user/test_code/test_robot_joints.py
"""

import argparse
import os

from matplotlib.pylab import rint

from isaaclab.app import AppLauncher



# 创建参数解析器
parser = argparse.ArgumentParser(description="Test robot joints with torque control.")
# 添加AppLauncher的命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()
# 启动Isaac Sim应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""后面的所有代码在应用启动后执行"""

import torch
import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ImuCfg, Imu
from VMC import VMCSolver

# VMC求解器实例化需要传入参数：motor_distance, L1, L2
# 根据URDF，这些参数需要从实际机器人测量得到
# 这里使用示例值，需要根据实际机器人调整
left_vmc = VMCSolver()
right_vmc = VMCSolver()

def quat_to_euler(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch = torch.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)

    return torch.stack((roll, pitch, yaw), dim=-1)



def design_scene() -> tuple[dict, list[list[float]]]:
    """设计场景：地面、灯光和机器人"""
    
    # 1. 创建地面
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    
    # 2. 创建灯光
    cfg_light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/Light", cfg_light)
    
    # 3. 获取USD文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    usd_path = os.path.join(current_dir, "../usd_file/USD/COD-2026RoboMaster-Balance.usd")
    
    print(f"[INFO]: Loading robot from: {usd_path}")
    
    # 4. 创建机器人的原点位置
    origins = [[0.0, 0.0, 0.0]]
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    
    # 5. 配置机器人的Articulation（关节系统）
    # 这是最关键的部分：定义机器人如何被控制
    robot_cfg = ArticulationCfg(
        # 机器人在场景中的路径（使用通配符.*可以匹配多个机器人）
        prim_path="/World/Origin.*/Robot",
        
        # spawn配置：如何生成机器人
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            
            # 刚体属性配置
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,  # 启用刚体物理
                max_linear_velocity=1000.0,  # 最大线速度
                max_angular_velocity=1000.0,  # 最大角速度
                max_depenetration_velocity=100.0,  # 最大去穿透速度
                enable_gyroscopic_forces=True,  # 启用陀螺效应
            ),
            
            # 关节系统根属性配置
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,  # 禁用自碰撞
                solver_position_iteration_count=4,  # 位置求解器迭代次数
                solver_velocity_iteration_count=1,  # 速度求解器迭代次数
            ),
            
            # 激活接触传感器（可以检测碰撞）
            activate_contact_sensors=True,
        ),
        
        # 初始状态配置
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),  # 机器人初始位置（抬高0.5米避免穿地）
            joint_pos = {'Left_front_joint':0.0,
                         'Left_rear_joint':0.0,
                         'Right_front_joint':0.0,
                         'Right_rear_joint':0.0,
                         },
        ),
        
        # 6. 执行器配置 - 这里定义如何控制关节
        # ImplicitActuator: 隐式执行器，使用PD控制器模拟真实电机
        actuators={
            # 定义一个执行器组，控制所有关节
            "all_joints": ImplicitActuatorCfg(
                # joint_names_expr: 正则表达式匹配关节名称
                # ".*" 表示匹配所有关节
                joint_names_expr=[".*"],
                
                # 扭矩限制（牛顿米）- 这限制了可以施加多大的力
                effort_limit_sim=80.0,
                
                # PD控制器参数：用于位置控制
                # stiffness: 刚度，控制位置误差产生的力
                # damping: 阻尼，控制速度产生的阻力
                stiffness=200.0,
                damping=5.0,
            ),
        },
    )


    
    # 7. 创建Articulation对象
    robot = Articulation(cfg=robot_cfg)
    
    # 8. 配置IMU传感器 - 放置在机器人base_link上
    imu_cfg = ImuCfg(
        # 传感器路径：放在base_link刚体上（从机器人的Body names中找到）
        prim_path="/World/Origin.*/Robot/base_link",
        
        # 更新频率：100Hz（每0.01秒更新一次）
        update_period=0.01,
        
        # 传感器相对于body的偏移（放在中心位置）
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),  # 位置偏移：中心
            rot=(1.0, 0.0, 0.0, 0.0),  # 姿态偏移（四元数: w,x,y,z）
        ),
        
        # 可视化传感器坐标系
        debug_vis=True,
    )
    
    # 创建IMU传感器对象
    imu_sensor = Imu(cfg=imu_cfg)
    
    # 返回场景实体和原点位置
    scene_entities = {
        "robot": robot,
        "imu": imu_sensor,  # 添加IMU传感器
    }
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """运行仿真循环"""
    
    # 获取机器人对象和IMU传感器
    robot = entities["robot"]
    imu_sensor = entities["imu"]
    
    # 获取仿真时间步长
    sim_dt = sim.get_physics_dt()
    count = 0
    
    # 指定要监控的关节
    target_joints = ['Left_front_joint']#'Left_front_joint', 'Left_rear_joint', 'Right_front_joint', 
    
    # 找到这些关节在joint_names中的索引
    joint_indices = []
    for target_joint in target_joints:
        if target_joint in robot.joint_names:
            joint_indices.append(robot.joint_names.index(target_joint))
    
    # 首次打印机器人信息
    print("\n" + "="*60)
    print("[INFO]: Robot Information")
    print("="*60)
    print(f"Number of robots: {robot.num_instances}")
    print(f"Number of bodies: {robot.num_bodies}")
    print(f"Number of joints: {robot.num_joints}")
    print(f"All joint names: {robot.joint_names}")
    print(f"\nMonitoring these joints: {target_joints}")
    print(f"Joint indices: {joint_indices}")
    print(f"Body names: {robot.body_names}")
    print("="*60 + "\n")
    
    # 显示joint 0的位置信息
    print("\n" + "="*60)
    print("[INFO]: 查看 joint 0 的位置信息")
    print("="*60)
    if len(joint_indices) > 0:
        joint_0_name = target_joints[0]
        joint_0_index = joint_indices[0]
        print(f"Joint 0 名称: {joint_0_name}")
        print(f"Joint 0 索引: {joint_0_index}")
        print(f"Joint 0 初始位置: {robot.data.joint_pos[0, joint_0_index].item():.4f} rad")
        print(f"Joint 0 初始速度: {robot.data.joint_vel[0, joint_0_index].item():.4f} rad/s")
    print("="*60 + "\n")
    print("[INFO]: 开始仿真，目标关节将保持在0位置...\n")
    
    # 主仿真循环
    while simulation_app.is_running():
        
        # 每500步重置一次机器人
        if count % 500 == 0:
            count = 0
            
            print(f"[INFO]: Resetting robot to initial state...")
            
            # 重置根状态（位置和速度）
            # 注意：需要加上origin偏移，因为状态是在世界坐标系中
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins  # 位置偏移
            robot.write_root_pose_to_sim(root_state[:, :7])  # 写入位置和姿态
            robot.write_root_velocity_to_sim(root_state[:, 7:])  # 写入线速度和角速度
            
            # 重置关节状态（位置和速度）
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            
            # 清除内部缓冲区
            robot.reset()
            imu_sensor.reset()  # 重置IMU传感器
        
        # 设置目标关节保持在0位置
        # 创建目标位置张量，默认保持当前位置
        joint_pos_target = robot.data.joint_pos.clone()
        
        # 将指定的6个关节设置为0位置
        for idx in joint_indices:
            joint_pos_target[0, idx] = 1.0#0.0
        
        # 设置关节位置目标（使用PD控制器自动计算所需扭矩）
        robot.set_joint_position_target(joint_pos_target)
        
        # 将数据写入仿真器
        robot.write_data_to_sim()
        
        # 执行一步仿真s
        sim.step()
        
        # 更新机器人的内部状态缓冲区
        # 这会从仿真器读取最新的状态（位置、速度等）
        robot.update(sim_dt)
        
        # 更新IMU传感器数据
        imu_sensor.update(sim_dt)

        # 使用VMC求解器计算关节力矩
        left_front_pos = robot.data.joint_pos[0, robot.joint_names.index('Left_front_joint')].item()
        right_front_pos = robot.data.joint_pos[0, robot.joint_names.index('Right_front_joint')].item()
        left_rear_pos = robot.data.joint_pos[0, robot.joint_names.index('Left_rear_joint')].item()
        right_rear_pos = robot.data.joint_pos[0, robot.joint_names.index('Right_rear_joint')].item()


        left_vmc.Resolve(torch.pi+right_rear_pos,-right_front_pos)
        right_vmc.Resolve(torch.pi-left_rear_pos, left_front_pos)
        
        # 每50步打印一次关节状态和IMU数据
        if count % 50 == 0 and count > 0:
            print(f"\n{'='*70}")
            print(f"[Step {count}] 机器人状态:")
            # 只打印目标关节的位置和速度
            print(f"  监控关节名称: {target_joints}")
            joint_positions = [robot.data.joint_pos[0, idx].item() for idx in joint_indices]
            joint_velocities = [robot.data.joint_vel[0, idx].item() for idx in joint_indices]
            print(f"  关节位置 (rad): {joint_positions}")
            print(f"  关节速度 (rad/s): {joint_velocities}")
            print(f"left front: {left_front_pos}, left rear: {torch.pi-left_rear_pos}, right front: {-right_front_pos}, right rear: {torch.pi+right_rear_pos}")
            print(f"  左侧VMC倒立摆长度: {left_vmc.GetPendulumLen():.4f} m")
            print(f"  左侧VMC倒立摆摆角: {left_vmc.GetPendulumRadian():.4f} rad")
            print(f"  右侧VMC倒立摆长度: {right_vmc.GetPendulumLen():.4f} m")
            print(f"  右侧VMC倒立摆摆角: {right_vmc.GetPendulumRadian():.4f} rad")


            # 打印IMU数据
            print(f"\n[Step {count}] IMU传感器数据:")
            print(f"  线性加速度 (m/s²): {imu_sensor.data.lin_acc_b[0].cpu().numpy()}")
            print(f"  角速度 (rad/s):     {imu_sensor.data.ang_vel_b[0].cpu().numpy()}")
            print(f"  姿态四元数 (w,x,y,z): {imu_sensor.data.quat_w[0].cpu().numpy()}")
            print(f"{'='*70}\n")

            euler = quat_to_euler(imu_sensor.data.quat_w[0])
            roll, pitch, yaw = euler.cpu().numpy()
            print(f"[Step {count}] IMU姿态欧拉角 (rad): Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}\n")

        count += 1


def main():
    """主函数"""
    
    # 初始化仿真上下文
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01,  # 仿真时间步长：10ms
        device=args_cli.device,  # 使用的设备（CPU或GPU）
        # 物理引擎配置
        gravity=(0.0, 0.0, -9.81),  # 重力加速度
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # 设置相机视角
    # 第一个参数是相机位置，第二个是相机看向的目标点
    sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, 0.5])
    
    # 设计场景（创建地面、灯光、机器人）
    scene_entities, scene_origins = design_scene()
    
    # 将原点位置转换为torch张量
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    
    # 重置仿真（这会初始化物理引擎）
    sim.reset()
    
    print("\n" + "="*60)
    print("[INFO]: Setup complete!")
    print("[INFO]: Starting joint motion test...")
    print("[INFO]: Watch the robot - joints should move if properly configured")
    print("="*60 + "\n")
    
    # 运行仿真器
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭应用
    simulation_app.close()
