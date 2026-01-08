"""测试机器人关节特性 - 零位、正方向、运动范围

这个脚本帮助你理解：
1. 关节的零位（0点）对应的物理姿态
2. 关节正方向的运动方向
3. 关节的运动范围限制

Usage:
    python user/test_code/test_joint_properties.py
    或
    ./isaaclab.sh -p user/test_code/test_joint_properties.py
"""

import argparse
import os
import torch

from isaaclab.app import AppLauncher

# 创建参数解析器
parser = argparse.ArgumentParser(description="Test robot joint properties.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""后面的所有代码在应用启动后执行"""

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg


def design_scene() -> tuple[dict, list[list[float]]]:
    """设计场景"""
    
    # 创建地面
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    
    # 创建灯光
    cfg_light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/Light", cfg_light)
    
    # 获取USD文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    usd_path = os.path.join(current_dir, "../usd_file/USD/COD-2026RoboMaster-Balance.usd")
    
    print(f"[INFO]: Loading robot from: {usd_path}")
    
    # 创建机器人原点
    origins = [[0.0, 0.0, 0.0]]
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    
    # 配置机器人
    robot_cfg = ArticulationCfg(
        prim_path="/World/Origin.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=80.0,
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )
    
    robot = Articulation(cfg=robot_cfg)
    
    scene_entities = {"robot": robot}
    return scene_entities, origins


def test_joint_properties(sim: sim_utils.SimulationContext, entities: dict, origins: torch.Tensor):
    """测试关节特性"""
    
    robot = entities["robot"]
    sim_dt = sim.get_physics_dt()
    
    # 打印基本信息
    print("\n" + "="*80)
    print("机器人关节信息")
    print("="*80)
    print(f"关节数量: {robot.num_joints}")
    print(f"关节名称: {robot.joint_names}")
    print("="*80 + "\n")
    
    # 重置到初始状态
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    
    # 运行几步让物理稳定
    for _ in range(100):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
    
    # ========== 测试1: 查看零位 ==========
    print("\n" + "="*80)
    print("测试 1: 关节零位（默认位置）")
    print("="*80)
    print("这是关节在初始状态下的位置值（单位：弧度）")
    print("观察机器人的物理姿态，这就是零位对应的姿态\n")
    
    robot.update(sim_dt)
    zero_positions = robot.data.joint_pos.clone()
    
    for i, (name, pos) in enumerate(zip(robot.joint_names, zero_positions[0])):
        print(f"  关节 {i}: {name:25s} = {pos.item():8.4f} rad ({pos.item()*180/3.14159:7.2f}°)")
    
    print("\n请观察机器人当前的物理姿态，这就是各个关节的零位姿态")
    print("按 Ctrl+C 可以随时退出\n")
    
    input(">>> 按回车继续测试关节运动范围...")
    
    # ========== 测试2: 测试每个关节的运动范围和正方向 ==========
    print("\n" + "="*80)
    print("测试 2: 关节运动范围和正方向")
    print("="*80)
    print("将逐个测试每个关节，观察运动方向和范围\n")
    
    for joint_idx in range(robot.num_joints):
        joint_name = robot.joint_names[joint_idx]
        
        print("\n" + "-"*80)
        print(f"测试关节 {joint_idx}: {joint_name}")
        print("-"*80)
        
        # 重置到零位
        robot.write_joint_state_to_sim(zero_positions, torch.zeros_like(zero_positions))
        for _ in range(50):
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim_dt)
        
        # 测试正向运动（+10度）
        print(f"\n1) 测试正向: 将关节从零位转动到 +10° (+0.1745 rad)")
        print("   观察机器人的运动方向...\n")
        
        target_pos = zero_positions.clone()
        target_pos[0, joint_idx] += 0.1745  # +10度
        
        # 使用位置控制测试
        test_actuator = ImplicitActuatorCfg(
            joint_names_expr=[joint_name],
            effort_limit=80.0,
            stiffness=100.0,  # 使用位置控制
            damping=10.0,
        )
        
        for step in range(200):
            # 设置目标位置
            robot.set_joint_position_target(target_pos[:, joint_idx], joint_ids=[joint_idx])
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim_dt)
            
            if step % 50 == 0:
                current_pos = robot.data.joint_pos[0, joint_idx].item()
                print(f"   Step {step:3d}: 当前位置 = {current_pos:8.4f} rad ({current_pos*180/3.14159:7.2f}°)")
        
        print(f"   ✓ 正向运动完成")
        
        # 等待用户确认
        input(f"   >>> 看到正向运动了吗？按回车继续测试负向...")
        
        # 测试负向运动（-10度）
        print(f"\n2) 测试负向: 将关节从零位转动到 -10° (-0.1745 rad)")
        print("   观察机器人的运动方向...\n")
        
        # 先回到零位
        robot.write_joint_state_to_sim(zero_positions, torch.zeros_like(zero_positions))
        for _ in range(50):
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim_dt)
        
        target_pos = zero_positions.clone()
        target_pos[0, joint_idx] -= 0.1745  # -10度
        
        for step in range(200):
            robot.set_joint_position_target(target_pos[:, joint_idx], joint_ids=[joint_idx])
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim_dt)
            
            if step % 50 == 0:
                current_pos = robot.data.joint_pos[0, joint_idx].item()
                print(f"   Step {step:3d}: 当前位置 = {current_pos:8.4f} rad ({current_pos*180/3.14159:7.2f}°)")
        
        print(f"   ✓ 负向运动完成")
        
        # 测试运动范围
        input(f"   >>> 按回车测试该关节的运动范围限制...")
        
        print(f"\n3) 测试运动范围限制")
        print("   尝试让关节转动到极限位置...\n")
        
        # 回到零位
        robot.write_joint_state_to_sim(zero_positions, torch.zeros_like(zero_positions))
        for _ in range(50):
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim_dt)
        
        # 测试正向极限
        print("   测试正向极限 (+180°)...")
        target_pos = zero_positions.clone()
        target_pos[0, joint_idx] += 3.14159  # +180度
        
        for step in range(300):
            robot.set_joint_position_target(target_pos[:, joint_idx], joint_ids=[joint_idx])
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim_dt)
        
        max_pos = robot.data.joint_pos[0, joint_idx].item()
        print(f"   正向极限位置: {max_pos:8.4f} rad ({max_pos*180/3.14159:7.2f}°)")
        
        # 回到零位
        robot.write_joint_state_to_sim(zero_positions, torch.zeros_like(zero_positions))
        for _ in range(50):
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim_dt)
        
        # 测试负向极限
        print("   测试负向极限 (-180°)...")
        target_pos = zero_positions.clone()
        target_pos[0, joint_idx] -= 3.14159  # -180度
        
        for step in range(300):
            robot.set_joint_position_target(target_pos[:, joint_idx], joint_ids=[joint_idx])
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim_dt)
        
        min_pos = robot.data.joint_pos[0, joint_idx].item()
        print(f"   负向极限位置: {min_pos:8.4f} rad ({min_pos*180/3.14159:7.2f}°)")
        
        print(f"\n   ✓ 运动范围: [{min_pos:8.4f}, {max_pos:8.4f}] rad")
        print(f"              [{min_pos*180/3.14159:7.2f}°, {max_pos*180/3.14159:7.2f}°]")
        
        # 回到零位
        robot.write_joint_state_to_sim(zero_positions, torch.zeros_like(zero_positions))
        for _ in range(50):
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim_dt)
        
        if joint_idx < robot.num_joints - 1:
            input(f"\n>>> 关节 {joint_name} 测试完成！按回车测试下一个关节...")
    
    # ========== 总结 ==========
    print("\n" + "="*80)
    print("测试完成！关节特性总结")
    print("="*80)
    
    # 最终再次显示所有关节信息
    robot.write_joint_state_to_sim(zero_positions, torch.zeros_like(zero_positions))
    for _ in range(50):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
    
    robot.update(sim_dt)
    
    print("\n所有关节的零位（默认位置）:")
    for i, (name, pos) in enumerate(zip(robot.joint_names, zero_positions[0])):
        print(f"  {i}. {name:25s} = {pos.item():8.4f} rad ({pos.item()*180/3.14159:7.2f}°)")
    
    print("\n说明:")
    print("  - 零位: 机器人加载时的默认关节位置")
    print("  - 正方向: 关节位置值增加时的运动方向")
    print("  - 运动范围: 关节可以运动的位置范围（受物理限制或URDF定义）")
    print("\n提示:")
    print("  - 如果是旋转关节，单位是弧度（rad）")
    print("  - 如果是滑动关节，单位是米（m）")
    print("  - 在强化学习中，通常观测值会减去零位，使得零位对应observation=0")
    print("="*80 + "\n")
    
    # 保持运行让用户观察
    print("仿真将继续运行，按 Ctrl+C 退出...")
    count = 0
    while simulation_app.is_running():
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        count += 1
        if count > 10000:
            break


def main():
    """主函数"""
    
    # 初始化仿真
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01,
        device=args_cli.device,
        gravity=(0.0, 0.0, -9.81),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # 设置相机视角
    sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, 0.5])
    
    # 设计场景
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    
    # 重置仿真
    sim.reset()
    
    print("\n" + "="*80)
    print("关节特性测试工具")
    print("="*80)
    print("此工具将帮助你理解机器人关节的:")
    print("  1. 零位 (0点) - 关节默认位置对应的物理姿态")
    print("  2. 正方向 - 关节位置值增加时的运动方向")
    print("  3. 运动范围 - 关节可以运动的最大范围")
    print("="*80 + "\n")
    
    # 运行测试
    try:
        test_joint_properties(sim, scene_entities, scene_origins)
    except KeyboardInterrupt:
        print("\n\n用户中断，退出...")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    simulation_app.close()
