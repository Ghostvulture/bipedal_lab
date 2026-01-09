"""
从URDF和STL文件中提取link的尺寸信息
"""

import numpy as np
import xml.etree.ElementTree as ET
import os


def parse_stl_ascii(file_path):
    """解析ASCII格式的STL文件"""
    vertices = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip().startswith('vertex'):
                coords = line.strip().split()[1:]
                vertices.append([float(x) for x in coords])
    return np.array(vertices)


def parse_stl_binary(file_path):
    """解析二进制格式的STL文件"""
    with open(file_path, 'rb') as f:
        # 跳过文件头（80字节）
        f.read(80)
        # 读取三角形数量
        num_triangles = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        
        vertices = []
        for _ in range(num_triangles):
            # 跳过法向量（12字节）
            f.read(12)
            # 读取3个顶点（每个顶点3个float，共36字节）
            for _ in range(3):
                vertex = np.frombuffer(f.read(12), dtype=np.float32)
                vertices.append(vertex)
            # 跳过属性字节（2字节）
            f.read(2)
    
    return np.array(vertices)


def parse_stl(file_path):
    """自动检测并解析STL文件"""
    try:
        # 尝试作为ASCII文件解析
        with open(file_path, 'r') as f:
            first_line = f.readline()
            if first_line.strip().startswith('solid'):
                return parse_stl_ascii(file_path)
    except:
        pass
    
    # 作为二进制文件解析
    return parse_stl_binary(file_path)


def get_bounding_box(vertices):
    """计算顶点的边界框"""
    if len(vertices) == 0:
        return None
    
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    
    dimensions = max_coords - min_coords
    center = (max_coords + min_coords) / 2
    
    return {
        'min': min_coords,
        'max': max_coords,
        'dimensions': dimensions,  # [length_x, width_y, height_z]
        'center': center
    }


def parse_urdf(urdf_path):
    """解析URDF文件获取link信息"""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    links_info = {}
    
    for link in root.findall('link'):
        link_name = link.get('name')
        
        # 获取视觉几何信息
        visual = link.find('visual')
        if visual is not None:
            geometry = visual.find('geometry')
            if geometry is not None:
                mesh = geometry.find('mesh')
                if mesh is not None:
                    mesh_file = mesh.get('filename')
                    links_info[link_name] = {
                        'mesh_file': mesh_file
                    }
        
        # 获取惯性信息
        inertial = link.find('inertial')
        if inertial is not None and link_name in links_info:
            origin = inertial.find('origin')
            mass = inertial.find('mass')
            
            if origin is not None:
                xyz = origin.get('xyz', '0 0 0').split()
                links_info[link_name]['com'] = [float(x) for x in xyz]
            
            if mass is not None:
                links_info[link_name]['mass'] = float(mass.get('value'))
    
    return links_info


def analyze_link_dimensions(urdf_path, stl_base_path, target_links=None):
    """
    分析指定link的尺寸
    
    Args:
        urdf_path: URDF文件路径
        stl_base_path: STL文件所在目录
        target_links: 要分析的link名称列表，None表示分析所有link
    """
    links_info = parse_urdf(urdf_path)
    
    print("=" * 80)
    print("Link尺寸分析")
    print("=" * 80)
    
    for link_name, info in links_info.items():
        # 如果指定了目标link，只处理目标link
        if target_links and link_name not in target_links:
            continue
        
        print(f"\n{link_name}:")
        print("-" * 80)
        
        # 打印质量和质心
        if 'mass' in info:
            print(f"  质量: {info['mass']:.3f} kg")
        if 'com' in info:
            print(f"  质心位置: [{info['com'][0]:.6f}, {info['com'][1]:.6f}, {info['com'][2]:.6f}] m")
        
        # 尝试读取STL文件
        mesh_file = info.get('mesh_file', '')
        if mesh_file:
            # 构建完整路径
            stl_path = os.path.join(stl_base_path, mesh_file)
            
            if os.path.exists(stl_path):
                try:
                    print(f"  STL文件: {mesh_file}")
                    vertices = parse_stl(stl_path)
                    bbox = get_bounding_box(vertices)
                    
                    if bbox:
                        print(f"  边界框最小值: [{bbox['min'][0]:.6f}, {bbox['min'][1]:.6f}, {bbox['min'][2]:.6f}] m")
                        print(f"  边界框最大值: [{bbox['max'][0]:.6f}, {bbox['max'][1]:.6f}, {bbox['max'][2]:.6f}] m")
                        print(f"  尺寸 [X, Y, Z]: [{bbox['dimensions'][0]:.6f}, {bbox['dimensions'][1]:.6f}, {bbox['dimensions'][2]:.6f}] m")
                        print(f"  几何中心: [{bbox['center'][0]:.6f}, {bbox['center'][1]:.6f}, {bbox['center'][2]:.6f}] m")
                        print(f"  顶点数量: {len(vertices)}")
                except Exception as e:
                    print(f"  错误: 无法解析STL文件 - {e}")
            else:
                print(f"  警告: STL文件不存在 - {stl_path}")


def get_link_length_from_joints(urdf_path, link1, link2):
    """
    从关节位置计算两个连接link之间的距离
    
    Args:
        urdf_path: URDF文件路径
        link1: 第一个link名称
        link2: 第二个link名称（应该是link1的子link）
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    for joint in root.findall('joint'):
        parent = joint.find('parent')
        child = joint.find('child')
        
        if parent is not None and child is not None:
            parent_link = parent.get('link')
            child_link = child.get('link')
            
            if parent_link == link1 and child_link == link2:
                origin = joint.find('origin')
                if origin is not None:
                    xyz = origin.get('xyz', '0 0 0').split()
                    xyz = np.array([float(x) for x in xyz])
                    distance = np.linalg.norm(xyz)
                    
                    print(f"\n关节 {joint.get('name')}:")
                    print(f"  父link: {parent_link}")
                    print(f"  子link: {child_link}")
                    print(f"  相对位置: [{xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f}] m")
                    print(f"  距离: {distance:.6f} m")
                    return xyz, distance
    
    return None, None


if __name__ == "__main__":
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "../usd_file/URDF/COD_2026_Balance_2_0.urdf")
    stl_base_path = os.path.join(current_dir, "../usd_file/URDF/COD_2026_Balance_2_0")
    
    # 分析指定的link
    target_links = ['Right_rear_link', 'Right_rear_child1_link']
    
    print("\n方法1: 从STL文件分析边界框")
    print("=" * 80)
    analyze_link_dimensions(urdf_path, stl_base_path, target_links)
    
    print("\n\n方法2: 从关节位置计算连杆长度")
    print("=" * 80)
    
    # 计算Right_rear_link的"长度"（从base_link到Right_rear_link的关节距离）
    print("\n从 base_link 到 Right_rear_link:")
    get_link_length_from_joints(urdf_path, 'base_link', 'Right_rear_link')
    
    # 计算Right_rear_child1_link的"长度"
    print("\n从 Right_rear_link 到 Right_rear_child1_link:")
    get_link_length_from_joints(urdf_path, 'Right_rear_link', 'Right_rear_child1_link')
    
    # 计算到轮子的距离
    print("\n从 Right_rear_child1_link 到 Right_Wheel_link:")
    get_link_length_from_joints(urdf_path, 'Right_rear_child1_link', 'Right_Wheel_link')
    
    print("\n" + "=" * 80)
    print("\n说明:")
    print("- 边界框尺寸给出了link在各个轴向的跨度")
    print("- X轴通常是长度方向")
    print("- Y轴通常是宽度方向")
    print("- Z轴通常是高度方向")
    print("- 关节位置给出了连杆之间的相对距离")
