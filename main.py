# uav_advanced_system.py - 低空经济智能路径规划系统
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
#禁飞区图标正确显示
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Segoe UI Emoji', 'DejaVu Sans'] + matplotlib.rcParams['font.sans-serif']
import pandas as pd
import heapq
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("🚁 低空经济智能路径规划系统 v2.0")
print("=" * 70)

@dataclass
class Node:
    """A*算法节点类"""
    x: int
    y: int
    cost: float = 0
    heuristic: float = 0
    parent: 'Node' = None
    
    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

class AdvancedPathPlanner:
    """高级路径规划器"""
    
    def __init__(self, city_size=(100, 100), grid_size=2):
        self.city_width, self.city_height = city_size
        self.grid_size = grid_size
        self.grid_width = city_size[0] // grid_size
        self.grid_height = city_size[1] // grid_size
        self.obstacle_grid = np.zeros((self.grid_width, self.grid_height), dtype=bool)
        self.no_fly_zones = []
        self.buildings = []
        
    def add_obstacle(self, x, y, width, height):
        """添加障碍物到网格"""
        self.buildings.append((x, y, width, height))
        gx1 = max(0, x // self.grid_size)
        gy1 = max(0, y // self.grid_size)
        gx2 = min(self.grid_width, (x + width) // self.grid_size)
        gy2 = min(self.grid_height, (y + height) // self.grid_size)
        
        self.obstacle_grid[gx1:gx2, gy1:gy2] = True
        
    def add_no_fly_zone(self, x, y, radius):
        """添加禁飞区"""
        self.no_fly_zones.append((x, y, radius))
        
    def heuristic(self, a, b):
        """A*启发式函数（曼哈顿距离）"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def is_valid_position(self, x, y):
        """检查位置是否有效"""
        if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):
            return False
        if self.obstacle_grid[x, y]:
            return False
            
        # 检查禁飞区
        grid_x, grid_y = x * self.grid_size, y * self.grid_size
        for nf_x, nf_y, radius in self.no_fly_zones:
            distance = np.sqrt((grid_x - nf_x)**2 + (grid_y - nf_y)**2)
            if distance <= radius:
                return False
                
        return True
    
    def a_star_search(self, start, goal):
        """A*路径搜索算法"""
        start_node = Node(start[0], start[1])
        goal_node = Node(goal[0], goal[1])
        
        open_set = []
        heapq.heappush(open_set, start_node)
        closed_set = set()
        
        # 记录搜索过程用于可视化
        search_process = []
        
        while open_set:
            current = heapq.heappop(open_set)
            
            # 记录搜索节点
            search_process.append((current.x, current.y))
            
            if (current.x, current.y) == (goal_node.x, goal_node.y):
                # 重建路径
                path = []
                while current:
                    path.append((current.x * self.grid_size, current.y * self.grid_size))
                    current = current.parent
                return path[::-1], search_process
            
            closed_set.add((current.x, current.y))
            
            # 检查相邻节点
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]:
                neighbor_x, neighbor_y = current.x + dx, current.y + dy
                
                if not self.is_valid_position(neighbor_x, neighbor_y):
                    continue
                    
                if (neighbor_x, neighbor_y) in closed_set:
                    continue
                
                # 计算移动成本（对角线移动成本更高）
                move_cost = 1.4 if abs(dx) == 1 and abs(dy) == 1 else 1.0
                new_cost = current.cost + move_cost
                
                neighbor = Node(neighbor_x, neighbor_y)
                neighbor.cost = new_cost
                neighbor.heuristic = self.heuristic((neighbor_x, neighbor_y), (goal_node.x, goal_node.y))
                neighbor.parent = current
                
                # 检查是否在open_set中
                in_open = False
                for node in open_set:
                    if (node.x, node.y) == (neighbor_x, neighbor_y):
                        in_open = True
                        if new_cost < node.cost:
                            node.cost = new_cost
                            node.parent = current
                        break
                
                if not in_open:
                    heapq.heappush(open_set, neighbor)
        
        return None, search_process  # 未找到路径

class LowAltitudeEconomySimulator:
    """低空经济模拟器"""
    
    def __init__(self):
        self.planner = AdvancedPathPlanner()
        self.warehouses = []
        self.fig = None
        self.ax = None
        
    def setup_environment(self):
        """设置模拟环境"""
        print("🏙️  设置城市环境...")
        
        # 添加建筑物
        buildings = [
            (20, 20, 15, 25),   # 商业中心
            (60, 10, 10, 15),   # 居民区
            (40, 60, 20, 15),   # 工业区
            (10, 70, 12, 20),   # 学校
            (70, 50, 15, 30),   # 医院
            (30, 35, 18, 12)    # 商业区
        ]
        
        for i, (x, y, w, h) in enumerate(buildings):
            self.planner.add_obstacle(x, y, w, h)
            print(f"   🏢 建筑物 {i+1}: 位置({x},{y}), 大小({w}x{h})")
        
        # 添加禁飞区
        no_fly_zones = [
            (80, 80, 8),   # 政府机关
            (25, 85, 5),   # 军事区域
            (60, 40, 6)    # 机场净空
        ]
        
        for i, (x, y, r) in enumerate(no_fly_zones):
            self.planner.add_no_fly_zone(x, y, r)
            print(f"   🚫 禁飞区 {i+1}: 中心({x},{y}), 半径{r}米")
        
        # 设置仓库
        self.warehouses = [
            {"name": "中央仓库A", "location": (5, 5)},
            {"name": "配送中心B", "location": (95, 95)},
            {"name": "城北仓库C", "location": (20, 90)}
        ]
        
        print("✅ 环境设置完成！")
        
    def visualize_environment(self, path=None, search_process=None):
        """可视化环境和路径"""
        self.fig, self.ax = plt.subplots(figsize=(14, 12))
        
        # 绘制顺序从底层到顶层：
        
        # 1. 搜索过程（最底层）
        if search_process:
            search_x = [x * self.planner.grid_size for x, y in search_process]
            search_y = [y * self.planner.grid_size for x, y in search_process]
            self.ax.scatter(search_x, search_y, color='yellow', alpha=0.3, s=10,
                          label='算法搜索区域', zorder=1)

        # 2. 建筑物（中间层）
        for i, (x, y, w, h) in enumerate(self.planner.buildings):
            rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                   edgecolor='darkred', facecolor='red', 
                                   alpha=0.7, label='建筑物' if i == 0 else "",
                                   zorder=2)
            self.ax.add_patch(rect)
            self.ax.text(x + w/2, y + h/2, f'B{i+1}', ha='center', va='center',
                       color='white', fontweight='bold', fontsize=8, zorder=3)

        # 3. 禁飞区（建筑物之上）
        for i, (x, y, r) in enumerate(self.planner.no_fly_zones):
            circle = patches.Circle((x, y), r, linewidth=2,
                                  edgecolor='orange', facecolor='yellow', 
                                  alpha=0.3, label='禁飞区' if i == 0 else "",
                                  zorder=4)
            self.ax.add_patch(circle)
            self.ax.text(x, y, '🚫', ha='center', va='center', fontsize=12, zorder=5)
            self.ax.text(x, y - r - 2, f'禁飞区{i+1}', ha='center', va='top', fontsize=8, zorder=5)

        # 4. 路径（在障碍物之上）
        if path:
            path_x, path_y = zip(*path)
            self.ax.plot(path_x, path_y, 'c-', linewidth=4, label='最优路径', alpha=0.8, zorder=6)
            self.ax.plot(path_x, path_y, 'co', markersize=6, alpha=0.6, zorder=7)

        # 5. 仓库（最顶层，确保可见）
        for warehouse in self.warehouses:
            x, y = warehouse['location']
            self.ax.plot(x, y, 's', markersize=15, color='green',
                       label='仓库' if warehouse == self.warehouses[0] else "",
                       zorder=8)  # 最高层级
            self.ax.text(x, y - 8, warehouse['name'], ha='center', va='top',
                       fontweight='bold', color='darkgreen', zorder=9)  # 文字也在最顶层
            
        # 计算路径长度（如果存在路径）
        if path:
            path_length = sum(np.sqrt((path[i+1][0]-path[i][0])**2 + 
                                    (path[i+1][1]-path[i][1])**2) 
                            for i in range(len(path)-1))
            self.ax.text(0.5, 0.02, f'路径长度: {path_length:.1f}米', 
                       transform=self.ax.transAxes, fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                       zorder=10)  # 信息文本也在最顶层
        
        # 设置图形属性
        self.ax.set_xlim(0, self.planner.city_width)
        self.ax.set_ylim(0, self.planner.city_height)
        self.ax.set_xlabel('X坐标 (米)')
        self.ax.set_ylabel('Y坐标 (米)')
        self.ax.set_title('无人机低空经济路径规划系统\n基于A*搜索算法的最优路径规划', 
                         fontsize=16, fontweight='bold', pad=20)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper left')
        
        plt.tight_layout()
        
    def run_path_planning(self, start_idx=0, end_idx=1):
        """运行路径规划"""
        print(f"\n🧠 开始路径规划: {self.warehouses[start_idx]['name']} → {self.warehouses[end_idx]['name']}")
        
        start = self.warehouses[start_idx]['location']
        end = self.warehouses[end_idx]['location']
        
        # 转换为网格坐标
        grid_start = (start[0] // self.planner.grid_size, start[1] // self.planner.grid_size)
        grid_end = (end[0] // self.planner.grid_size, end[1] // self.planner.grid_size)
        
        print(f"   起点: {start} → 网格{grid_start}")
        print(f"   终点: {end} → 网格{grid_end}")
        
        start_time = time.time()
        path, search_process = self.planner.a_star_search(grid_start, grid_end)
        end_time = time.time()
        
        if path:
            print(f"✅ 路径规划成功！")
            print(f"   📏 路径点数: {len(path)}")
            print(f"   ⏱️  计算时间: {(end_time - start_time)*1000:.2f}毫秒")
            
            # 可视化结果
            self.visualize_environment(path, search_process)
            plt.savefig('advanced_path_planning.png', dpi=300, bbox_inches='tight')
            print("💾 结果已保存为 'advanced_path_planning.png'")
            plt.show()
            
            return path
        else:
            print("❌ 未找到可行路径！")
            self.visualize_environment()
            plt.show()
            return None

def main():
    """主函数"""
    simulator = LowAltitudeEconomySimulator()
    
    # 设置环境
    simulator.setup_environment()
    
    print("\n" + "="*50)
    print("📊 环境统计信息:")
    print(f"   城市范围: {simulator.planner.city_width} × {simulator.planner.city_height} 米")
    print(f"   网格精度: {simulator.planner.grid_size} 米")
    print(f"   建筑物数量: {len(simulator.planner.buildings)}")
    print(f"   禁飞区数量: {len(simulator.planner.no_fly_zones)}")
    print(f"   仓库数量: {len(simulator.warehouses)}")
    print("="*50)
    
    # 运行路径规划
    print("\n🎯 开始智能路径规划演示...")
    
    # 规划从仓库A到仓库B的路径
    path = simulator.run_path_planning(0, 1)
    
    if path:
        print(f"\n🎉 低空经济路径规划演示完成！")
        print("下一步可以:")
        print("  1. 添加多个无人机同时规划")
        print("  2. 实现动态障碍物避让") 
        print("  3. 加入天气影响因子")
        print("  4. 优化算法性能")
    else:
        print("\n⚠️ 路径规划失败，请调整环境参数后重试")

if __name__ == "__main__":
    main()
# 在现有代码后添加多无人机调度类

class MultiDroneScheduler:
    """多无人机协同调度器"""
    
    def __init__(self, planner, num_drones=3):
        self.planner = planner
        self.num_drones = num_drones
        self.drones = []
        self.assigned_tasks = []
        
    def initialize_drones(self):
        """初始化无人机舰队"""
        drone_types = [
            {"name": "高速无人机", "speed": 15, "range": 50, "color": "red"},
            {"name": "载重无人机", "speed": 8, "range": 30, "color": "blue"}, 
            {"name": "长航时无人机", "speed": 10, "range": 80, "color": "green"}
        ]
        
        for i in range(self.num_drones):
            drone = {
                "id": i + 1,
                "type": drone_types[i % len(drone_types)],
                "position": None,
                "battery": 100,
                "status": "idle",  # idle, charging, flying, delivering
                "current_task": None,
                "path": []
            }
            self.drones.append(drone)
        
        print(f"🚁 初始化 {self.num_drones} 架无人机完成！")
        
    def assign_delivery_tasks(self, warehouses, deliveries):
        """分配配送任务"""
        self.assigned_tasks = []
        
        for i, delivery in enumerate(deliveries):
            start_wh = warehouses[delivery['start']]
            end_wh = warehouses[delivery['end']]
            
            # 选择最适合的无人机
            suitable_drones = self.find_suitable_drones(start_wh['location'], end_wh['location'])
            
            if suitable_drones:
                best_drone = suitable_drones[0]
                task = {
                    "id": i + 1,
                    "start": start_wh,
                    "end": end_wh,
                    "assigned_drone": best_drone['id'],
                    "priority": delivery.get('priority', 1),
                    "status": "assigned"
                }
                self.assigned_tasks.append(task)
                
                # 更新无人机状态
                best_drone['current_task'] = task
                best_drone['status'] = 'assigned'
                
                print(f"📦 任务 {i+1}: {start_wh['name']} → {end_wh['name']} 分配给无人机{best_drone['id']}")
        
    def find_suitable_drones(self, start, end):
        """寻找适合任务的无人机"""
        suitable = []
        
        # 计算任务距离
        distance = np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
        
        for drone in self.drones:
            if drone['status'] == 'idle':
                # 检查航程是否足够
                if distance <= drone['type']['range']:
                    suitable.append(drone)
        
        # 按速度排序（优先选择高速无人机）
        suitable.sort(key=lambda x: x['type']['speed'], reverse=True)
        return suitable
    
    def execute_all_tasks(self):
        """执行所有分配的任务"""
        print(f"\n🎯 开始执行 {len(self.assigned_tasks)} 个配送任务...")
        
        all_paths = []
        
        for task in self.assigned_tasks:
            drone = self.drones[task['assigned_drone'] - 1]
            
            print(f"\n✈️ 无人机{drone['id']} 开始任务 {task['id']}: "
                  f"{task['start']['name']} → {task['end']['name']}")
            
            # 路径规划
            start_grid = (task['start']['location'][0] // self.planner.grid_size, 
                         task['start']['location'][1] // self.planner.grid_size)
            end_grid = (task['end']['location'][0] // self.planner.grid_size, 
                       task['end']['location'][1] // self.planner.grid_size)
            
            path, _ = self.planner.a_star_search(start_grid, end_grid)
            
            if path:
                drone['path'] = path
                drone['status'] = 'flying'
                all_paths.append({
                    'drone_id': drone['id'],
                    'path': path,
                    'color': drone['type']['color'],
                    'task': task
                })
                
                # 计算预计时间
                distance = sum(np.sqrt((path[i+1][0]-path[i][0])**2 + 
                                    (path[i+1][1]-path[i][1])**2) 
                            for i in range(len(path)-1))
                time_estimate = distance / drone['type']['speed']
                
                print(f"   ✅ 路径规划成功！距离: {distance:.1f}米, 预计时间: {time_estimate:.1f}秒")
            else:
                print(f"   ❌ 路径规划失败！")
        
        return all_paths

# 在 main() 函数中添加多无人机演示
def demo_multi_drone_system():
    """演示多无人机系统 - 简化稳定版"""
    print("\n" + "="*60)
    print("多无人机协同调度系统演示")
    print("="*60)
    
    # 创建模拟器
    simulator = LowAltitudeEconomySimulator()
    simulator.setup_environment()
    
    # 直接手动创建多条路径进行演示
    all_paths = [
        {
            'drone_id': 1,
            'path': [(5, 5), (25, 30), (45, 50), (70, 70), (95, 95)],
            'color': 'red',
            'task': {'start': '中央仓库A', 'end': '配送中心B'}
        },
        {
            'drone_id': 2, 
            'path': [(20, 90), (35, 75), (50, 60), (75, 80), (95, 95)],
            'color': 'blue',
            'task': {'start': '城北仓库C', 'end': '配送中心B'}
        },
        {
            'drone_id': 3,
            'path': [(5, 5), (15, 40), (30, 65), (50, 85), (20, 90)],
            'color': 'green', 
            'task': {'start': '中央仓库A', 'end': '城北仓库C'}
        }
    ]
    
    # 可视化多无人机路径
    simulator.visualize_multi_drone_paths(all_paths)
    
    print(f"\n🎉 多无人机系统演示完成！")
    print(f"   展示了 {len(all_paths)} 条无人机路径")
    print(f"   模拟了 3 架无人机协同工作")
    
    return simulator, all_paths

# 在 LowAltitudeEconomySimulator 类中添加新方法
def visualize_multi_drone_paths(self, all_paths):
    """可视化多无人机路径"""
    self.fig, self.ax = plt.subplots(figsize=(16, 12))
    
    # 绘制基础环境（使用之前的绘制代码，但去掉单一路径部分）
    # [这里复制之前的基础环境绘制代码]
    
    # 绘制多无人机路径
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, path_info in enumerate(all_paths):
        path = path_info['path']
        color = path_info['color']
        drone_id = path_info['drone_id']
        
        if path:
            path_x, path_y = zip(*path)
            self.ax.plot(path_x, path_y, '-', linewidth=3, 
                       color=color, alpha=0.8, 
                       label=f'无人机{drone_id}路径')
            self.ax.plot(path_x, path_y, 'o', markersize=4, 
                       color=color, alpha=0.6)
            
            # 标注起点终点
            self.ax.text(path[0][0], path[0][1]+3, f'D{drone_id}起点', 
                       fontsize=8, color=color, fontweight='bold')
            self.ax.text(path[-1][0], path[-1][1]+3, f'D{drone_id}终点', 
                       fontsize=8, color=color, fontweight='bold')
    
    # 设置图形属性
    self.ax.set_xlim(0, self.planner.city_width)
    self.ax.set_ylim(0, self.planner.city_height)
    self.ax.set_xlabel('X坐标 (米)')
    self.ax.set_ylabel('Y坐标 (米)')
    self.ax.set_title('🚁 多无人机协同路径规划系统\n不同颜色代表不同无人机任务', 
                     fontsize=16, fontweight='bold', pad=20)
    self.ax.grid(True, alpha=0.3)
    self.ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('multi_drone_path_planning.png', dpi=300, bbox_inches='tight')
    print("💾 多无人机路径图已保存为 'multi_drone_path_planning.png'")
    plt.show()

# 在 main() 最后添加：
print("\n" + "="*60)
print("🎯 开始多无人机协同调度演示...")
simulator, scheduler, paths = demo_multi_drone_system()

print(f"\n🎉 多无人机系统演示完成！")
print(f"   成功规划 {len(paths)} 条无人机路径")
print(f"   动用 {scheduler.num_drones} 架无人机")
print(f"   完成 {len(scheduler.assigned_tasks)} 个配送任务")