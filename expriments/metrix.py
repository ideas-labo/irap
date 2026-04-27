# from KM import *
from IRAPE.KM import *


def points_to_f(points, x):
    """
    将浮点数列表变成一个分段线性函数，返回给定x对应的y值
    
    Args:
        points: 点列表，每个点是包含两个元素[x, y]的列表或元组，按x坐标升序排列
        x: 自变量x的值
    
    Returns:
        x对应的因变量y的值
    """
    if not points:
        return 0.0  # 或者可以根据需求返回None或其他值
    
    if len(points) == 1:
        return points[0][1]
    
    min_x = points[0][0]
    max_x = points[-1][0]
    
    # 如果x小于最小x值，返回第一个点的y值
    if x < min_x:
        return points[0][1]
    
    # 如果x大于最大x值，返回最后一个点的y值
    if x > max_x:
        return points[-1][1]
    
    # 如果x在points范围内，找到包含x的线段并进行线性插值
    # 遍历相邻的点对
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        
        if x1 <= x <= x2:
            # 线性插值: y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
            if x2 == x1:  # 避免除零错误，这种情况应该返回y1或y2
                return y1
            y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
            return y
    
    # 理论上不应该到达这里，但为了安全
    return points[-1][1]

def euclidean_dis(point1, point2):
    """
    计算欧式距离
    """
    return math.sqrt((point1[0] - point2[0]) * (point1[0] - point2[0]) + (point1[1] - point2[1]) * (point1[1] - point2[1]))

def integral_area(points, l, r):
    """
    计算points所对应的分段线性函数与x轴围成的面积，定义域为[l,r]
    
    Args:
        points: 点列表，每个点是包含两个元素[x, y]的列表或元组，按x坐标升序排列
        l: 积分的左边界
        r: 积分的右边界
    
    Returns:
        与x轴围成的面积（绝对值）
    """
    if not points or l >= r:
        return 0.0
    
    # 确保积分区间在[l, r]内
    total_area = 0.0
    
    # 获取points的x坐标范围
    min_x = points[0][0]
    max_x = points[-1][0]
    
    # 情况1: 积分区间完全在points定义域左侧
    if r <= min_x:
        # 在[l, r]区间内，y值恒为points[0][1]
        return abs((r - l) * points[0][1])
    
    # 情况2: 积分区间完全在points定义域右侧
    if l >= max_x:
        # 在[l, r]区间内，y值恒为points[-1][1]
        return abs((r - l) * points[-1][1])
    
    # 情况3: 积分区间与points定义域有重叠或跨越
    # 分为三部分：左侧外推部分、中间分段线性部分、右侧外推部分
    
    # 左侧外推部分：[l, min(max(l, min_x), r)]
    if l < min_x:
        left_end = min(min_x, r)
        total_area += abs((left_end - l) * points[0][1])
    
    # 右侧外推部分：[max(min(r, max_x), l), r]
    if r > max_x:
        right_start = max(max_x, l)
        total_area += abs((r - right_start) * points[-1][1])
    
    # 中间分段线性部分：在[min_x, max_x]与[l, r]的交集内
    mid_l = max(l, min_x)
    mid_r = min(r, max_x)
    
    if mid_l < mid_r:
        # 找到在[mid_l, mid_r]区间内的所有点（包括边界点）
        # 需要找到包含区间[mid_l, mid_r]的点对
        
        # 找到第一个x坐标 >= mid_l的点的索引
        start_idx = 0
        for i in range(len(points)):
            if points[i][0] >= mid_l:
                start_idx = i
                break
        if start_idx > 0 and points[start_idx][0] > mid_l:
            start_idx -= 1  # 从左边相邻的点开始
        
        # 找到最后一个x坐标 <= mid_r的点的索引
        end_idx = len(points) - 1
        for i in range(len(points) - 1, -1, -1):
            if points[i][0] <= mid_r:
                end_idx = i
                break
        if end_idx < len(points) - 1 and points[end_idx][0] < mid_r:
            end_idx += 1  # 包含右边相邻的点
        
        # 确保索引范围有效
        start_idx = max(0, start_idx)
        end_idx = min(len(points) - 1, end_idx)
        
        # 计算中间部分的面积
        for i in range(start_idx, min(end_idx, len(points) - 1)):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            
            # 确定当前线段在[mid_l, mid_r]内的有效区间
            seg_l = max(x1, mid_l)
            seg_r = min(x2, mid_r)
            
            if seg_l < seg_r:
                # 计算线段在[seg_l, seg_r]上与x轴围成的面积
                # 线段方程: y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
                
                # 在seg_l处的y值
                y_at_seg_l = y1 + (y2 - y1) * (seg_l - x1) / (x2 - x1) if x2 != x1 else y1
                # 在seg_r处的y值  
                y_at_seg_r = y1 + (y2 - y1) * (seg_r - x1) / (x2 - x1) if x2 != x1 else y1
                
                # 梯形面积 = (上底 + 下底) * 高 / 2
                # 这里是与x轴围成的面积，需要考虑函数值的正负
                # 面积 = |梯形面积|，即计算与x轴之间的绝对面积
                
                # 梯形面积（带符号）
                trap_signed_area = (y_at_seg_l + y_at_seg_r) * (seg_r - seg_l) / 2.0
                # 取绝对值
                trap_area = abs(trap_signed_area)
                
                total_area += trap_area
    
    return total_area

def project_x(points, l, r):
    """
    将点列表中所有点的x值投影到区间[0,1]中
    
    Args:
        points: 点列表，每个点是包含两个元素[x, y]的列表或元组
        l: 原始x值的左边界
        r: 原始x值的右边界
    
    Returns:
        投影后的新点列表，不修改原始列表
    """
    import copy
    
    # 深拷贝原始列表以避免修改
    projected_points = copy.deepcopy(points)
    
    # 检查边界是否有效
    if l == r:
        # 如果左右边界相等，将所有点的x坐标设为0.5（中间值）
        for point in projected_points:
            if len(point) >= 2:
                point[0] = 0.5
    else:
        # 对每个点进行线性投影
        for point in projected_points:
            if len(point) >= 2:
                # 线性投影公式: (x - l) / (r - l) -> [0, 1]
                original_x = point[0]
                projected_x = (original_x - l) / (r - l)
                # 确保投影后的值在[0, 1]范围内
                projected_x = max(0.0, min(1.0, projected_x))
                point[0] = projected_x
    
    return projected_points

def p_to_p_dis(list1 : list, list2 : list):
    """
    求两个点列表的点对点平均距离
    """

    # 求定义域
    l = min(list1[0][0], list2[0][0])
    r = max(list1[-1][0], list2[-1][0])

    # 将定义域映射到[0,1]
    norm_list1 = project_x(list1, l, r)
    norm_list2 = project_x(list2, l, r)

    _, match = KM(norm_list1, norm_list2)

    s = 0.0
    for i in range(len(match)):
        if match[i][1] != -1:
            s += euclidean_dis(norm_list1[i], norm_list2[match[i][1]]) # 累加配对点的欧式距离
    
    s += abs(len(list1) - len(list2)) # 点数不一致施加惩罚

    s /= min(len(list1), len(list2)) # 求平均点对点距离

    return s

def area_dis(list1 : list, list2 : list):
    """
    求两个点列表所对应的分段线性函数的积分面积差异
    """

    # 求定义域
    l = min(list1[0][0], list2[0][0])
    r = max(list1[-1][0], list2[-1][0])

    # 将定义域映射到[0,1]
    norm_list1 = project_x(list1, l, r)
    norm_list2 = project_x(list2, l, r) 

    return abs(integral_area(norm_list1, 0, 1) - integral_area(norm_list2, 0, 1))

def chebyshev_dis(list1: list, list2: list):
    """
    计算两个分段线性函数之间的最大偏差（Chebyshev距离）
    即 max_x |f1(x) - f2(x)|
    
    Args:
        list1: 第一个点列表
        list2: 第二个点列表
    
    Returns:
        两个函数之间的最大垂直距离
    """
    if not list1 and not list2:
        return 0.0
    
    if not list1:
        # 如果list1为空，f1(x)可以视为恒为0
        max_val = 0.0
        for point in list2:
            if len(point) >= 2:
                max_val = max(max_val, abs(point[1]))
        return max_val
    
    if not list2:
        # 如果list2为空，f2(x)可以视为恒为0  
        max_val = 0.0
        for point in list1:
            if len(point) >= 2:
                max_val = max(max_val, abs(point[1]))
        return max_val
    
    # 获取两个列表的x范围
    x_min1, x_max1 = list1[0][0], list1[-1][0]
    x_min2, x_max2 = list2[0][0], list2[-1][0]
    
    # 合并所有关键x点：两个列表的端点和交点
    all_x_points = set()
    
    # 添加两个列表的所有x坐标点
    for point in list1:
        all_x_points.add(point[0])
    for point in list2:
        all_x_points.add(point[0])
    
    # 添加区间端点
    all_x_points.add(min(x_min1, x_min2))  # 整体最左
    all_x_points.add(max(x_max1, x_max2))  # 整体最右
    
    # 添加重叠区域的边界点
    overlap_left = max(x_min1, x_min2)
    overlap_right = min(x_max1, x_max2)
    if overlap_left <= overlap_right:
        all_x_points.add(overlap_left)
        all_x_points.add(overlap_right)
    
    all_x_points = sorted(list(all_x_points))
    
    max_diff = 0.0
    
    # 在所有关键点和它们之间的中点处计算差异
    for x in all_x_points:
        y1 = points_to_f(list1, x)
        y2 = points_to_f(list2, x)
        diff = abs(y1 - y2)
        max_diff = max(max_diff, diff)
    
    # 为了更精确地找到最大值，还需要检查相邻关键点之间的最大差异
    # 对于分段线性函数，最大差异可能出现在关键点处，但我们也需要考虑线段内部
    for i in range(len(all_x_points) - 1):
        x_left = all_x_points[i]
        x_right = all_x_points[i + 1]
        
        # 在区间内部取一些采样点
        # 由于是分段线性函数，最大值会在端点处取得，但为了完整性，我们可以检查中点
        x_mid = (x_left + x_right) / 2.0
        y1_mid = points_to_f(list1, x_mid)
        y2_mid = points_to_f(list2, x_mid)
        diff_mid = abs(y1_mid - y2_mid)
        max_diff = max(max_diff, diff_mid)
        
        # 还可以检查每个原始线段的端点
        # 实际上由于分段线性函数的性质，最大值会在我们已经添加的点处取得
        # 但为了确保不会遗漏，我们可以更密集地采样
        # 这里我们检查区间三等分点
        x_third1 = x_left + (x_right - x_left) / 3.0
        x_third2 = x_left + 2 * (x_right - x_left) / 3.0
        
        y1_third1 = points_to_f(list1, x_third1)
        y2_third1 = points_to_f(list2, x_third1)
        diff_third1 = abs(y1_third1 - y2_third1)
        max_diff = max(max_diff, diff_third1)
        
        y1_third2 = points_to_f(list1, x_third2)
        y2_third2 = points_to_f(list2, x_third2)
        diff_third2 = abs(y1_third2 - y2_third2)
        max_diff = max(max_diff, diff_third2)
    
    return max_diff

def RMSE_dis(list1 : list, list2 : list):
    # 求定义域
    l = min(list1[0][0], list2[0][0])
    r = max(list1[-1][0], list2[-1][0])

    # 将定义域映射到[0,1]
    norm_list1 = project_x(list1, l, r)
    norm_list2 = project_x(list2, l, r)
    
    # 在[0,1]区间内采样10个点: 0.1, 0.2, ..., 1.0
    sample_points = [i * 0.1 for i in range(1, 11)]
    
    # 计算差异的平方和
    sum_squared_diff = 0.0
    n = len(sample_points)
    
    for x in sample_points:
        y1 = points_to_f(norm_list1, x)
        y2 = points_to_f(norm_list2, x)
        diff = y1 - y2
        sum_squared_diff += diff * diff
    
    # 计算均方根误差
    rmse = (sum_squared_diff / n) ** 0.5
    return rmse

def test_matrix():
    print("开始测试所有距离函数...")
    
    # 测试 points_to_f 函数
    print("\n=== 测试 points_to_f 函数 ===")
    points = [[0, 0], [1, 1], [2, 0]]
    print(f"点列表: {points}")
    print(f"x=0.5 -> y={points_to_f(points, 0.5)}")  # 应该返回0.5
    print(f"x=-1 -> y={points_to_f(points, -1)}")    # 应该返回0 (外推)
    print(f"x=3 -> y={points_to_f(points, 3)}")      # 应该返回0 (外推)
    print(f"x=1.5 -> y={points_to_f(points, 1.5)}")  # 应该返回0.5
    
    # 测试 euclidean_dis 函数
    print("\n=== 测试 euclidean_dis 函数 ===")
    point1 = [0, 0]
    point2 = [3, 4]
    dist = euclidean_dis(point1, point2)
    print(f"点{point1}和点{point2}之间的距离: {dist}")  # 应该是5.0
    point3 = [1, 1]
    point4 = [4, 5]
    dist2 = euclidean_dis(point3, point4)
    print(f"点{point3}和点{point4}之间的距离: {dist2}")  # 应该是5.0
    
    # 测试 integral_area 函数
    print("\n=== 测试 integral_area 函数 ===")
    points1 = [[0, 1], [1, 1]]  # 矩形，面积应该是1
    area1 = integral_area(points1, 0, 1)
    print(f"矩形函数面积: {area1}")  # 应该是1.0
    points2 = [[0, 0], [1, 1], [2, 0]]  # 三角形，面积应该是1
    area2 = integral_area(points2, 0, 2)
    print(f"三角形函数面积: {area2}")  # 应该是1.0
    
    # 测试 project_x 函数
    print("\n=== 测试 project_x 函数 ===")
    points3 = [[2, 1], [4, 3], [6, 2]]
    print(f"原始点列表: {points3}")
    projected = project_x(points3, 2, 6)
    print(f"投影到[0,1]: {projected}")
    # 应该是[[0.0, 1], [0.5, 3], [1.0, 2]]
    
    # 测试 p_to_p_dis 函数
    print("\n=== 测试 p_to_p_dis 函数 ===")
    list1 = [[0, 0], [1, 1]]
    list2 = [[0, 0.1], [1, 1.1]]
    p2p_dist = p_to_p_dis(list1, list2)
    print(f"点对点距离: {p2p_dist}")
    
    list3 = [[0, 0], [1, 1], [2, 2]]
    list4 = [[0, 0], [1, 1]]
    p2p_dist2 = p_to_p_dis(list3, list4)
    print(f"不同长度列表的点对点距离: {p2p_dist2}")
    
    # 测试 area_dis 函数
    print("\n=== 测试 area_dis 函数 ===")
    list5 = [[0, 1], [1, 1]]  # 面积为1的矩形
    list6 = [[0, 2], [1, 2]]  # 面积为2的矩形
    area_diff = area_dis(list5, list6)
    print(f"面积差异: {area_diff}")  # 应该是1.0
    
    list7 = [[0, 0], [1, 1], [2, 0]]  # 三角形，归一化后面积
    list8 = [[0, 0], [1, 0], [2, 0]]  # 零函数
    area_diff2 = area_dis(list7, list8)
    print(f"三角形与零函数的面积差异: {area_diff2}")
    
    # 测试 chebyshev_dis 函数
    print("\n=== 测试 chebyshev_dis 函数 ===")
    list9 = [[0, 0], [1, 1]]
    list10 = [[0, 0.5], [1, 1.5]]  # 与list9整体相差0.5
    cheb_dist = chebyshev_dis(list9, list10)
    print(f"Chebyshev距离 (整体偏移0.5): {cheb_dist}")  # 应该是0.5
    
    list11 = [[0, 0], [1, 1]]
    list12 = [[0, 0], [0.5, 2], [1, 0]]  # 在x=0.5处有较大差异
    cheb_dist2 = chebyshev_dis(list11, list12)
    print(f"Chebyshev距离 (峰值差异): {cheb_dist2}")
    
    # 测试 RMSE_dis 函数
    print("\n=== 测试 RMSE_dis 函数 ===")
    list13 = [[0, 0], [1, 0]]  # 零函数
    list14 = [[0, 0.1], [1, 0.1]]  # 常数0.1函数
    rmse1 = RMSE_dis(list13, list14)
    print(f"RMSE距离 (常数偏移0.1): {rmse1}")  # 应该接近0.1
    
    list15 = [[0, 0], [1, 1]]
    list16 = [[0, 0.5], [1, 0.5]]
    rmse2 = RMSE_dis(list15, list16)
    print(f"RMSE距离 (线性与常数): {rmse2}")
    
    print("\n所有测试完成！")

# 运行测试
if __name__ == "__main__":
    test_matrix()