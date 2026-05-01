import numpy as np
import math

def standardization(f):
    if not f: return []
    tmp = np.array([data[0] for data in f]).reshape(-1, 1)
    median = np.median(tmp, axis=0)
    q75, q25 = np.percentile(tmp, [75, 25], axis=0)
    iqr = q75 - q25
    iqr = np.where(iqr == 0, 1, iqr)
    tmp = (tmp - median) / iqr
    res = [[tmp[i, 0], f[i][1]] for i in range(len(f))]
    return sorted(res, key=lambda x: x[0])

def dis(point1, point2):
    return math.fabs(point1[0] - point2[0])

def KM(list1, list2):
    n = len(list1)
    m = len(list2)

    # 1. Ensure left points count is less than or equal to right points count
    if n > m:
        weight, temp_matching = KM(list2, list1)
        result = [(-1, -1)] * n
        for j_idx, i_idx in temp_matching:
            if i_idx != -1:
                result[i_idx] = (i_idx, j_idx)
        return weight, result

    # 2. Local variables initialization (completely abandon global variables)
    graph = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            graph[i][j] = -dis(list1[i], list2[j])

    lx = np.max(graph, axis=1)
    ly = np.zeros(m)
    cx = np.full(n, -1, dtype=int)
    cy = np.full(m, -1, dtype=int)
    
    # Precision protection threshold
    EPS = 1e-12 

    def find_path(u, dx, dy, slack, m_count):
        dx[u] = True
        for v in range(m_count):
            if dy[v]: continue
            
            diff = lx[u] + ly[v] - graph[u][v]
            if abs(diff) < EPS:  # Use threshold to determine equality
                dy[v] = True
                if cy[v] == -1 or find_path(cy[v], dx, dy, slack, m_count):
                    cy[v] = u
                    cx[u] = v
                    return True
            else:
                if slack[v] > diff:
                    slack[v] = diff
        return False

    # 3. Main loop
    for i in range(n):
        slack = np.full(m, float('inf'))
        # Safety limit: The number of label adjustments for a single point should not exceed m (total right nodes)
        # Normally, KM algorithm should be able to increase a matching point within m adjustments
        safe_limit = m + 10 
        
        while safe_limit > 0:
            dx = np.zeros(n, dtype=bool)
            dy = np.zeros(m, dtype=bool)

            if find_path(i, dx, dy, slack, m):
                break

            # Calculate minimum adjustment
            # Find minimum slack only among unvisited y nodes
            unvisited_dy = ~dy
            if not np.any(unvisited_dy):
                break
                
            delta = np.min(slack[unvisited_dy])
            
            # If delta is abnormal (like inf or extremely small), force break
            if delta == float('inf') or delta < EPS:
                # Even if perfect matching isn't found, forcibly reduce lx slightly to break dead loop
                delta = EPS if delta < EPS else 0
                if delta == 0: break 

            # 4. Key update: Synchronize update of vertex labels and Slack
            for j in range(n):
                if dx[j]: lx[j] -= delta
            for j in range(m):
                if dy[j]: ly[j] += delta
                else: slack[j] -= delta
            
            safe_limit -= 1
        
        if safe_limit <= 0:
            # Still not matched successfully, for return value safety, we can manually perform an illegal but safe match
            # Or skip this point, total_weight calculation will ignore points with cx[i] == -1
            pass

    # 5. Calculate results
    total_weight = 0
    result = []
    for i in range(n):
        target = int(cx[i])
        if target != -1:
            total_weight += graph[i][target]
        result.append((i, target))

    return total_weight, result

# Example usage
def example_usage():
    # Example 1: Point matching problem
    print("=== Point matching problem ===")
    points1 = [(0, 0), (1, 1), (2, 2)]  # Left point set
    points2 = [(0.5, 0.5), (1.5, 1.5), (2.5, 2.5)]  # Right point set
    
    print("Left point set:", points1)
    print("Right point set:", points2)
    
    weight, matching = KM(points1, points2)
    
    print("Maximum matching weight:", -weight)  # Because we used negative distance
    print("Matching results:")
    for left_idx, right_idx in matching:
        print(f"  Left point {points1[left_idx]} -> Right point {points2[right_idx]}")

    """Test cases where the number of points on left and right sides are unequal"""
    print("\n=== Test cases where left points count is less than right points count ===")
    
    # Case 1: 3 points on left, 5 points on right
    print("Case 1: 3 points on left, 5 points on right")
    left_points = [(0, 0), (1, 1), (2, 2)]
    right_points = [(0.2, 0.1), (0.8, 1.2), (1.9, 2.1), (3, 3), (4, 4)]
    
    print("Left point set:", left_points)
    print("Right point set:", right_points)
    
    weight, matching = KM(left_points, right_points)
    
    print("Matching weight:", -weight)
    print("Matching results:")
    for left_idx, right_idx in matching:
        left_point = left_points[left_idx]
        right_point = right_points[right_idx]
        dist = dis(left_point, right_point)
        print(f"  {left_point} -> {right_point}, distance: {dist:.3f}")
    
    # Case 2: 2 points on left, 6 points on right
    print("\nCase 2: 2 points on left, 6 points on right")
    left_points = [(0, 0), (2, 2)]
    right_points = [(0.1, 0.1), (0.5, 0.5), (1.9, 1.8), (2.1, 2.2), (3, 1), (1, 3)]
    
    print("Left point set:", left_points)
    print("Right point set:", right_points)
    
    weight, matching = KM(left_points, right_points)
    
    print("Matching weight:", -weight)
    print("Matching results:")
    for left_idx, right_idx in matching:
        left_point = left_points[left_idx]
        right_point = right_points[right_idx]
        dist = dis(left_point, right_point)
        print(f"  {left_point} -> {right_point}, distance: {dist:.3f}")

    # Case 3: 6 points on left, 2 points on right
    left_points = [(0.1, 0.1), (0.5, 0.5), (1.9, 1.8), (2.1, 2.2), (3, 1), (1, 3)]
    right_points = [(0, 0), (2, 2)]
    print("\nLeft point set:", left_points)
    print("Right point set:", right_points)
    
    weight, matching = KM(left_points, right_points)
    
    print("Matching weight:", -weight)
    print("Matching results:")
    for left_idx, right_idx in matching:
        left_point = left_points[left_idx]
        if right_idx == -1:
            print(f"Point {left_point} has no match")
            continue
        right_point = right_points[right_idx]
        dist = dis(left_point, right_point)
        print(f"  {left_point} -> {right_point}, distance: {dist:.3f}")

if __name__ == "__main__":
    example_usage()