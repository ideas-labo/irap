"""
Basic adjustment tool with heuristic step size
Single modification does not maintain state records
Batch modification is regarded as one session and needs to maintain state records
"""

import copy
import random

class BaseOp:
    # Static member variable to store state shared by all instances
    state = {}  # Record the state of each session, mapping from tag => state, where state is a dictionary that records the historical operation status and current step size of the x-axis and y-axis of a point

    @staticmethod
    def update_state(tag: str, sign: bool, op: bool, step_size: float):
        """Update the state of a point, need to pass sign (True for updating x-axis, False for y-axis), 
        op (True for increase operation, False for decrease operation), and step_size (step size of current operation)"""

        # Initialize if it's the first update
        if not tag in BaseOp.state:
            BaseOp.state[tag] = {}
            BaseOp.state[tag]["x"] = {"step_size": 0.1, "last_op": True}
            BaseOp.state[tag]["y"] = {"step_size": 0.1, "last_op": True}

        # Update x-axis for current operation
        if sign:
            BaseOp.state[tag]["x"]["last_op"] = op
            BaseOp.state[tag]["x"]["step_size"] = step_size
        # Update y-axis for current operation
        else:
            BaseOp.state[tag]["y"]["last_op"] = op
            BaseOp.state[tag]["y"]["step_size"] = step_size

    @staticmethod
    def clear_state():
        """Clear historical operation state for a new session"""
        BaseOp.state = {}

    @staticmethod
    def get_step_size(tag: str, sign: bool, op: bool) -> float:
        """Get heuristic step size for the operation on the current point"""

        if tag in BaseOp.state:
            if sign:  # Operate on x-axis
                if op == BaseOp.state[tag]["x"]["last_op"]:  # Same operation type as last time
                    return BaseOp.state[tag]["x"]["step_size"]
                else:  # Different operation type from last time
                    return BaseOp.state[tag]["x"]["step_size"] / 2
            else:  # Operate on y-axis
                if op == BaseOp.state[tag]["y"]["last_op"]:  # Same operation type as last time
                    return BaseOp.state[tag]["y"]["step_size"]
                else:  # Different operation type from last time
                    return BaseOp.state[tag]["y"]["step_size"] / 2
        return 0.1 # Return default step size

    @staticmethod
    def gen_tag(points):
        """Randomly generate a unique identifier for each point"""
        for point in points:
            if len(point) < 3: # Add tag if the point doesn't have one
                tag = str(random.randint(0, 10**9))
                point.append(tag)

    @staticmethod
    def round_point(point):
        """Keep 3 decimal places for both horizontal and vertical coordinates of the point"""
        point[0] = round(point[0], 3)
        point[1] = round(point[1], 3)
        return point
    
    @staticmethod
    def round_points_list(points):
        """Keep 3 decimal places for horizontal and vertical coordinates of all points in the list"""
        for point in points:
            point[0] = round(point[0], 3)
            point[1] = round(point[1], 3)    
        return points

    @staticmethod
    def add_point(points: list, l: int, r: int) -> list:
        """
        Add a point to the list, inserted between l and r, return the modified list.
        - If l == -1 and r == 0, insert a point at the beginning of the list.
        - If l == n-1 and r == n (n is the length of the list), insert a point at the end of the list.
        - Otherwise, insert a point between index l and r with coordinates as the average of the two points.

        Args:
            points (list): List containing points, each point is a list with two elements [x, y].
            l (int): Index or special flag of the first point.
            r (int): Index or special flag of the second point.

        Returns:
            list: Modified list.
        """
        # Create a deep copy of the input list using copy.deepcopy to avoid affecting original data
        points_copy = copy.deepcopy(points)
        
        if not isinstance(points_copy, list):
            print(f"Warning: Input 'points' is not a list type: {type(points_copy)}")
            return points_copy
        n = len(points_copy)

        # Case 1: Insert point at the beginning (l == -1, r == 0)
        if l == -1 and r == 0:
            if n == 0:
                print("Warning: List is empty, cannot insert point at the beginning (no reference point available).")
                return points_copy
            ref_point = points_copy[0]
            if not isinstance(ref_point, (list, tuple)):
                print(f"Warning: Element '{ref_point}' at index 0 is not a valid point (list or tuple).")
                return points_copy
            new_x = ref_point[0] * 0.9
            new_y = ref_point[1]
            new_point = [round(new_x, 3), round(new_y, 3)]  # Keep 3 decimal places
            # Create new list and insert at the beginning
            new_points = [new_point] + points_copy
            # --- Modification: Sort by x coordinate ---
            new_points.sort(key=lambda p: p[0])
            return BaseOp.round_points_list(new_points)

        # Case 2: Insert point at the end (l == n-1, r == n)
        if l == n - 1 and r == n:
            if n == 0:
                print("Warning: List is empty, cannot insert point at the end (no reference point available).")
                return points_copy
            ref_point = points_copy[n-1]
            if not isinstance(ref_point, (list, tuple)):
                print(f"Warning: Element '{ref_point}' at index {n-1} is not a valid point (list or tuple).")
                return points_copy
            new_x = ref_point[0] * 1.1
            new_y = ref_point[1]
            new_point = [round(new_x, 3), round(new_y, 3)]  # Keep 3 decimal places
            # Create new list and append to the end
            new_points = points_copy + [new_point]
            # --- Modification: Sort by x coordinate ---
            new_points.sort(key=lambda p: p[0])
            return BaseOp.round_points_list(new_points)

        # Case 3: Insert point between l and r (standard case)
        if not (0 <= l < n and 0 <= r < n):
            print(f"Warning: Index l ({l}) or r ({r}) is out of range [0, {n-1}].")
            return points_copy

        # To ensure meaningful insertion order, we assume l < r
        # Swap l and r if l > r
        if l > r:
            l, r = r, l
        elif l == r:
            # Insert after l if l == r
            r = l + 1
            if r > n: # Prevent out of bounds
                print(f"Warning: Calculated insertion position r ({r}) is out of list range.")
                return points_copy

        point_l = points_copy[l]
        point_r = points_copy[r]

        if not isinstance(point_l, (list, tuple)):
            print(f"Warning: Element '{point_l}' at index l ({l}) is not a valid point (list or tuple).")
            return points_copy
        if not isinstance(point_r, (list, tuple)):
            print(f"Warning: Element '{point_r}' at index r ({r}) is not a valid point (list or tuple).")
            return points_copy

        # Calculate average value
        new_x = (point_l[0] + point_r[0]) / 2.0
        new_y = (point_l[1] + point_r[1]) / 2.0
        new_point = [round(new_x, 3), round(new_y, 3)]  # Keep 3 decimal places

        # Create a new list to avoid modifying the original list
        new_points = copy.deepcopy(points_copy)
        # Insert new point at position r
        new_points.insert(r, new_point)
        # --- Modification: Sort by x coordinate ---
        new_points.sort(key=lambda p: p[0])
        return BaseOp.round_points_list(new_points)

    @staticmethod
    def del_point(points: list, loc: int) -> list:
        """
        Delete the point at position loc and return the modified list.

        Args:
            points (list): List containing points.
            loc (int): Index of the point to be deleted.

        Returns:
            list: Modified list.
        """
        # Create a deep copy of the input list using copy.deepcopy to avoid affecting original data
        points_copy = copy.deepcopy(points)
        
        if not points_copy:
            print("Warning: List is empty, cannot delete point.")
            return points_copy

        if not (0 <= loc < len(points_copy)):
            print(f"Warning: Index loc ({loc}) is out of range [0, {len(points_copy)-1}].")
            return points_copy

        if not isinstance(points_copy[loc], (list, tuple)) :
            print(f"Warning: Element '{points_copy[loc]}' at index loc ({loc}) is not a valid point (list or tuple).")
            return points_copy

        # Create a new list to avoid modifying the original list
        new_points = copy.deepcopy(points_copy)
        # Delete the point at the specified index
        new_points.pop(loc)
        # --- Modification: Sort by x coordinate ---
        new_points.sort(key=lambda p: p[0])
        return BaseOp.round_points_list(new_points)

    @staticmethod
    def increase(points: list, loc: int, sign: bool) -> list:
        """
        Increase the horizontal coordinate (when sign=True) or vertical coordinate (when sign=False) 
        of the point at loc by 10%, return the modified list.

        Args:
            points (list): List containing points.
            loc (int): Index of the point to be modified.
            sign (bool): True to modify horizontal coordinate, False to modify vertical coordinate.

        Returns:
            list: Modified list.
        """
        
        if not points:
            print("Warning: List is empty, cannot modify point.")
            return points

        if not (0 <= loc < len(points)):
            print(f"Warning: Index loc ({loc}) is out of range [0, {len(points)-1}].")
            return points

        point_to_modify = points[loc]
        if not isinstance(point_to_modify, (list, tuple)):
            print(f"Warning: Element '{point_to_modify}' at index loc ({loc}) is not a valid point (list or tuple).")
            return points

        # Create a new list to avoid modifying the original list
        new_points = copy.deepcopy(points)
        new_points.sort(key=lambda p: p[0])  # Sort first to ensure correct order
        point = new_points[loc]

        if sign:  # Modify horizontal coordinate (x)
            # Calculate limit range
            min_val = new_points[loc-1][0] if loc > 0 else float('-inf')
            max_val = new_points[loc+1][0] if loc < len(new_points)-1 else float('inf')
            factor = BaseOp.get_step_size(new_points[loc][2], sign, True) if len(new_points[loc]) == 3 else 0.1 # Get current step size

            while True:
                new_x = point[0] * (1 + factor)
                if min_val <= new_x and new_x <= max_val:
                    point[0] = round(new_x, 3)  # Keep 3 decimal places
                    break
                factor /= 2

        else:  # Modify vertical coordinate (y)
            factor = BaseOp.get_step_size(new_points[loc][2], sign, True) if len(new_points[loc]) == 3 else 0.1 # Get current step size
            point[1] = round(max(0.0, min(point[1] * (1 + factor), 1.0)), 3)  # Keep 3 decimal places

        # Sort by x coordinate
        new_points.sort(key=lambda p: p[0])
        return BaseOp.round_points_list(new_points)

    @staticmethod
    def decrease(points: list, loc: int, sign: bool) -> list:
        """
        Decrease the horizontal coordinate (when sign=True) or vertical coordinate (when sign=False) 
        of the point at loc by 10%, return the modified list.

        Args:
            points (list): List containing points.
            loc (int): Index of the point to be modified.
            sign (bool): True to modify horizontal coordinate, False to modify vertical coordinate.

        Returns:
            list: Modified list.
        """
        
        if not points:
            print("Warning: List is empty, cannot modify point.")
            return points

        if not (0 <= loc < len(points)):
            print(f"Warning: Index loc ({loc}) is out of range [0, {len(points)-1}].")
            return points

        point_to_modify = points[loc]
        if not isinstance(point_to_modify, (list, tuple)):
            print(f"Warning: Element '{point_to_modify}' at index loc ({loc}) is not a valid point (list or tuple).")
            return points

        # Create a new list to avoid modifying the original list
        new_points = copy.deepcopy(points)
        new_points.sort(key=lambda p: p[0])  # Sort first to ensure correct order
        point = new_points[loc]

        if sign:  # Modify horizontal coordinate (x)
            # Calculate limit range
            min_val = new_points[loc-1][0] if loc > 0 else float('-inf')
            max_val = new_points[loc+1][0] if loc < len(new_points)-1 else float('inf')
            factor = BaseOp.get_step_size(new_points[loc][2], sign, False) if len(new_points[loc]) == 3 else 0.1 # Get current step size
        
            while True:
                new_x = point[0] * (1 - factor)
                if min_val <= new_x and new_x <= max_val:
                    point[0] = round(new_x, 3)  # Keep 3 decimal places
                    break
                factor /= 2
            
        else:  # Modify vertical coordinate (y)
            factor = BaseOp.get_step_size(new_points[loc][2], sign, False) if len(new_points[loc]) == 3 else 0.1# Get current step size
            point[1] = round(max(0.0, min(point[1] * (1 - factor), 1.0)), 3)# Keep 3 decimal places

        # Sort by x coordinate
        new_points.sort(key=lambda p: p[0])
        return BaseOp.round_points_list(new_points)
     
    @staticmethod
    def batch_exec(points: list, ops: list) -> list: 
        """
        Execute the above four basic operations according to the given operation sequence.

        Args:
            points (list): Initial list of points.
            ops (list): Operation sequence, each operation is a dictionary containing the "type" field.
                        Example: {"type": "increase", "loc": 0, "sign": True}
                        Supported types: "add_point", "del_point", "increase", "decrease"

        Returns:
            list: Final list of points after executing all operations.
        """
        # Create a copy to avoid modifying the original input
        current_points = copy.deepcopy(points)

        BaseOp.clear_state() # Clear session state

        for i, op in enumerate(ops):
            if not isinstance(op, dict):
                print(f"Warning: ops[{i}] is not a dictionary type: {op}. Skipping this operation.")
                continue

            op_type = op.get("type")
            if not op_type:
                print(f"Warning: ops[{i}] missing 'type' field: {op}. Skipping this operation.")
                continue

            # Call the corresponding static method according to op_type
            # All methods accept current_points as the first parameter and return the modified list
            if op_type == "add_point":
                l = op.get("l")
                r = op.get("r")
                if l is None or r is None:
                    print(f"Warning: ops[{i}] ('add_point') missing required parameters 'l' or 'r': {op}. Skipping this operation.")
                    continue
                # Call method and update current_points
                current_points = BaseOp.add_point(current_points, l, r)

            elif op_type == "del_point":
                loc = op.get("loc")
                if loc is None:
                    print(f"Warning: ops[{i}] ('del_point') missing required parameter 'loc': {op}. Skipping this operation.")
                    continue
                current_points = BaseOp.del_point(current_points, loc)

            elif op_type == "increase":
                loc = op.get("loc")
                sign = op.get("sign")
                if loc is None or sign is None:
                    print(f"Warning: ops[{i}] ('increase') missing required parameters 'loc' or 'sign': {op}. Skipping this operation.")
                    continue
                current_points = BaseOp.increase(current_points, loc, sign)

            elif op_type == "decrease":
                loc = op.get("loc")
                sign = op.get("sign")
                if loc is None or sign is None:
                    print(f"Warning: ops[{i}] ('decrease') missing required parameters 'loc' or 'sign': {op}. Skipping this operation.")
                    continue
                current_points = BaseOp.decrease(current_points, loc, sign)

            else:
                print(f"Warning: Unknown operation type '{op_type}' in ops[{i}]: {op}. Skipping this operation.")
                continue

        return [[point[0], point[1]] for point in current_points] # Return points without tags

# --- Test Code ---
if __name__ == "__main__":
    # Initial point set
    initial_points = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    n = len(initial_points)
    print(f"Initial point set: {initial_points}, length n: {n}\n")

    # 1. Test add_point - standard case
    print("--- Test add_point (Standard Case) ---")
    points_after_add_standard = BaseOp.add_point(initial_points, 0, 1)
    print(f"After adding point between index 0 and 1: {points_after_add_standard}")
    # Expected: [1.0, 2.0], [(1.0+5.0)/2, (2.0+6.0)/2] = [3.0, 4.0], [3.0, 4.0], [5.0, 6.0]

    # 2. Test add_point - insert at beginning
    print("\n--- Test add_point (Insert at beginning l=-1, r=0) ---")
    points_after_add_start = BaseOp.add_point(initial_points, -1, 0)
    print(f"After inserting point at beginning: {points_after_add_start}")
    # Expected: [1.0*0.9, 2.0] = [0.9, 2.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]

    # 3. Test add_point - insert at end
    print("\n--- Test add_point (Insert at end l=n-1, r=n) ---")
    points_after_add_end = BaseOp.add_point(initial_points, n-1, n)
    print(f"After inserting point at end: {points_after_add_end}")
    # Expected: [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [5.0*1.1, 6.0] = [5.5, 6.0]

    # 4. Test add_point - boundary case (empty list)
    print("\n--- Test add_point (Boundary Case) ---")
    empty_list = []
    print(f"Insert at beginning of empty list: {BaseOp.add_point(empty_list, -1, 0)}")
    print(f"Insert at end of empty list: {BaseOp.add_point(empty_list, -1, 0)}") # l, r do not meet end condition, standard logic will be tried
    print(f"Insert at end of empty list (correct parameters): {BaseOp.add_point(empty_list, -1, 0)}") # n=0, l=n-1=-1, r=n=0, meets beginning condition
    # Correction: Insert at end of empty list (n=0, l=n-1=0-1=-1, r=n=0 does not meet end condition l=n-1=0-1=-1, r=n=0, which is actually beginning)
    # To insert at end of empty list, need l=0-1=-1, r=0, which is same as beginning. So empty list can only insert once at beginning/end.
    # If list is empty, beginning and end are the same position.
    print(f"Insert at standard position (0, 0) of empty list: {BaseOp.add_point(empty_list, 0, 0)}") # Out of range

    # 5. Test del_point
    print("\n--- Test del_point ---")
    points_after_del = BaseOp.del_point(initial_points, 1)
    print(f"After deleting point at index 1: {points_after_del}")
    # Expected: [1.0, 2.0], [5.0, 6.0]

    # 6. Test increase
    print("\n--- Test increase ---")
    points_after_inc_x = BaseOp.increase(initial_points, 0, True) # Increase x coordinate of index 0
    print(f"After increasing x coordinate of index 0 by 10%: {points_after_inc_x}")
    # Expected: x = 1.0 * 1.10 = 1.10

    # 7. Test decrease
    print("\n--- Test decrease ---")
    points_after_dec_y = BaseOp.decrease(initial_points, 0, False) # Decrease y coordinate of index 0
    print(f"After decreasing y coordinate of index 0 by 10%: {points_after_dec_y}")
    # Expected: y = 2.0 * 0.90 = 1.8

    print(f"\nIs original point set changed: {initial_points}") # Verify original list is not modified

    # 8. Test batch_exec
    # Initial point set
    print("\n--- Test batch_exec ---")
    initial_points = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    print(f"Initial point set: {initial_points}\n")

    # Define operation sequence
    operations = [
        {"type": "increase", "loc": 0, "sign": True},  # Increase x coordinate of index 0
        {"type": "decrease", "loc": 0, "sign": True},  # Decrease x coordinate of index 0
        {"type": "increase", "loc": 0, "sign": True},  # Increase x coordinate of index 0
        {"type": "decrease", "loc": 1, "sign": False}, # Decrease y coordinate of index 1
        {"type": "add_point", "l": 1, "r": 2},         # Add point between index 1 and 2
        {"type": "del_point", "loc": 0},               # Delete point at index 0
        {"type": "add_point", "l": -1, "r": 0},        # Add point at beginning
        {"type": "add_point", "l": 3, "r": 4},         # Add point at end (note: list length has changed)
    ]

    print(f"Executing operation sequence: {operations}\n")

    # Execute batch operations
    final_points = BaseOp.batch_exec(initial_points, operations)

    print(f"Final point set: {final_points}")
    print(f"Is original point set changed: {initial_points}") # Verify original list is not modified