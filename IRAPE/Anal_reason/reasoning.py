from IRAPE.KM import *
import copy
from IRAPE.base_op import BaseOp

class Reasoner:

    @staticmethod
    def diff_reasoning(A_k : list, A_k_prime : list) -> list:
        """Difference reasoning function: given the initial and final states of the point list, return the state transition operation sequence"""

        # Perform optimal point matching
        _, match = KM(A_k, A_k_prime)
        ops = []

        # Create a copy
        points = copy.deepcopy(A_k)

        # Modify points
        for i in range(len(points)):
            matched_point = A_k_prime[match[i][1]]
            x1, y1 = matched_point[0], matched_point[1]
            x2, y2 = points[i][0], points[i][1]

            if x1 < x2: # Need to decrease
                ops.append({'type' : 'decrease', 'loc': i, 'sign' : True})
                points = BaseOp.decrease(points, i, True)
            elif x1 > x2: # Need to increase
                ops.append({'type' : 'increase', 'loc': i, 'sign' : True})
                points = BaseOp.increase(points, i, True)      

            if y1 < y2:
                ops.append({'type' : 'decrease', 'loc': i, 'sign' : False})
                points = BaseOp.decrease(points, i, False)  
            elif y1 > y2:
                ops.append({'type' : 'increase', 'loc': i, 'sign' : False})
                points = BaseOp.increase(points, i, False)  

        # Insert points
        LEN = len(A_k_prime) # Maximum number of rows in the matrix
        matrix = np.full((LEN, 4), -1, dtype=float) # All positions initialized to -1, indicating empty
        
        for i in range(len(A_k_prime)): # Fill columns 1 and 2 of the matrix
            matrix[i, 0] = A_k_prime[i][0]
            matrix[i, 1] = A_k_prime[i][1]

        for i in range(len(A_k)):
            matrix[match[i][1], 2] = A_k[i][0]
            matrix[match[i][1], 3] = A_k[i][1]

        current_row = -1
        for i in range(LEN):
            if matrix[i, 2] == -1 and matrix[i, 3] == -1: # Encounter empty row
                ops.append({'type' : 'add_point', 'l' : current_row, 'r' : current_row + 1})
                points = BaseOp.add_point(points, current_row, current_row + 1)
            current_row += 1

        # Adjust points after inserting new points
        # Modify points
        for i in range(len(points)):
            x1, y1 = A_k_prime[i][0], A_k_prime[i][1]
            x2, y2 = points[i][0], points[i][1]

            if x1 < x2: # Need to decrease
                ops.append({'type' : 'decrease', 'loc': i, 'sign' : True})
                points = BaseOp.decrease(points, i, True)
            elif x1 > x2: # Need to increase
                ops.append({'type' : 'increase', 'loc': i, 'sign' : True})
                points = BaseOp.increase(points, i, True)      

            if y1 < y2:
                ops.append({'type' : 'decrease', 'loc': i, 'sign' : False})
                points = BaseOp.decrease(points, i, False)  
            elif y1 > y2:
                ops.append({'type' : 'increase', 'loc': i, 'sign' : False})
                points = BaseOp.increase(points, i, False)    

        return ops

if __name__ == '__main__':
    A_k = [[5.4, 0.0], [6.0, 1.0]]
    A_k_prime = [[4.86, 0.0], [5.43, 0.5], [6.0, 1.0]]
    ops = Reasoner.diff_reasoning(A_k, A_k_prime)
    print(ops)