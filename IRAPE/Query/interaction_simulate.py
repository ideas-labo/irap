import copy
import random

from IRAPE.base_op import BaseOp

"""
Given the initial state and target state, adjust the initial state toward the target state within the given number of operations,
output the operation sequence and the adjusted state. One query call is regarded as one session.
"""
def query(src: list, op_num: int) -> tuple:
    op_cnt = 0
    ops = []
    BaseOp.clear_state()
    
    # Use deep copy to maintain session isolation
    working_src = copy.deepcopy(src)
    BaseOp.gen_tag(working_src)

    print("\n" + "="*50)
    print("STARTING INTERACTIVE ADJUSTMENT SESSION")
    print("="*50)

    for step in range(op_num):
        # --- NEW: Dashboard to display current state ---
        print(f"\n>>> CURRENT STATE (Step {step + 1}/{op_num}):")
        print("-" * 30)
        print(f"{'Index':<8} | {'X (T/Delta)':<12} | {'Y (Sat.)':<10} | {'Tag':<10}")
        for idx, point in enumerate(working_src):
            # point format: [x, y, tag]
            tag_str = str(point[2]) if len(point) > 2 else "N/A"
            print(f"{idx:<8} | {point[0]:<12.4f} | {point[1]:<10.4f} | {tag_str:<10}")
        print("-" * 30)

        # --- Level 1: Interval to modify? ---
        if len(working_src) < 2:
            print("Warning: Fewer than 2 points available. Cannot form an interval.")
            if len(working_src) == 0: break
        else:
            print("\nAvailable Intervals:")
            for idx in range(len(working_src) - 1):
                p1, p2 = working_src[idx], working_src[idx+1]
                print(f"({idx+1}) Interval {idx}-{idx+1}: [{p1[0]:.2f}, {p2[0]:.2f}]")
        
        try:
            interval_choice = int(input("\n[Question 1] Select Interval ID to modify: ")) - 1
            l_idx = interval_choice
            r_idx = interval_choice + 1
            
            # Basic validation for interval index
            if l_idx < 0 or r_idx >= len(working_src):
                print("Invalid interval! Please restart this step.")
                continue

            # --- Level 2: Adjustment Intent? ---
            print("\n[Question 2] Adjustment Intent?")
            print(" (1) Adjust Precision (Add/Delete points)")
            print(" (2) Adjust Difficulty (Modify coordinates)")
            intent = input(" Selection: ")

            if intent == "1":
                # --- Level 3: Add or Delete? ---
                print("\n[Question 3] Add or Delete point?")
                print(" (1) Add point (\textsc{ADD})")
                print(" (2) Delete point (\textsc{REMOVE})")
                prec_choice = input(" Selection: ")

                if prec_choice == "1":
                    # Atomic: ADD
                    ops.append({"type": "add_point", "l": l_idx, "r": r_idx})
                    working_src = BaseOp.add_point(working_src, l_idx, r_idx)
                    BaseOp.gen_tag(working_src) # Regenerate tags for the new point
                    print(f"Action Success: Point added between {l_idx} and {r_idx}.")

                else:
                    # Level 4: Which endpoint?
                    print(f"\n[Question 4] Which endpoint to remove?")
                    print(f" (1) Left Endpoint (Index {l_idx})")
                    print(f" (2) Right Endpoint (Index {r_idx})")
                    end_choice = input(" Selection: ")
                    target_idx = l_idx if end_choice == "1" else r_idx
                    
                    # Atomic: REMOVE
                    ops.append({"type": "del_point", "loc": target_idx})
                    working_src = BaseOp.del_point(working_src, target_idx)
                    print(f"Action Success: Point at index {target_idx} removed.")

            elif intent == "2":
                # --- Level 3: Which endpoint (for Difficulty)? ---
                print(f"\n[Question 3] Which endpoint to adjust?")
                print(f" (1) Left Endpoint (Index {l_idx})")
                print(f" (2) Right Endpoint (Index {r_idx})")
                end_choice = input(" Selection: ")
                target_idx = l_idx if end_choice == "1" else r_idx

                # --- Level 4: x or y? ---
                print("\n[Question 4] Select Dimension:")
                print(" (1) x-axis (T or Delta)")
                print(" (2) y-axis (Satisfaction)")
                dim_choice = input(" Selection: ")
                is_x = True if dim_choice == "1" else False

                # --- Level 5: Increase or Decrease? ---
                print("\n[Question 5] Direction:")
                print(" (1) Increase (\textsc{CHANGE})")
                print(" (2) Decrease (\textsc{CHANGE})")
                dir_choice = input(" Selection: ")

                # Atomic: CHANGE (Increase/Decrease)
                if dir_choice == "1":
                    ops.append({"type": "increase", "loc": target_idx, "sign": is_x})
                    factor = BaseOp.get_step_size(working_src[target_idx][2], is_x, True)
                    working_src = BaseOp.increase(working_src, target_idx, is_x)
                    BaseOp.update_state(working_src[target_idx][2], is_x, True, factor)
                    print(f"Action Success: Increased {'x' if is_x else 'y'} at index {target_idx}.")
                else:
                    ops.append({"type": "decrease", "loc": target_idx, "sign": is_x})
                    factor = BaseOp.get_step_size(working_src[target_idx][2], is_x, False)
                    working_src = BaseOp.decrease(working_src, target_idx, is_x)
                    BaseOp.update_state(working_src[target_idx][2], is_x, False, factor)
                    print(f"Action Success: Decreased {'x' if is_x else 'y'} at index {target_idx}.")
            else:
                print("Invalid intent selection.")
                
        except (ValueError, IndexError):
            print("Error: Please enter a valid numerical option.")
            continue

    print("\n" + "="*50)
    print("SESSION COMPLETE. EXPORTING RESULTS...")
    print("="*50)
    
    BaseOp.clear_state()
    # Strip tags for the final result_src to match the original [x, y] format
    result_src = [[point[0], point[1]] for point in working_src]
    return ops, result_src


if __name__ == "__main__":
    src1 = [[1.0, 0.5], [3.0, 0.8]]
    print(f"Source list: {src1}")
    print(f"Target list: {dest1}")
    ops1, res1 = query(src1, 2)
    print(ops1)
    print(res1)