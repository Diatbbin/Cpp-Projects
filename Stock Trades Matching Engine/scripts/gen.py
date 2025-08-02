import random
import string
import sys

def generate_instrument():
    """Generate a random instrument (e.g., stock symbol)"""
    return ''.join(random.choices(string.ascii_uppercase, k=1))

def generate_price():
    return round(random.uniform(1.00, 100.00), 2)

def generate_count():
    """Generate a random count between 1 and 100"""
    return random.randint(1, 100)

def generate_thread_id(num_threads):
    return random.randint(0, num_threads - 1)

def generate_testcase(num_threads, num_testcases):
    testcases = []
    order_ids = {}  # map of order ID to thread ID
    used_order_ids = set()  # set of used order IDs
    thread_orders = {i: [] for i in range(num_threads)}  # map of thread ID to list of order IDs
    connected_threads = set()  # set of connected threads
    next_order_id = 1
    
    testcases.append(str(num_threads))
    
    # Connect all threads
    for thread_id in range(num_threads):
        testcases.append(f"{thread_id} o")  # open
        connected_threads.add(thread_id)
    
    for _ in range(num_testcases):
        thread_id = random.choice(list(connected_threads))
        action = random.choice(['B', 'S', 'C'])
        
        if action == 'B' or action == 'S':  # buy/sell
            order_id = next_order_id
            next_order_id += 1
            instrument = generate_instrument()
            price = generate_price()
            count = generate_count()
            testcases.append(f"{thread_id} {action} {order_id} {instrument} {price} {count}")
            order_ids[order_id] = thread_id
            thread_orders[thread_id].append(order_id)
        elif action == 'C':  # cancel
            if thread_orders[thread_id]:
                order_id = random.choice(thread_orders[thread_id])
                testcases.append(f"{thread_id} C {order_id}")
                thread_orders[thread_id].remove(order_id)
                del order_ids[order_id]
    
    # Disconnect all threads
    for thread_id in connected_threads:
        testcases.append(f"{thread_id} x")  # disconnect
    
    return '\n'.join(testcases)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 test_case.py <num_threads> <num_testcases>")
        sys.exit(1)
    
    try:
        num_threads = int(sys.argv[1])
        num_testcases = int(sys.argv[2])
        if num_threads <= 0 or num_testcases <= 0:
            print("Error: Number of threads and test cases must be positive integers.")
            sys.exit(1)
    except ValueError:
        print("Error: Number of threads and test cases must be integers.")
        sys.exit(1)
    
    print(generate_testcase(num_threads, num_testcases))
