import ctypes
import os
import time

import numpy as np

# sudo apt install libomp5 -y


# --- Configuration ---
LIB_PATH = "./without_simd/libtensorflow.so"
# LIB_PATH = "./with_simd/libtensorflow.so"
MODEL_PATH = "./frozen_graph.pb"
INPUT_NAME = "x"
OUTPUT_NAME = "Identity"
IMG_SHAPE = (1, 600, 600, 3)
RUNS = 20



# --- Check for library ---
if not os.path.exists(LIB_PATH):
    raise RuntimeError(f"TensorFlow library not found at {LIB_PATH}. Please ensure it's downloaded and in the correct location.")




# Load library
tf = ctypes.cdll.LoadLibrary(LIB_PATH)

# ---- Define required C types and function signatures ----

# Define the TF_Output struct which is required by TF_SessionRun
class TF_Output(ctypes.Structure):
    _fields_ = [("oper", ctypes.c_void_p),
                ("index", ctypes.c_int)]

# Status object functions
tf.TF_NewStatus.restype = ctypes.c_void_p
tf.TF_DeleteStatus.argtypes = [ctypes.c_void_p]
tf.TF_GetCode.argtypes = [ctypes.c_void_p]
tf.TF_GetCode.restype = ctypes.c_int
tf.TF_Message.argtypes = [ctypes.c_void_p]
tf.TF_Message.restype = ctypes.c_char_p

# Graph object functions
tf.TF_NewGraph.restype = ctypes.c_void_p
tf.TF_DeleteGraph.argtypes = [ctypes.c_void_p]
tf.TF_GraphOperationByName.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
tf.TF_GraphOperationByName.restype = ctypes.c_void_p

# CORRECTED: Added definitions for TF_ImportGraphDefOptions
tf.TF_NewImportGraphDefOptions.restype = ctypes.c_void_p
tf.TF_DeleteImportGraphDefOptions.argtypes = [ctypes.c_void_p]
tf.TF_GraphImportGraphDef.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

# Session object functions
tf.TF_NewSessionOptions.restype = ctypes.c_void_p
tf.TF_DeleteSessionOptions.argtypes = [ctypes.c_void_p]
tf.TF_NewSession.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
tf.TF_NewSession.restype = ctypes.c_void_p
tf.TF_DeleteSession.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

# Buffer object functions
tf.TF_NewBufferFromString.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
tf.TF_NewBufferFromString.restype = ctypes.c_void_p
tf.TF_DeleteBuffer.argtypes = [ctypes.c_void_p]

# Tensor object functions
tf.TF_NewTensor.restype = ctypes.c_void_p
tf.TF_DeleteTensor.argtypes = [ctypes.c_void_p]

# SessionRun function
tf.TF_SessionRun.argtypes = [
    ctypes.c_void_p,  # session
    ctypes.c_void_p,  # run_options
    ctypes.POINTER(TF_Output), ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,  # inputs
    ctypes.POINTER(TF_Output), ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,  # outputs
    ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,  # target_opers
    ctypes.c_void_p  # status
]

# ---- Helper Functions ----

def _no_op_deallocator(data, size, arg):
    """A no-op deallocator for TF_NewTensor to prevent double-freeing memory."""
    pass

# Keep a reference to the C-callable function to prevent garbage collection
C_NO_OP_DEALLOCATOR = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)(_no_op_deallocator)


def check_status(status):
    """Raise an exception if the TF_Status is not OK."""
    if tf.TF_GetCode(status) != 0: # 0 is TF_OK
        msg = tf.TF_Message(status)
        raise RuntimeError(f"TensorFlow C-API Error: {msg.decode('utf-8')}")

def load_graph(path, status):
    """Load a frozen graph definition from a .pb file."""
    graph = tf.TF_NewGraph()
    with open(path, "rb") as f:
        data = f.read()
    
    graph_def_buffer = tf.TF_NewBufferFromString(data, len(data))
    
    # CORRECTED: Explicitly create and delete import options for safety.
    # Passing null pointers for optional structs can be a source of crashes.
    import_options = tf.TF_NewImportGraphDefOptions()
    tf.TF_GraphImportGraphDef(graph, graph_def_buffer, import_options, status)
    tf.TF_DeleteImportGraphDefOptions(import_options)
    tf.TF_DeleteBuffer(graph_def_buffer)
    
    check_status(status)
    return graph

def make_tensor(array: np.ndarray):
    """Create a TF_Tensor from a numpy array."""
    array = np.ascontiguousarray(array, dtype=np.float32)
    
    dims = (ctypes.c_int64 * len(array.shape))(*array.shape)
    nbytes = array.nbytes

    # CORRECTED: Use data_as(c_void_p) for more explicit pointer casting.
    tensor = tf.TF_NewTensor(
        1,  # TF_FLOAT = 1
        dims,
        len(array.shape),
        array.ctypes.data_as(ctypes.c_void_p),
        nbytes,
        C_NO_OP_DEALLOCATOR,
        None
    )
    return tensor

# ---- Main Execution ----
status = tf.TF_NewStatus()
graph = None
session = None
opts = None
input_tensor = None
# Pointer to hold the output tensor from the session run
output_tensor_ptr = (ctypes.c_void_p * 1)() 

try:
    # ---- Setup ----
    graph = load_graph(MODEL_PATH, status)

    

    opts = tf.TF_NewSessionOptions()
    session = tf.TF_NewSession(graph, opts, status)
    check_status(status)

    input_op_ptr = tf.TF_GraphOperationByName(graph, INPUT_NAME.encode('utf-8'))
    output_op_ptr = tf.TF_GraphOperationByName(graph, OUTPUT_NAME.encode('utf-8'))

    if not input_op_ptr or not output_op_ptr:
        raise RuntimeError("Input or output op not found in graph. Check names.")

    input_op = TF_Output(oper=input_op_ptr, index=0)
    output_op = TF_Output(oper=output_op_ptr, index=0)

    image = np.random.rand(*IMG_SHAPE).astype(np.float32)
    input_tensor = make_tensor(image)

    inputs_array = (TF_Output * 1)(input_op)
    input_values_array = (ctypes.c_void_p * 1)(input_tensor)
    outputs_array = (TF_Output * 1)(output_op)
    
    # --- Benchmarking Loop ---
    print(f"Running benchmark for {RUNS} iterations...")
    times = []
    for i in range(RUNS):
        start = time.time()
        tf.TF_SessionRun(
            session,
            None, # run_options
            inputs_array, input_values_array, 1,
            outputs_array, output_tensor_ptr, 1,
            None, 0, None, # target_opers
            status
        )
        check_status(status)
        end = time.time()
        
        tf.TF_DeleteTensor(output_tensor_ptr[0])
        
        # Don't time the first run (warm-up)
        if i > 0:
            times.append(end - start)

    # --- Results ---
    if times:
        avg_time_ms = np.mean(times) * 1000
        std_dev_ms = np.std(times) * 1000
        print(f"Average time: {avg_time_ms:.2f} ms ± {std_dev_ms:.2f} ms")
    else:
        print("Not enough runs to calculate average time.")

finally:
    # ---- Cleanup ----
    print("Cleaning up TensorFlow resources...")
    if input_tensor:
        tf.TF_DeleteTensor(input_tensor)
    if session:
        tf.TF_DeleteSession(session, status)
        check_status(status)
    if opts:
        tf.TF_DeleteSessionOptions(opts)
    if graph:
        tf.TF_DeleteGraph(graph)
    if status:
        tf.TF_DeleteStatus(status)
    print("Cleanup complete.")


