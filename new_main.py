import os
import sys
import time
import json
import ctypes
import subprocess
import argparse
import numpy as np

# --- Configuration ---
MODEL_PATH = "./frozen_graph.pb"
INPUT_NAME = "x"
OUTPUT_NAME = "Identity"
IMG_SHAPE = (1, 600, 600, 3)
RUNS = 20


class TF_Output(ctypes.Structure):
    _fields_ = [("oper", ctypes.c_void_p),
                ("index", ctypes.c_int)]


def run_benchmark_for_lib(lib_path: str):
    # Reduce TF C++ log noise so child stdout stays clean for JSON
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    if not os.path.exists(lib_path):
        raise RuntimeError(f"TensorFlow library not found at {lib_path}.")

    tflib = ctypes.cdll.LoadLibrary(lib_path)

    # Status object functions
    tflib.TF_NewStatus.restype = ctypes.c_void_p
    tflib.TF_DeleteStatus.argtypes = [ctypes.c_void_p]
    tflib.TF_GetCode.argtypes = [ctypes.c_void_p]
    tflib.TF_GetCode.restype = ctypes.c_int
    tflib.TF_Message.argtypes = [ctypes.c_void_p]
    tflib.TF_Message.restype = ctypes.c_char_p

    # Graph object functions
    tflib.TF_NewGraph.restype = ctypes.c_void_p
    tflib.TF_DeleteGraph.argtypes = [ctypes.c_void_p]
    tflib.TF_GraphOperationByName.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    tflib.TF_GraphOperationByName.restype = ctypes.c_void_p

    # Import options
    tflib.TF_NewImportGraphDefOptions.restype = ctypes.c_void_p
    tflib.TF_DeleteImportGraphDefOptions.argtypes = [ctypes.c_void_p]
    tflib.TF_GraphImportGraphDef.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

    # Session object functions
    tflib.TF_NewSessionOptions.restype = ctypes.c_void_p
    tflib.TF_DeleteSessionOptions.argtypes = [ctypes.c_void_p]
    tflib.TF_NewSession.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    tflib.TF_NewSession.restype = ctypes.c_void_p
    tflib.TF_DeleteSession.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    # Buffer object functions
    tflib.TF_NewBufferFromString.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    tflib.TF_NewBufferFromString.restype = ctypes.c_void_p
    tflib.TF_DeleteBuffer.argtypes = [ctypes.c_void_p]

    # Tensor object functions
    tflib.TF_NewTensor.restype = ctypes.c_void_p
    tflib.TF_DeleteTensor.argtypes = [ctypes.c_void_p]

    # SessionRun function
    tflib.TF_SessionRun.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(TF_Output), ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
        ctypes.POINTER(TF_Output), ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
        ctypes.c_void_p
    ]

    def _no_op_deallocator(data, size, arg):
        pass

    c_no_op_deallocator = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)(_no_op_deallocator)

    def check_status(status):
        if tflib.TF_GetCode(status) != 0:
            msg = tflib.TF_Message(status)
            raise RuntimeError(f"TensorFlow C-API Error: {msg.decode('utf-8')}")

    def load_graph(path, status):
        graph = tflib.TF_NewGraph()
        with open(path, "rb") as f:
            data = f.read()
        graph_def_buffer = tflib.TF_NewBufferFromString(data, len(data))
        import_options = tflib.TF_NewImportGraphDefOptions()
        tflib.TF_GraphImportGraphDef(graph, graph_def_buffer, import_options, status)
        tflib.TF_DeleteImportGraphDefOptions(import_options)
        tflib.TF_DeleteBuffer(graph_def_buffer)
        check_status(status)
        return graph

    def make_tensor(array: np.ndarray):
        array = np.ascontiguousarray(array, dtype=np.float32)
        dims = (ctypes.c_int64 * len(array.shape))(*array.shape)
        nbytes = array.nbytes
        tensor = tflib.TF_NewTensor(
            1,
            dims,
            len(array.shape),
            array.ctypes.data_as(ctypes.c_void_p),
            nbytes,
            c_no_op_deallocator,
            None
        )
        return tensor

    status = tflib.TF_NewStatus()
    graph = None
    session = None
    opts = None
    input_tensor = None
    output_tensor_ptr = (ctypes.c_void_p * 1)()

    results = {
        "avg_ms": None,
        "std_ms": None,
        "runs": RUNS
    }

    try:
        graph = load_graph(MODEL_PATH, status)
        opts = tflib.TF_NewSessionOptions()
        session = tflib.TF_NewSession(graph, opts, status)
        check_status(status)

        input_op_ptr = tflib.TF_GraphOperationByName(graph, INPUT_NAME.encode('utf-8'))
        output_op_ptr = tflib.TF_GraphOperationByName(graph, OUTPUT_NAME.encode('utf-8'))
        if not input_op_ptr or not output_op_ptr:
            raise RuntimeError("Input or output op not found in graph. Check names.")

        input_op = TF_Output(oper=input_op_ptr, index=0)
        output_op = TF_Output(oper=output_op_ptr, index=0)

        image = np.random.rand(*IMG_SHAPE).astype(np.float32)
        input_tensor = make_tensor(image)

        inputs_array = (TF_Output * 1)(input_op)
        input_values_array = (ctypes.c_void_p * 1)(input_tensor)
        outputs_array = (TF_Output * 1)(output_op)

        times = []
        for i in range(RUNS):
            start = time.time()
            tflib.TF_SessionRun(
                session,
                None,
                inputs_array, input_values_array, 1,
                outputs_array, output_tensor_ptr, 1,
                None, 0, None,
                status
            )
            check_status(status)
            end = time.time()
            tflib.TF_DeleteTensor(output_tensor_ptr[0])
            if i > 0:
                times.append(end - start)

        if times:
            results["avg_ms"] = float(np.mean(times) * 1000)
            results["std_ms"] = float(np.std(times) * 1000)
        return results
    finally:
        if input_tensor:
            tflib.TF_DeleteTensor(input_tensor)
        if session:
            tflib.TF_DeleteSession(session, status)
            check_status(status)
        if opts:
            tflib.TF_DeleteSessionOptions(opts)
        if graph:
            tflib.TF_DeleteGraph(graph)
        if status:
            tflib.TF_DeleteStatus(status)


def discover_libraries():
    libs = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for folder in ["with_simd", "without_simd"]:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for name in sorted(os.listdir(folder_path)):
            if name == "libtensorflow.so" :
                libs.append((folder, name, os.path.join(folder_path, name)))
    return libs


def _run_child_mode(lib_path: str):
    # Child process: load exactly one TF library and emit JSON on stdout
    try:
        results = run_benchmark_for_lib(lib_path)
        # Ensure only JSON is printed to stdout
        print(json.dumps({
            "ok": True,
            "results": results
        }))
    except Exception as e:
        print(json.dumps({
            "ok": False,
            "error": str(e)
        }))


def _run_parent_mode():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}.")

    libs = discover_libraries()
    if not libs:
        print("No .so libraries found in with_simd/ and without_simd/.")
        return

    print(f"Discovered {len(libs)} libraries. Running {RUNS} iterations each (first is warm-up).\n")
    for folder, name, path in libs:
        print(f"=== {folder}/{name} ===")
        try:
            env = os.environ.copy()
            lib_dir = os.path.dirname(path)
            env["LD_LIBRARY_PATH"] = lib_dir + (":" + env["LD_LIBRARY_PATH"] if "LD_LIBRARY_PATH" in env and env["LD_LIBRARY_PATH"] else "")
            env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            # Invoke this script in child mode
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--child", path],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            # Try to parse the last non-empty line of stdout as JSON
            stdout_text = proc.stdout.decode("utf-8", errors="replace")
            last_line = ""
            for line in stdout_text.strip().splitlines():
                if line.strip():
                    last_line = line.strip()
            payload = json.loads(last_line) if last_line else {"ok": False, "error": "No output"}
            if not payload.get("ok"):
                err_snippet = payload.get("error") or proc.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(err_snippet or "Child process failed")
            res = payload["results"]
            if res["avg_ms"] is not None:
                print(f"Average time: {res['avg_ms']:.2f} ms ± {res['std_ms']:.2f} ms\n")
            else:
                print("Not enough runs to calculate average time.\n")
        except Exception as e:
            print(f"Error running benchmark for {folder}/{name}: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Isolated TensorFlow C API benchmark runner")
    parser.add_argument("--child", dest="child_lib", default=None, help="Run in child mode for a specific libtensorflow.so path")
    args = parser.parse_args()

    if args.child_lib:
        _run_child_mode(args.child_lib)
    else:
        _run_parent_mode()


if __name__ == "__main__":
    main()


