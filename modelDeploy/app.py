from bestmultithread.main import run
import multiprocessing as mp

def main():
    print("Starting model deployment...")
    mp.set_start_method("spawn", force=True)
    mp.freeze_support()
    run()

if __name__ == "__main__":
    main()