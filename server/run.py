import test
import subprocess

PORT = 8080

def main():
    ngrok_process = subprocess.Popen(
        ["ngrok", "http", "--url=obviously-needed-crab.ngrok-free.app", str(PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if ngrok_process.poll() is not None:
        print("ngrok process failed to start")
        return
    
    print("ngrok process started successfully")
    test.run_server(PORT)




if __name__ == "__main__":
    main()