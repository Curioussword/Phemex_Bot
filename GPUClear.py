import os
import subprocess

def get_bot_processes():
    """Identify processes specifically related to the bot."""
    try:
        # Use `ps` to list all processes and filter for bot-specific keywords
        result = subprocess.run(
            ["ps", "aux"], stdout=subprocess.PIPE, text=True
        )
        lines = result.stdout.splitlines()

        # Define keywords to identify bot-related processes
        keywords = [
            "/workspace/Phemex_Bot",  # General workspace path
            "main.py",               # Main script
            "state_manager.py",      # State management logic
            "bot.py",                # Bot execution logic
            "dqn_agent.py"           # DQN agent logic
        ]

        # Filter lines containing any of the keywords
        bot_processes = [
            line for line in lines if any(keyword in line for keyword in keywords)
        ]

        return bot_processes
    except Exception as e:
        print(f"[ERROR] Failed to fetch bot processes: {e}")
        return []

def kill_bot_processes(bot_processes):
    """Kill processes specifically related to the bot."""
    for process in bot_processes:
        try:
            # Extract the PID (second column in `ps aux` output)
            pid = int(process.split()[1])
            print(f"[INFO] Killing process with PID: {pid}")
            os.kill(pid, 9)  # Send SIGKILL signal
        except Exception as e:
            print(f"[ERROR] Failed to kill process: {e}")

def main():
    print("[INFO] Searching for bot-related processes...")
    bot_processes = get_bot_processes()

    if not bot_processes:
        print("[INFO] No bot-related processes found.")
        return

    print(f"[INFO] Found {len(bot_processes)} bot-related processes:")
    for process in bot_processes:
        print(process)

    confirm = input("Do you want to terminate these processes? (y/n): ").strip().lower()
    if confirm == "y":
        kill_bot_processes(bot_processes)
        print("[INFO] All specified processes have been terminated.")
    else:
        print("[INFO] No processes were terminated.")

if __name__ == "__main__":
    main()
