from datetime import datetime

class LogTimePos:
    def __init__(self):
        self.log_file = f"logs/log-{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}.txt"
        
        with open(self.log_file, "w+") as f:
            f.write("")

    def log_info(self, delta_time, pos):
        with open(self.log_file, "a") as f:
            f.write(str(delta_time) + " " + " ".join(map(str, pos)) + "\n")