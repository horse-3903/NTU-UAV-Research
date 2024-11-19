class Task_Queue:
    def __init__(self) -> None:
        self.queue = []
        self.current_task = None        
        
    def append(self, name, task) -> None:
        self.queue.append({
            "name": name,
            "task": task
        })
        
        if len(self.queue) == 1:
            self.current_task = task
    
    def advance(self) -> None:
        self.queue.pop(0)
        self.current_task = self.queue[0].get("task")
        
    def run(self):
        self.current_task()