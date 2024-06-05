import os
import platform
import GPUtil

class SystemInfo:
    """Capture and report system configuration and environment, including GPU details if available."""
    def __init__(self, environment):
        self.hardware = platform.processor()
        self.memory_usage = f"{os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024. ** 3):.2f} GB"  # Total memory in GB
        self.runtime_environment = environment
        self.gpu_info = self.get_gpu_info()

    def get_gpu_info(self):
        """Retrieve GPU information using GPUtil."""
        gpus = GPUtil.getGPUs()
        gpu_details = []
        for gpu in gpus:
            gpu_details.append({
                "gpu_id": gpu.id,
                "gpu_name": gpu.name,
                "gpu_load": f"{gpu.load * 100}%",
                "gpu_free_memory": f"{gpu.memoryFree} MB",
                "gpu_used_memory": f"{gpu.memoryUsed} MB",
                "gpu_total_memory": f"{gpu.memoryTotal} MB",
                "gpu_temperature": f"{gpu.temperature} C"
            })
        return gpu_details

    def get_system_info(self):
        """Return the system information as a dictionary, including GPU details."""
        return {
            "hardware": self.hardware,
            "memory_usage": self.memory_usage,
            "runtime_environment": self.runtime_environment,
            "gpu_info": self.gpu_info
        }

# Example usage
if __name__ == "__main__":
    system_info = SystemInfo("Local")
    print(system_info.get_system_info())
