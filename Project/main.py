import torch
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessExitedException
import main_config_ssl as ssl

def find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def main():
    args = ssl.parse_args()
    args.port = find_free_port()  # Ensure a free port is assigned
    world_size = torch.cuda.device_count()

    try:
        mp.spawn(ssl.main_worker,
                 args=(world_size, args),
                 nprocs=world_size,
                 join=True)
    except ProcessExitedException as e:
        print(f"Process exited unexpectedly: {e}")

if __name__ == "__main__":
    main()



